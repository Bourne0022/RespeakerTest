"""Record beamformed audio from a ReSpeaker XVF3800.

This script:
1. Configures the XVF3800 fixed beam to point at a user-defined angle (default 30°).
2. Routes the processed beamformed output to the left USB audio channel.
3. Records up to a configurable duration, or stops earlier on keypress.
4. Applies a smoothed RMS gate with attack/hold hysteresis to filter out
   transient noise and avoid chopping speech during brief pauses.
5. Writes the gated audio as a 16 kHz mono WAV file.

Dependencies:
    pip install pyusb sounddevice

On Windows, the official ReSpeaker control driver/libusb setup is required for
USB control commands. On Linux/macOS, libusb permissions must allow access to
the control interface.
"""

from __future__ import annotations

import array
from collections import deque
import math
import pathlib
import platform
import queue
import struct
import sys
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party imports – fail early with clear install instructions
# ---------------------------------------------------------------------------

try:
    import sounddevice as sd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: sounddevice.\n"
        "Install it with:  pip install sounddevice"
    ) from exc

try:
    import usb.core
    import usb.util
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pyusb.\n"
        "Install it with:  pip install pyusb"
    ) from exc

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import libusb_package  # type: ignore[import-untyped]
except Exception:
    libusb_package = None  # only needed on Windows


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VID = 0x2886
PID = 0x001A
SAMPLE_RATE = 16_000
CHANNELS = 2                      # stereo USB capture (left = beamformed, right = silent)
SAMPLE_WIDTH_BYTES = 2            # int16 PCM
BLOCKSIZE = 1024                  # frames per USB audio block
DEFAULT_DURATION_SEC = 30.0
DEFAULT_BEAM_AZIMUTH_DEG = 30.0
DEFAULT_RMS_THRESHOLD = 0.02
DEFAULT_ATTACK_MS = 40.0
DEFAULT_HOLD_MS = 400.0
DEFAULT_DENOISE = True
DEFAULT_DENOISE_STRENGTH = 0.65
DEFAULT_DENOISE_MIN_GAIN = 0.35
DEFAULT_RATIO_THRESHOLD = 0.0
DEFAULT_OFFAXIS_SUPPRESSION = True
DEFAULT_OFFAXIS_SUPPRESSION_STRENGTH = 0.80
DEFAULT_OFFAXIS_SUPPRESSION_MIN_GAIN = 0.18

# ---------------------------------------------------------------------------
# XVF3800 USB control parameter table
#
# Format: (resid, cmdid, value_count, access, value_type)
#
# Reference – official ReSpeaker XVF3800 Python host control SDK:
#   https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY
#
# Note on value counts:
#   The official SDK defines AEC_FIXEDBEAMSAZIMUTH_VALUES / _ELEVATION_VALUES
#   with count=4 (supporting up to 4 fixed beams).  This script configures
#   only 2 beams (primary + opposite), which matches the common use-case and
#   has been verified with firmware v2.1+.  AUDIO_MGR_OP_L / _OP_R use
#   count=2 (source index + option byte) as required by the DSP audio
#   manager register layout.
# ---------------------------------------------------------------------------

PARAMETERS = {
    "AEC_FIXEDBEAMSAZIMUTH_VALUES":  (33, 81, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSONOFF":           (33, 37, 1, "rw", "int32"),
    "AEC_FIXEDBEAMSGATING":          (33, 83, 1, "rw", "uint8"),
    "AUDIO_MGR_OP_L":                (35, 15, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R":                (35, 19, 2, "rw", "uint8"),
    "AEC_AZIMUTH_VALUES":            (33, 75, 4, "ro", "radians"),
    "AEC_SPENERGY_VALUES":           (33, 80, 4, "ro", "float"),
    "AUDIO_MGR_SELECTED_AZIMUTHS":   (35, 11, 2, "ro", "radians"),
    "DOA_VALUE":                     (20, 18, 2, "ro", "uint16"),
    "AEC_MIC_ARRAY_TYPE":            (33, 73, 1, "ro", "int32"),
}


# ======================================================================
# RmsGate – smoothed RMS gate with attack/hold hysteresis
# ======================================================================

class RmsGate:
    """Noise gate with EMA smoothing and hysteresis.

    Replaces a hard per-chunk threshold with three mechanisms that together
    produce smoother, more musical gating:

    1. **EMA smoothing** – each incoming RMS sample is blended into an
       exponential moving average so that isolated spikes (clicks, pops,
       door slams) don't immediately open the gate.

    2. **Attack time** – the smoothed signal must stay above the threshold
       for *attack_blocks* consecutive chunks before the gate opens.  This
       further rejects impulsive noise that survives the EMA.

    3. **Hold time** – once open, the gate stays open until the smoothed
       signal has been below the threshold for *hold_blocks* consecutive
       chunks.  This prevents the gate from fluttering closed during
       short inter-word or inter-syllable silences.

    Parameters
    ----------
    threshold : float
        Normalised RMS threshold (0.0 – 1.0).
    block_size : int
        Audio frames per chunk.
    sample_rate : int
        Sample rate in Hz.
    attack_ms : float
        Minimum above-threshold time before the gate opens (ms).
    hold_ms : float
        Minimum below-threshold time before the gate closes (ms).
    smooth_alpha : float
        EMA coefficient.  Smaller = heavier smoothing (0 < alpha ≤ 1).
    """

    def __init__(
        self,
        threshold: float,
        block_size: int = BLOCKSIZE,
        sample_rate: int = SAMPLE_RATE,
        attack_ms: float = DEFAULT_ATTACK_MS,
        hold_ms: float = DEFAULT_HOLD_MS,
        smooth_alpha: float = 0.15,
    ) -> None:
        if not 0.0 < smooth_alpha <= 1.0:
            raise ValueError("smooth_alpha must be in (0, 1]")
        if attack_ms < 0 or hold_ms < 0:
            raise ValueError("attack_ms and hold_ms must be non-negative")

        self.threshold = threshold
        # Convert ms → number of audio blocks
        sec_per_block = block_size / sample_rate
        self.attack_blocks = max(1, int(attack_ms / 1000 / sec_per_block))
        self.hold_blocks = max(1, int(hold_ms / 1000 / sec_per_block))
        self.alpha = smooth_alpha

        self._smoothed: float = 0.0
        self._above_streak: int = 0
        self._below_streak: int = 0
        self._open: bool = False

    # -- read-only properties ------------------------------------------------

    @property
    def open(self) -> bool:
        """Return True if the gate is currently passing audio."""
        return self._open

    @property
    def smoothed_rms(self) -> float:
        """Return the current EMA-smoothed RMS value."""
        return self._smoothed

    # -- core update ---------------------------------------------------------

    def update(self, rms: float) -> bool:
        """Feed a new instantaneous RMS value; return True if audio should pass.

        Call once per audio chunk.  The return value is deterministic and
        depends only on the history of *rms* values since construction.
        """
        # Exponential moving average – smooths out impulsive noise
        self._smoothed = self.alpha * rms + (1.0 - self.alpha) * self._smoothed

        if self._smoothed >= self.threshold:
            self._above_streak += 1
            self._below_streak = 0
        else:
            self._below_streak += 1
            self._above_streak = 0

        # Attack: open only after N consecutive above-threshold blocks.
        # The first few ms of speech onset are lost, but for close-talk
        # recording this is negligible (< 40 ms typical).
        if not self._open and self._above_streak >= self.attack_blocks:
            self._open = True

        # Hold: close only after N consecutive below-threshold blocks.
        # This bridges short pauses between words.
        elif self._open and self._below_streak >= self.hold_blocks:
            self._open = False

        return self._open


# ======================================================================
# SpatialMonitor – DOA + energy-based spatial verification
# ======================================================================

class SpatialMonitor:
    """Background DOA and speech-energy monitor for spatial filtering.

    Reads ``AUDIO_MGR_SELECTED_AZIMUTHS`` and ``AEC_SPENERGY_VALUES``
    from the XVF3800 on a background thread so the recording loop is
    never blocked by USB control-transfer latency.

    The main recording loop calls :meth:`check` to get the latest cached
    readings and a pass/fail verdict.

    Parameters
    ----------
    ctrl : ReSpeakerControl
        Connected control interface (must stay open for the monitor's lifetime).
    target_angle_deg : float
        Expected DOA in degrees.
    angle_tolerance_deg : float
        Allowed deviation from *target_angle_deg*.
    ref_energy : float or None
        Reference speech energy from calibration at the target distance.
        ``None`` disables the energy gate.
    energy_tolerance : float
        Allowed fractional drop below *ref_energy* (0.0 – 1.0).
    ratio_threshold : float
        Minimum beam power ratio E0/(E0+E1) for angle verification.
        Sound aligned with beam 0 produces ratio near 1.0; off-axis sound
        leaks into beam 1 and drops the ratio.
    """

    def __init__(
        self,
        ctrl: ReSpeakerControl,
        target_angle_deg: float,
        angle_tolerance_deg: float,
        ref_energy: Optional[float],
        energy_tolerance: float,
        ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    ) -> None:
        self._ctrl = ctrl
        self.target_angle = target_angle_deg
        self.angle_tolerance = angle_tolerance_deg
        self.ref_energy = ref_energy
        self.energy_tolerance = energy_tolerance
        self.ratio_threshold = ratio_threshold

        self._lock = threading.Lock()
        self._latest_doa: Optional[float] = None       # radians or NaN
        self._latest_energy0: Optional[float] = None    # beam-0 (target) speech energy
        self._latest_energy1: Optional[float] = None    # beam-1 (opposite) speech energy
        self._latest_energy2: Optional[float] = None    # beam-2 (target+90°) speech energy
        self._latest_energy3: Optional[float] = None    # beam-3 (target-90°) speech energy
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # -- background thread ----------------------------------------------------

    def start(self, interval: float = 0.25) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._poll, args=(interval,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _poll(self, interval: float) -> None:
        while self._running:
            try:
                doa_vals = self._ctrl.read(
                    "AUDIO_MGR_SELECTED_AZIMUTHS", retry=True, max_retries=5
                )
                if doa_vals is not None and len(doa_vals) >= 2:
                    with self._lock:
                        self._latest_doa = doa_vals[0]
            except Exception:
                pass

            try:
                energy_vals = self._ctrl.read(
                    "AEC_SPENERGY_VALUES", retry=True, max_retries=5
                )
                if energy_vals is not None and len(energy_vals) >= 4:
                    with self._lock:
                        self._latest_energy0 = energy_vals[0]
                        self._latest_energy1 = energy_vals[1]
                        self._latest_energy2 = energy_vals[2]
                        self._latest_energy3 = energy_vals[3]
            except Exception:
                pass

            time.sleep(interval)

    # -- main-loop API --------------------------------------------------------

    def check(self):
        """Return ``(doa_ok, energy_ok, ratio_ok, doa_deg, energy, ratio)``.

        *doa_deg* is ``None`` when no valid DOA has been read yet.
        *energy* is E0 (target beam speech energy), ``None`` when unavailable.
        *ratio* is the 4-beam focus ratio E0/sum(E0..E3), ``None`` when
        fewer than 4 energy values are available.

        All gates default to ``True`` when data is unavailable so that
        transient read failures never discard audio.
        """
        with self._lock:
            doa = self._latest_doa
            e0 = self._latest_energy0
            e1 = self._latest_energy1
            e2 = self._latest_energy2
            e3 = self._latest_energy3

        doa_ok = True
        energy_ok = True
        ratio_ok = True
        doa_deg: Optional[float] = None
        energy: Optional[float] = None
        ratio: Optional[float] = None

        if doa is not None and not math.isnan(doa):
            doa_deg = math.degrees(doa)
            diff = abs(doa_deg - self.target_angle)
            if diff > 180.0:
                diff = 360.0 - diff
            doa_ok = diff <= self.angle_tolerance

        if e0 is not None:
            energy = e0
            # Energy gate: lower-bound using calibration reference at target distance.
            if self.ref_energy is not None and self.ref_energy > 0.0:
                min_energy = self.ref_energy * max(0.0, 1.0 - self.energy_tolerance)
                energy_ok = e0 >= min_energy

            # 4-beam focus ratio + dominance check.
            # Beam 0 (target) must dominate the side-guard beams (2, 3) and
            # the total-energy focus ratio must meet the threshold.
            if e1 is not None and e2 is not None and e3 is not None:
                total = e0 + e1 + e2 + e3
                if total > 0.0:
                    ratio = e0 / total
                    # Focus: E0 must hold at least ratio_threshold of total energy.
                    # Dominance: E0 must be >= both side beams (2, 3).
                    # (We don't require E0 >= E1 because E1 points backwards.)
                    ratio_ok = (
                        ratio >= self.ratio_threshold
                        and e0 >= e2
                        and e0 >= e3
                    )

        return doa_ok, energy_ok, ratio_ok, doa_deg, energy, ratio


# ======================================================================
# SpatialMonitor: official DOA_VALUE voice + 4-beam focus check
# ======================================================================

class SpatialMonitor:
    """Monitor official voice/DOA values and beam-energy focus.

    The distance gate is based on calibrated RMS, not raw USB energy, so the
    runtime comparison stays in the same scale as the calibration reference.
    """

    def __init__(
        self,
        ctrl: ReSpeakerControl,
        target_angle_deg: float,
        angle_tolerance_deg: float,
        ref_rms: Optional[float],
        distance_ratio: float,
        ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    ) -> None:
        self._ctrl = ctrl
        self.target_angle = target_angle_deg
        self.angle_tolerance = angle_tolerance_deg
        self.ref_rms = ref_rms
        self.distance_ratio = distance_ratio
        self.ratio_threshold = ratio_threshold

        self._lock = threading.Lock()
        self._latest_doa_deg: Optional[float] = None
        self._latest_speech: Optional[bool] = None
        self._latest_energy0: Optional[float] = None
        self._latest_energy1: Optional[float] = None
        self._latest_energy2: Optional[float] = None
        self._latest_energy3: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self, interval: float = 0.20) -> None:
        self._running = True
        self._thread = threading.Thread(
            target=self._poll, args=(interval,), daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _poll(self, interval: float) -> None:
        while self._running:
            doa_value_ok = False
            try:
                doa_value = self._ctrl.read("DOA_VALUE", retry=True, max_retries=5)
                if doa_value is not None and len(doa_value) >= 2:
                    with self._lock:
                        self._latest_doa_deg = float(int(doa_value[0]) % 360)
                        self._latest_speech = bool(int(doa_value[1]))
                    doa_value_ok = True
            except Exception:
                pass

            if not doa_value_ok:
                try:
                    doa_vals = self._ctrl.read(
                        "AUDIO_MGR_SELECTED_AZIMUTHS", retry=True, max_retries=5
                    )
                    if doa_vals is not None and len(doa_vals) >= 1:
                        doa_rad = doa_vals[0]
                        if doa_rad is not None and not math.isnan(doa_rad):
                            with self._lock:
                                self._latest_doa_deg = math.degrees(doa_rad) % 360.0
                except Exception:
                    pass

            try:
                energy_vals = self._ctrl.read(
                    "AEC_SPENERGY_VALUES", retry=True, max_retries=5
                )
                if energy_vals is not None and len(energy_vals) >= 4:
                    with self._lock:
                        self._latest_energy0 = energy_vals[0]
                        self._latest_energy1 = energy_vals[1]
                        self._latest_energy2 = energy_vals[2]
                        self._latest_energy3 = energy_vals[3]
            except Exception:
                pass

            time.sleep(interval)

    def check(self, rms_level: Optional[float] = None):
        """Return speech/angle/energy/focus verdicts for the latest readings."""
        with self._lock:
            doa_deg = self._latest_doa_deg
            speech = self._latest_speech
            e0 = self._latest_energy0
            e1 = self._latest_energy1
            e2 = self._latest_energy2
            e3 = self._latest_energy3

        speech_ok = bool(speech)
        doa_ok = False
        energy_ok = True
        ratio_ok = True
        energy: Optional[float] = None
        ratio: Optional[float] = None

        if doa_deg is not None:
            diff = abs(doa_deg - self.target_angle)
            if diff > 180.0:
                diff = 360.0 - diff
            doa_ok = diff <= self.angle_tolerance

        if rms_level is not None:
            energy = rms_level
            if self.ref_rms is not None and self.ref_rms > 0.0:
                min_energy = self.ref_rms * max(0.0, self.distance_ratio)
                energy_ok = rms_level >= min_energy

            if (
                e1 is not None
                and e2 is not None
                and e3 is not None
                and (e0 + e1 + e2 + e3) > 0.0
            ):
                total = e0 + e1 + e2 + e3
                ratio = e0 / total
                if self.ratio_threshold > 0.0:
                    ratio_ok = (
                        ratio >= self.ratio_threshold
                        and e0 >= e2
                        and e0 >= e3
                    )

        return speech_ok, doa_ok, energy_ok, ratio_ok, doa_deg, energy, ratio


# ======================================================================
# Calibration helpers
# ======================================================================

def _save_calibration(
    path: str,
    target_angle_deg: float,
    target_distance_m: float,
    ref_energy: float,
    ref_rms: float,
) -> None:
    import json as _json
    from datetime import datetime as _datetime
    data = {
        "target_angle_deg": target_angle_deg,
        "target_distance_m": target_distance_m,
        "ref_energy": round(ref_energy, 6),
        "ref_rms": round(ref_rms, 6),
        "calibrated_at": _datetime.now().isoformat(),
    }
    with open(path, "w") as fh:
        _json.dump(data, fh, indent=2)
    print(f"Calibration saved → {path}")


def _load_calibration(path: str) -> dict:
    import json as _json
    with open(path, "r") as fh:
        return _json.load(fh)

# ======================================================================
# ReSpeakerControl – minimal XVF3800 USB parameter read/write
# ======================================================================

def _float_from_bytes(data: bytes) -> float:
    """Unpack a little-endian IEEE-754 float from 4 bytes."""
    return struct.unpack("<f", data)[0]


def _float_to_bytes(value: float) -> bytes:
    """Pack a float into 4 little-endian bytes."""
    return struct.pack("<f", value)

class ReSpeakerControl:
    """Minimal USB control wrapper for the XVF3800.

    Implements vendor-specific control transfers for reading and writing
    DSP parameters (beam angles, audio routing, AEC settings).

    Protocol reference:
      bmRequestType = CTRL_OUT | TYPE_VENDOR | RECIP_DEVICE  (0x40) for writes
      bmRequestType = CTRL_IN  | TYPE_VENDOR | RECIP_DEVICE  (0xC0) for reads
      wValue        = cmdid                                   (write)
                    = 0x80 | cmdid                            (read)
      wIndex        = resid
      data          = little-endian payload

    This matches the official xvf_host.py implementation.
    """

    TIMEOUT_MS = 100_000

    def __init__(self, dev: usb.core.Device) -> None:
        self.dev = dev
        self._claim_interface()

    # -- interface claiming --------------------------------------------------

    def _claim_interface(self) -> None:
        """Claim the XVF3800 vendor control interface (interface 3)."""
        interface_num = 3
        try:
            if self.dev.is_kernel_driver_active(interface_num):
                self.dev.detach_kernel_driver(interface_num)
        except (usb.core.USBError, NotImplementedError):
            pass  # Windows doesn't have kernel drivers for this interface

        try:
            usb.util.claim_interface(self.dev, interface_num)
        except usb.core.USBError as exc:
            raise RuntimeError(
                "Cannot claim USB control interface 3.\n"
                "• On Linux: run with sudo or add a udev rule for VID 0x2886.\n"
                "• On Windows: verify the libusb/WinUSB driver is installed\n"
                "  on the XVF3800 control interface (use Zadig if needed)."
            ) from exc

    # -- write ---------------------------------------------------------------

    def write(self, name: str, values: list[float | int]) -> None:
        """Write parameter values via vendor control transfer."""
        resid, cmdid, count, access, value_type = PARAMETERS[name]

        if access == "ro":
            raise ValueError(f"'{name}' is read-only – cannot write.")
        if len(values) != count:
            raise ValueError(
                f"'{name}' expects {count} value(s), got {len(values)}."
            )

        payload = self._encode_payload(values, value_type)

        try:
            self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,          # wValue reserved
                cmdid,      # wValue – command id
                resid,      # wIndex – resource id
                payload,    # data – parameter value(s)
                self.TIMEOUT_MS,
            )
        except usb.core.USBError as exc:
            raise RuntimeError(
                f"USB write failed for '{name}': {exc}\n"
                "The device may be disconnected or in use by another process."
            ) from exc

    # -- read ----------------------------------------------------------------

    def read(self, name: str, retry: bool = True, max_retries: int = 30):
        """Read parameter values via vendor control transfer.

        When *retry* is True (default), the method automatically retries on
        SERVICER_COMMAND_RETRY (status 64) to work around transient firmware
        busy states.
        """
        resid, cmdid, count, access, value_type = PARAMETERS[name]

        if access == "wo":
            raise ValueError(f"'{name}' is write-only – cannot read.")

        read_cmdid = 0x80 | cmdid
        data_len = self._read_length(count, value_type)

        for attempt in range(max_retries if retry else 1):
            try:
                response = self.dev.ctrl_transfer(
                    usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0,              # wValue reserved
                    read_cmdid,     # wValue – read-flagged command id
                    resid,          # wIndex – resource id
                    data_len,       # wLength – expected response size
                    self.TIMEOUT_MS,
                )
            except usb.core.USBError as exc:
                raise RuntimeError(
                    f"USB read failed for '{name}': {exc}\n"
                    "The device may have been disconnected."
                ) from exc

            status = response[0]
            if status == 0:  # CONTROL_SUCCESS
                return self._decode_response(response, count, value_type)
            if status == 64:  # SERVICER_COMMAND_RETRY
                import time
                time.sleep(0.01)
                continue
            # Unknown non-zero status – brief wait and retry
            if retry:
                import time
                time.sleep(0.01)
                continue
            break

        raise RuntimeError(
            f"USB read failed for '{name}' after {max_retries} retries "
            f"(firmware busy or parameter unavailable)."
        )

    # -- encode / decode internals -------------------------------------------

    @staticmethod
    def _encode_payload(values: list[float | int], value_type: str) -> bytearray:
        """Serialize a list of Python values into a little-endian USB payload."""
        payload = bytearray()
        if value_type in ("float", "radians"):
            for v in values:
                payload += _float_to_bytes(float(v))
        elif value_type == "uint8":
            for v in values:
                payload += int(v).to_bytes(1, "little", signed=False)
        elif value_type == "int32":
            for v in values:
                payload += int(v).to_bytes(4, "little", signed=True)
        else:
            raise ValueError(f"Unsupported write type: {value_type}")
        return payload

    @staticmethod
    def _read_length(count: int, value_type: str) -> int:
        """Compute USB read transfer length (payload + 1-byte status header)."""
        if value_type in ("uint8", "char"):
            return count + 1
        if value_type in ("float", "radians", "uint32", "int32"):
            return count * 4 + 1
        if value_type == "uint16":
            return count * 2 + 1
        raise ValueError(f"Unsupported read type: {value_type}")

    @staticmethod
    def _decode_response(response, count: int, value_type: str):
        """Parse USB response bytes into a tuple of Python values.

        The first byte is a status/header byte and is discarded.
        """
        raw = response.tobytes()
        if value_type == "uint16":
            return tuple(
                int.from_bytes(raw[i : i + 2], "little")
                for i in range(1, len(raw), 2)
            )
        if value_type in ("float", "radians"):
            return tuple(
                _float_from_bytes(raw[i : i + 4])
                for i in range(1, len(raw), 4)
            )
        if value_type == "uint8":
            return tuple(response.tolist()[1:])
        if value_type == "int32":
            return tuple(
                int.from_bytes(raw[i : i + 4], "little", signed=True)
                for i in range(1, len(raw), 4)
            )
        raise ValueError(f"Unsupported read type: {value_type}")

    # -- teardown ------------------------------------------------------------

    def close(self) -> None:
        """Release the USB interface and device resources."""
        try:
            usb.util.release_interface(self.dev, 3)
        except usb.core.USBError:
            pass
        usb.util.dispose_resources(self.dev)


# ======================================================================
# Device discovery
# ======================================================================

def find_device(vid: int = VID, pid: int = PID) -> Optional[usb.core.Device]:
    """Locate the XVF3800 USB control interface.

    On Windows, libusb_package.find() picks up the bundled libusb DLL so
    the user doesn't need to install a system-wide driver manually.
    On other platforms the system libusb is used directly.
    """
    if platform.system().lower().startswith("win") and libusb_package is not None:
        return libusb_package.find(idVendor=vid, idProduct=pid)
    return usb.core.find(idVendor=vid, idProduct=pid)


# ======================================================================
# Angle conversion
# ======================================================================

def _radians(deg: float) -> float:
    return deg * math.pi / 180.0


# ======================================================================
# Audio input device selection
# ======================================================================

def pick_input_device(device_hint: Optional[str]) -> Optional[int]:
    """Return the sounddevice index of the first matching USB input device.

    When *device_hint* is None, the function searches for common ReSpeaker /
    USB audio device name fragments.
    """

    devices = sd.query_devices()
    hints = (device_hint,) if device_hint else (
        "reSpeaker", "respeaker", "XVF3800", "xvf3800",
        "USB Audio", "USB Audio CODEC",
    )

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        name = dev["name"]
        if any(h.lower() in name.lower() for h in hints):
            return idx
    return None


# ======================================================================
# Audio processing helpers
# ======================================================================

def rms_int16_mono(raw_bytes: bytes, channels: int = CHANNELS) -> float:
    """Compute normalised RMS (0.0–1.0) from interleaved int16 PCM bytes.

    Only the first channel is used for the RMS calculation.
    """

    samples = array.array("h")
    samples.frombytes(raw_bytes)

    if sys.byteorder != "little":
        samples.byteswap()

    mono = samples[0::channels] if channels > 1 else samples
    n = len(mono)
    if n == 0:
        return 0.0

    acc = sum(s * s for s in mono)
    return math.sqrt(acc / n) / 32768.0


def extract_mono_bytes(raw_bytes: bytes, channels: int = CHANNELS) -> bytes:
    """Extract the first channel from interleaved int16 PCM as raw bytes."""

    samples = array.array("h")
    samples.frombytes(raw_bytes)

    if sys.byteorder != "little":
        samples.byteswap()

    mono = samples[0::channels] if channels > 1 else samples
    return mono.tobytes()


def suppress_offaxis_bytes(
    raw_bytes: bytes,
    channels: int = CHANNELS,
    strength: float = DEFAULT_OFFAXIS_SUPPRESSION_STRENGTH,
    min_gain: float = DEFAULT_OFFAXIS_SUPPRESSION_MIN_GAIN,
) -> tuple[bytes, bool, float]:
    """Return target-channel PCM with opposite/reference speech softened.

    The USB left channel is the target fixed beam. When enabled, the right
    channel is routed to the opposite fixed beam and used only as a reference.
    Frequencies that are stronger in the reference beam are attenuated in the
    target beam; target-dominant frequencies are preserved.
    """

    if np is None or channels < 2 or strength <= 0.0:
        return extract_mono_bytes(raw_bytes, channels=channels), False, 0.0

    samples = np.frombuffer(raw_bytes, dtype="<i2")
    if samples.size < channels * 16:
        return extract_mono_bytes(raw_bytes, channels=channels), False, 0.0

    frames = samples[: (samples.size // channels) * channels].reshape(-1, channels)
    target = frames[:, 0].astype(np.float32) / 32768.0
    reference = frames[:, 1].astype(np.float32) / 32768.0

    target_rms = float(np.sqrt(np.mean(target * target))) if target.size else 0.0
    ref_rms = float(np.sqrt(np.mean(reference * reference))) if reference.size else 0.0
    if ref_rms < 0.002 or target_rms <= 0.0:
        return frames[:, 0].astype("<i2").tobytes(), False, ref_rms

    relative_ref = ref_rms / max(target_rms, 1e-6)
    if relative_ref < 0.18:
        return frames[:, 0].astype("<i2").tobytes(), False, ref_rms

    n = target.size
    target_spec = np.fft.rfft(target)
    ref_spec = np.fft.rfft(reference)
    target_mag = np.abs(target_spec)
    ref_mag = np.abs(ref_spec)
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE)

    speech_band = (freqs >= 140.0) & (freqs <= 4200.0)
    reference_share = ref_mag / (target_mag + ref_mag + 1e-8)
    dominance = np.clip((reference_share - 0.52) / 0.40, 0.0, 1.0)

    chunk_strength = min(max(strength, 0.0), 1.0)
    if relative_ref < 0.75:
        chunk_strength *= 0.55
    elif relative_ref > 1.25:
        chunk_strength = min(1.0, chunk_strength * 1.15)

    gain = np.ones_like(target_mag, dtype=np.float32)
    speech_gain = 1.0 - chunk_strength * dominance[speech_band]
    speech_floor = min(max(min_gain, 0.05), 0.80)
    gain[speech_band] = np.maximum(speech_gain, speech_floor)

    cleaned = np.fft.irfft(target_spec * gain, n=n).astype(np.float32)
    peak_in = float(np.max(np.abs(target))) if target.size else 0.0
    peak_out = float(np.max(np.abs(cleaned))) if cleaned.size else 0.0
    if peak_in > 0.0 and peak_out > peak_in:
        cleaned *= peak_in / peak_out

    pcm = np.clip(cleaned * 32767.0, -32768, 32767).astype("<i2")
    suppressed = bool(np.any((dominance[speech_band] > 0.15) & (gain[speech_band] < 0.92)))
    return pcm.tobytes(), suppressed, ref_rms


@dataclass
class DenoiseResult:
    applied: bool = False
    noise_rms: float = 0.0
    peak_rms_after: float = 0.0


def light_denoise_wav(
    path: str,
    strength: float = DEFAULT_DENOISE_STRENGTH,
    min_gain: float = DEFAULT_DENOISE_MIN_GAIN,
) -> DenoiseResult:
    """Apply a conservative offline spectral gate to the mono WAV at *path*.

    This is intentionally light: it estimates a noise profile from the quietest
    saved frames, subtracts part of that profile, and keeps a higher gain floor
    in the speech band so the target voice is not chopped into syllables.
    """

    if np is None:
        print("Denoise skipped: numpy is not installed.", file=sys.stderr)
        return DenoiseResult(applied=False)

    wav_path = pathlib.Path(path)
    if not wav_path.exists() or wav_path.stat().st_size == 0:
        return DenoiseResult(applied=False)

    with wave.open(str(wav_path), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        raw = reader.readframes(frame_count)

    if channels != 1 or sample_width != SAMPLE_WIDTH_BYTES or sample_rate != SAMPLE_RATE:
        print("Denoise skipped: WAV format is not 16 kHz mono int16.", file=sys.stderr)
        return DenoiseResult(applied=False)

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size < 512:
        return DenoiseResult(applied=False)

    frame_size = 512
    hop = 256
    total_frames = max(1, int(math.ceil(max(0, samples.size - frame_size) / hop)) + 1)
    padded_len = (total_frames - 1) * hop + frame_size
    padded = np.zeros(padded_len, dtype=np.float32)
    padded[: samples.size] = samples

    window = np.hanning(frame_size).astype(np.float32)
    spectra = []
    frame_rms = []
    for start in range(0, padded_len - frame_size + 1, hop):
        frame = padded[start : start + frame_size]
        frame_rms.append(float(np.sqrt(np.mean(frame * frame))))
        spectra.append(np.fft.rfft(frame * window))

    rms_values = np.asarray(frame_rms, dtype=np.float32)
    quiet_count = max(3, min(len(rms_values), int(math.ceil(len(rms_values) * 0.20))))
    quiet_idx = np.argsort(rms_values)[:quiet_count]
    noise_mag = np.median(
        np.abs(np.asarray([spectra[int(i)] for i in quiet_idx])),
        axis=0,
    )
    noise_rms = float(np.median(rms_values[quiet_idx]))

    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    gain_floor = np.full_like(freqs, min_gain, dtype=np.float32)
    speech_band = (freqs >= 120.0) & (freqs <= 3800.0)
    gain_floor[speech_band] = np.maximum(gain_floor[speech_band], 0.55)
    gain_floor[freqs < 80.0] = np.minimum(gain_floor[freqs < 80.0], 0.20)

    out = np.zeros(padded_len, dtype=np.float32)
    norm = np.zeros(padded_len, dtype=np.float32)
    eps = 1e-8
    strength = min(max(strength, 0.0), 1.0)

    for idx, start in enumerate(range(0, padded_len - frame_size + 1, hop)):
        spec = spectra[idx]
        mag = np.abs(spec)
        gain = 1.0 - strength * (noise_mag / (mag + eps))
        gain = np.clip(gain, gain_floor, 1.0)
        clean = np.fft.irfft(spec * gain, n=frame_size).astype(np.float32)
        out[start : start + frame_size] += clean * window
        norm[start : start + frame_size] += window * window

    out = out / np.maximum(norm, eps)
    out = out[: samples.size]

    original_peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    clean_peak = float(np.max(np.abs(out))) if out.size else 0.0
    if original_peak > 0.0 and clean_peak > original_peak:
        out *= original_peak / clean_peak

    peak_after = float(np.sqrt(np.max(out * out))) if out.size else 0.0
    pcm = np.clip(out * 32767.0, -32768, 32767).astype("<i2")

    with wave.open(str(wav_path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(SAMPLE_WIDTH_BYTES)
        writer.setframerate(SAMPLE_RATE)
        writer.writeframes(pcm.tobytes())

    return DenoiseResult(applied=True, noise_rms=noise_rms, peak_rms_after=peak_after)


# ======================================================================
# Keyboard listener (cross-platform)
# ======================================================================

def _any_key_stop(stop_event: threading.Event) -> None:
    """Block until the user presses a key, then set *stop_event*.

    Safe to call from a daemon thread.  On non-interactive terminals
    (piped stdin) this returns immediately without blocking.
    """

    if sys.stdin is None or not sys.stdin.isatty():
        return

    if platform.system().lower().startswith("win"):
        import msvcrt
        msvcrt.getch()
        stop_event.set()
        return

    # Unix: set terminal to raw mode and poll with select()
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except termios.error:
        return

    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                sys.stdin.read(1)
                stop_event.set()
                break
    except (termios.error, OSError, ValueError):
        pass
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except termios.error:
            pass


# ======================================================================
# Recording statistics
# ======================================================================

@dataclass
class RecordStats:
    received_chunks: int = 0
    saved_chunks: int = 0
    received_frames: int = 0
    saved_frames: int = 0
    peak_rms: float = 0.0
    speech_rejects: int = 0
    doa_rejects: int = 0
    energy_rejects: int = 0
    ratio_rejects: int = 0
    speech_samples: int = 0
    denoise_applied: bool = False
    denoise_noise_rms: float = 0.0
    denoise_peak_rms: float = 0.0
    offaxis_suppressed_chunks: int = 0
    offaxis_reference_peak_rms: float = 0.0
    doa_samples: list[float] = None  # type: ignore[assignment]
    energy_samples: list[float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.doa_samples is None:
            self.doa_samples = []
        if self.energy_samples is None:
            self.energy_samples = []


# ======================================================================
# Beam configuration with read-back verification
# ======================================================================

def configure_device(ctrl: ReSpeakerControl, beam_deg: float) -> None:
    """Configure fixed-beam mode and route beamformed audio to left channel.

    After writing each parameter group the function reads the values back
    and compares them, printing warnings on mismatch so the user can
    diagnose configuration problems before recording starts.

    Audio routing:
      Left  channel → source 6 (beamformer output 0, the beam at *beam_deg*)
      Right channel → source 0 (silence)
    """

    beam_rad = _radians(beam_deg)
    opposite_rad = _radians((beam_deg + 180.0) % 360.0)
    side1_rad = _radians((beam_deg + 90.0) % 360.0)
    side2_rad = _radians((beam_deg - 90.0) % 360.0)

    print(f"Configuring fixed beam: azimuth={beam_deg:.1f}°, elevation=0°")
    print(f"  Beam 0 → {beam_deg:.1f}° (target, audio out)")
    print(f"  Beam 1 → {(beam_deg + 180) % 360:.1f}° (opposite)")
    print(f"  Beam 2 → {(beam_deg + 90) % 360:.1f}° (side guard)")
    print(f"  Beam 3 → {(beam_deg - 90) % 360:.1f}° (side guard)")

    # -- write beam parameters (4 beams: target, opposite, +90°, -90°) --------
    ctrl.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [beam_rad, opposite_rad, side1_rad, side2_rad])
    ctrl.write("AEC_FIXEDBEAMSELEVATION_VALUES", [0.0, 0.0, 0.0, 0.0])
    ctrl.write("AEC_FIXEDBEAMSGATING", [0])           # disable per-beam gating
    ctrl.write("AEC_FIXEDBEAMSONOFF", [1])             # enable fixed-beam mode

    # Route processed beamformer output to left channel; silence right.
    ctrl.write("AUDIO_MGR_OP_L", [6, 0])
    ctrl.write("AUDIO_MGR_OP_R", [0, 0])

    # -- read-back verification ----------------------------------------------
    try:
        azimuths = ctrl.read("AEC_FIXEDBEAMSAZIMUTH_VALUES", retry=True, max_retries=10)
    except RuntimeError:
        # Read-back may fail on some firmware versions; the write usually
        # succeeds anyway.  Fall back to the read-only azimuth register.
        try:
            azimuths = ctrl.read("AEC_AZIMUTH_VALUES", retry=True, max_retries=10)
        except RuntimeError:
            azimuths = None
            print("  Note: could not read back azimuths (firmware busy) – "
                  "writes typically succeed.", file=sys.stderr)

    try:
        elevations = ctrl.read("AEC_FIXEDBEAMSELEVATION_VALUES", retry=True, max_retries=10)
    except RuntimeError:
        elevations = None

    try:
        enabled = ctrl.read("AEC_FIXEDBEAMSONOFF", retry=True, max_retries=10)
    except RuntimeError:
        enabled = None

    try:
        op_l = ctrl.read("AUDIO_MGR_OP_L", retry=True, max_retries=10)
    except RuntimeError:
        op_l = None

    if azimuths is not None and len(azimuths) >= 4:
        az_ok = all(abs(a - e) < 0.1 for a, e in zip(azimuths, [beam_rad, opposite_rad, side1_rad, side2_rad]))
    elif azimuths is not None and len(azimuths) >= 2:
        az_ok = all(abs(a - e) < 0.1 for a, e in zip(azimuths[:2], [beam_rad, opposite_rad]))
    else:
        az_ok = True  # can't verify, assume OK

    if elevations is not None and len(elevations) >= 4:
        el_ok = all(abs(e) < 0.1 for e in elevations[:4])
    elif elevations is not None and len(elevations) >= 2:
        el_ok = all(abs(e) < 0.1 for e in elevations[:2])
    else:
        el_ok = True

    en_ok = (enabled == (1,)) if enabled is not None else True
    route_ok = (op_l == (6, 0)) if op_l is not None else True

    if not az_ok:
        print(f"  WARNING: azimuth mismatch – wrote {[beam_rad, opposite_rad, side1_rad, side2_rad]}, "
              f"read {azimuths}", file=sys.stderr)
    if not el_ok:
        print(f"  WARNING: elevation mismatch – wrote [0.0, 0.0, 0.0, 0.0], "
              f"read {elevations}", file=sys.stderr)
    if not en_ok:
        print(f"  WARNING: fixed-beam mode may be OFF "
              f"(read AEC_FIXEDBEAMSONOFF={enabled})", file=sys.stderr)
    if not route_ok:
        print(f"  WARNING: audio routing mismatch – wrote [6, 0], "
              f"read {op_l}", file=sys.stderr)

    if az_ok and el_ok and en_ok and route_ok:
        print("  Beam configuration verified OK.")

    # -- optional diagnostics ------------------------------------------------
    try:
        ae = ctrl.read("AEC_AZIMUTH_VALUES")
        sp = ctrl.read("AEC_SPENERGY_VALUES")
        print(f"  AEC_AZIMUTH_VALUES : {ae}")
        print(f"  AEC_SPENERGY_VALUES: {sp}")
    except RuntimeError:
        pass


# ======================================================================
# Official 2-beam fixed-beam configuration
# ======================================================================

def configure_device(
    ctrl: ReSpeakerControl,
    beam_deg: float,
    reference_beam: bool = False,
) -> None:
    """Configure two fixed beams and optionally route the opposite beam to USB R."""

    beam_rad = _radians(beam_deg)
    opposite_rad = _radians((beam_deg + 180.0) % 360.0)

    print(f"Configuring fixed beam: azimuth={beam_deg:.1f}°, elevation=0°")
    print(f"  Beam 0 -> {beam_deg:.1f}° (target, audio out)")
    print(f"  Beam 1 -> {(beam_deg + 180.0) % 360.0:.1f}° (opposite/reference)")

    ctrl.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [beam_rad, opposite_rad])
    ctrl.write("AEC_FIXEDBEAMSELEVATION_VALUES", [0.0, 0.0])
    ctrl.write("AEC_FIXEDBEAMSGATING", [0 if reference_beam else 1])
    ctrl.write("AEC_FIXEDBEAMSONOFF", [1])
    ctrl.write("AUDIO_MGR_OP_L", [6, 0])
    ctrl.write("AUDIO_MGR_OP_R", [6, 1] if reference_beam else [0, 0])

    try:
        azimuths = ctrl.read("AEC_FIXEDBEAMSAZIMUTH_VALUES", retry=True, max_retries=10)
    except RuntimeError:
        azimuths = None
    try:
        elevations = ctrl.read("AEC_FIXEDBEAMSELEVATION_VALUES", retry=True, max_retries=10)
    except RuntimeError:
        elevations = None
    try:
        enabled = ctrl.read("AEC_FIXEDBEAMSONOFF", retry=True, max_retries=10)
    except RuntimeError:
        enabled = None
    try:
        gating = ctrl.read("AEC_FIXEDBEAMSGATING", retry=True, max_retries=10)
    except RuntimeError:
        gating = None
    try:
        op_l = ctrl.read("AUDIO_MGR_OP_L", retry=True, max_retries=10)
    except RuntimeError:
        op_l = None
    try:
        op_r = ctrl.read("AUDIO_MGR_OP_R", retry=True, max_retries=10)
    except RuntimeError:
        op_r = None

    az_ok = True
    if azimuths is not None and len(azimuths) >= 2:
        az_ok = all(abs(a - e) < 0.1 for a, e in zip(azimuths[:2], [beam_rad, opposite_rad]))
    el_ok = True
    if elevations is not None and len(elevations) >= 2:
        el_ok = all(abs(e) < 0.1 for e in elevations[:2])
    en_ok = (enabled == (1,)) if enabled is not None else True
    expected_gating = (0,) if reference_beam else (1,)
    expected_op_r = (6, 1) if reference_beam else (0, 0)
    gating_ok = (gating == expected_gating) if gating is not None else True
    route_ok = (op_l == (6, 0)) if op_l is not None else True
    route_ok = route_ok and ((op_r == expected_op_r) if op_r is not None else True)

    if not az_ok:
        print(f"  WARNING: azimuth mismatch - wrote {[beam_rad, opposite_rad]}, read {azimuths}", file=sys.stderr)
    if not el_ok:
        print(f"  WARNING: elevation mismatch - wrote [0.0, 0.0], read {elevations}", file=sys.stderr)
    if not en_ok:
        print(f"  WARNING: fixed-beam mode may be OFF (read AEC_FIXEDBEAMSONOFF={enabled})", file=sys.stderr)
    if not gating_ok:
        print(f"  WARNING: fixed-beam gating may be OFF (read AEC_FIXEDBEAMSGATING={gating})", file=sys.stderr)
    if not route_ok:
        print(
            f"  WARNING: audio routing mismatch - wrote L=[6, 0], R={list(expected_op_r)}, "
            f"read L={op_l}, R={op_r}",
            file=sys.stderr,
        )

    if az_ok and el_ok and en_ok and gating_ok and route_ok:
        print("  Beam configuration verified OK.")
    if reference_beam:
        print("  USB right channel carries opposite-beam reference for off-axis suppression.")

    try:
        doa = ctrl.read("DOA_VALUE", retry=True, max_retries=5)
        sp = ctrl.read("AEC_SPENERGY_VALUES", retry=True, max_retries=5)
        print(f"  DOA_VALUE         : {doa}  (angle_deg, speech_flag)")
        print(f"  AEC_SPENERGY_VALUES: {sp}")
    except RuntimeError:
        pass


# ======================================================================
# Main recording routine
# ======================================================================

def record(
    output_path: str,
    duration_sec: float = DEFAULT_DURATION_SEC,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    device_hint: Optional[str] = None,
    beam_deg: float = DEFAULT_BEAM_AZIMUTH_DEG,
    attack_ms: float = DEFAULT_ATTACK_MS,
    hold_ms: float = DEFAULT_HOLD_MS,
    angle_tolerance_deg: float = 25.0,
    ref_rms: Optional[float] = None,
    distance_ratio: float = 0.30,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    enable_spatial: bool = True,
    stop_event: Optional[threading.Event] = None,
    trigger_on_voice: bool = False,
) -> RecordStats:
    """Run the recording session.

    Parameters
    ----------
    output_path : str
        Path for the output mono WAV file (16 kHz int16).
    duration_sec : float
        Maximum recording duration in seconds.
    rms_threshold : float
        Normalised RMS threshold for the noise gate (0.0 – 1.0).
    device_hint : str or None
        Substring to match against audio input device names.
    beam_deg : float
        Fixed-beam azimuth in degrees.
    attack_ms : float
        Gate attack time in ms.
    hold_ms : float
        Gate hold time in ms.
    angle_tolerance_deg : float
        Allowed DOA deviation from *beam_deg* before a chunk is rejected.
    ref_rms : float or None
        Reference RMS from calibration at the target distance.  ``None``
        disables the distance gate.
    distance_ratio : float
        Minimum fraction of *ref_rms* required to pass the distance gate.
    enable_spatial : bool
        When False the spatial monitor is not started (plain RMS-only gate).
    stop_event : threading.Event or None
        Optional external stop signal, used by GUI frontends.
    trigger_on_voice : bool
        When True, the WAV file is opened only after the first chunk that
        passes the RMS/DOA/energy checks.  If no qualifying speech is seen,
        no WAV file is created.

    Returns
    -------
    RecordStats
    """

    # ---- 1. locate USB control device -------------------------------------
    dev = find_device()
    if dev is None:
        raise RuntimeError(
            "No XVF3800 control device found (VID=0x2886 PID=0x001A).\n"
            "• Verify the microphone USB cable is connected.\n"
            "• On Windows: install the libusb/WinUSB driver on the control\n"
            "  interface using Zadig or the official ReSpeaker driver package.\n"
            "• On Linux: check 'lsusb' and ensure you have rw permissions."
        )

    ctrl = ReSpeakerControl(dev)
    spatial: Optional[SpatialMonitor] = None
    wf: Optional[wave.Wave_write] = None
    try:
        # ---- 2. configure beam & routing ----------------------------------
        configure_device(ctrl, beam_deg)

        # ---- 3. select audio input device ---------------------------------
        input_index = pick_input_device(device_hint)
        if input_index is None:
            devices = sd.query_devices()
            details = "\n".join(
                f"  [{i}] {d['name']}  "
                f"(inputs={d['max_input_channels']}, sr={int(d['default_samplerate'])} Hz)"
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            )
            raise RuntimeError(
                "No matching USB audio input device found.\n"
                "Use --device-hint with a substring of the device name, "
                "or pass an index via the environment variable "
                "SD_INPUT_DEVICE_INDEX.\n\n"
                f"Available input devices:\n{details}"
            )

        dev_info = sd.query_devices(input_index)
        print(f"Audio input : [{input_index}] {dev_info['name']}")
        print(f"Sample rate : {SAMPLE_RATE} Hz  |  channels: {CHANNELS} (stereo)")

        # ---- 4. initialise RMS gate ---------------------------------------
        gate = RmsGate(
            threshold=rms_threshold,
            block_size=BLOCKSIZE,
            sample_rate=SAMPLE_RATE,
            attack_ms=attack_ms,
            hold_ms=hold_ms,
        )

        # ---- 4b. initialise spatial monitor (DOA + energy) ---------------
        if enable_spatial:
            spatial = SpatialMonitor(
                ctrl=ctrl,
                target_angle_deg=beam_deg,
                angle_tolerance_deg=angle_tolerance_deg,
                ref_rms=ref_rms,
                distance_ratio=distance_ratio,
                ratio_threshold=ratio_threshold,
            )
            spatial.start(interval=0.25)
            if ref_rms is not None:
                print(f"Spatial gate : DOA ±{angle_tolerance_deg:.0f}°  |  "
                      f"RMS >= {distance_ratio * 100:.0f}% of ref "
                      f"(ref={ref_rms:.4f})  |  focus >= {ratio_threshold:.2f}")
            else:
                print(f"Spatial gate : DOA ±{angle_tolerance_deg:.0f}°  |  "
                      f"RMS gate DISABLED  |  focus >= {ratio_threshold:.2f}")

        # ---- 5. keyboard listener (daemon thread) -------------------------
        if stop_event is None:
            stop_event = threading.Event()
        keyboard_stop_enabled = sys.stdin is not None and sys.stdin.isatty()
        if keyboard_stop_enabled:
            kb_thread = threading.Thread(
                target=_any_key_stop, args=(stop_event,), daemon=True
            )
            kb_thread.start()

        # ---- 6. audio queue & callback ------------------------------------
        q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        stats = RecordStats()
        t_start = time.monotonic()
        last_status_time = t_start

        def callback(indata, frames, _time_info, status) -> None:
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            try:
                q.put_nowait(bytes(indata))
            except queue.Full:
                pass  # drop the newest chunk when the main loop falls behind

        # ---- 7. record ----------------------------------------------------
        print(f"\nRecording → {output_path}")
        print(f"Beam azimuth  : {beam_deg:.1f}°")
        print(f"RMS threshold : {rms_threshold:.4f}  "
              f"(attack={attack_ms:.0f} ms, hold={hold_ms:.0f} ms)")
        if duration_sec > 0:
            print(f"Max duration  : {duration_sec:.1f} s")
        else:
            print("Max duration  : unlimited")
        if keyboard_stop_enabled:
            print("Press any key to stop early.\n")
        else:
            print("Use the GUI stop button or an external stop signal to stop early.\n")

        def _ensure_wave_open() -> wave.Wave_write:
            nonlocal wf
            if wf is None:
                wf = wave.open(output_path, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(SAMPLE_RATE)
            return wf

        if not trigger_on_voice:
            _ensure_wave_open()

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCKSIZE,
            device=input_index,
            callback=callback,
        ):
            while True:
                # -- stop conditions --
                if stop_event.is_set():
                    print("\nStopped by user request.")
                    break

                elapsed = time.monotonic() - t_start
                if duration_sec > 0 and elapsed >= duration_sec:
                    print(f"\nReached max duration ({duration_sec:.1f} s).")
                    break

                # -- fetch next audio block --
                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                stats.received_chunks += 1
                stats.received_frames += len(chunk) // (SAMPLE_WIDTH_BYTES * CHANNELS)

                # -- RMS gate decision --
                level = rms_int16_mono(chunk, channels=CHANNELS)
                if level > stats.peak_rms:
                    stats.peak_rms = level

                if gate.update(level):
                    # -- spatial filter (DOA + energy + ratio) --
                    write_chunk = True
                    if spatial is not None:
                        doa_ok, energy_ok, ratio_ok, doa_deg, energy, ratio = spatial.check()
                        if doa_deg is not None:
                            stats.doa_samples.append(doa_deg)
                        if energy is not None:
                            stats.energy_samples.append(energy)
                        if not doa_ok:
                            stats.doa_rejects += 1
                            write_chunk = False
                        if not energy_ok:
                            stats.energy_rejects += 1
                            write_chunk = False
                        if not ratio_ok:
                            stats.ratio_rejects += 1
                            write_chunk = False

                    if write_chunk:
                        if trigger_on_voice and wf is None:
                            _ensure_wave_open()
                            print("检测到符合条件的人声，开始写入 WAV。")
                        mono = extract_mono_bytes(chunk, channels=CHANNELS)
                        if wf is not None:
                            wf.writeframesraw(mono)
                            stats.saved_chunks += 1
                            stats.saved_frames += len(mono) // SAMPLE_WIDTH_BYTES

                # -- periodic status (every ~2 s) --
                now = time.monotonic()
                if now - last_status_time >= 2.0:
                    state = "OPEN" if gate.open else "CLOSED"
                    parts = [
                        f"  [{elapsed:5.1f}s]  ",
                        f"RMS={level:.4f} (smoothed={gate.smoothed_rms:.4f})  ",
                        f"gate={state}",
                    ]
                    if spatial is not None:
                        doa_ok, energy_ok, ratio_ok, doa_deg, energy_val, ratio_val = spatial.check()
                        if doa_deg is not None:
                            tag = "OK" if doa_ok else "OFF-AXIS"
                            parts.append(f"  DOA={doa_deg:.1f}° {tag}")
                        else:
                            parts.append("  DOA=--")
                        if ratio_val is not None:
                            tag = "OK" if ratio_ok else "LOW"
                            parts.append(f"  focus={ratio_val:.2f} {tag}")
                        parts.append(f"  saved={stats.saved_frames / SAMPLE_RATE:.1f}s")
                    else:
                        parts.append(f"  saved={stats.saved_frames / SAMPLE_RATE:.1f}s")
                    print("".join(parts))
                    last_status_time = now

    except KeyboardInterrupt:
        print("\nRecording interrupted by Ctrl+C.", file=sys.stderr)
    except sd.PortAudioError as exc:
        raise RuntimeError(
            f"Audio device error: {exc}\n"
            "The USB audio device may have been disconnected or is in use "
            "by another application."
        ) from exc
    finally:
        if spatial is not None:
            spatial.stop()
        if wf is not None:
            wf.close()
        ctrl.close()

    if trigger_on_voice and stats.saved_frames == 0:
        print("未检测到符合条件的人声，未生成 WAV 文件。", file=sys.stderr)

    return stats


# ======================================================================
# Voice-first recording routine
# ======================================================================

def record(
    output_path: str,
    duration_sec: float = DEFAULT_DURATION_SEC,
    rms_threshold: float = DEFAULT_RMS_THRESHOLD,
    device_hint: Optional[str] = None,
    beam_deg: float = DEFAULT_BEAM_AZIMUTH_DEG,
    attack_ms: float = DEFAULT_ATTACK_MS,
    hold_ms: float = DEFAULT_HOLD_MS,
    angle_tolerance_deg: float = 25.0,
    ref_rms: Optional[float] = None,
    ref_energy: Optional[float] = None,
    distance_ratio: float = 0.30,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    enable_spatial: bool = True,
    stop_event: Optional[threading.Event] = None,
    trigger_on_voice: bool = True,
    denoise: bool = DEFAULT_DENOISE,
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH,
    offaxis_suppression: bool = DEFAULT_OFFAXIS_SUPPRESSION,
    offaxis_suppression_strength: float = DEFAULT_OFFAXIS_SUPPRESSION_STRENGTH,
) -> RecordStats:
    """Record chunks that pass device VAD, DOA and optional distance checks."""

    dev = find_device()
    if dev is None:
        raise RuntimeError(
            "No XVF3800 control device found (VID=0x2886 PID=0x001A). "
            "Check USB connection and WinUSB/libusb access."
        )

    ctrl = ReSpeakerControl(dev)
    spatial: Optional[SpatialMonitor] = None
    wf: Optional[wave.Wave_write] = None
    try:
        configure_device(ctrl, beam_deg, reference_beam=offaxis_suppression)

        input_index = pick_input_device(device_hint)
        if input_index is None:
            devices = sd.query_devices()
            details = "\n".join(
                f"  [{i}] {d['name']} (inputs={d['max_input_channels']})"
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            )
            raise RuntimeError(f"No matching XVF3800 input device found.\n{details}")

        dev_info = sd.query_devices(input_index)
        print(f"Audio input : [{input_index}] {dev_info['name']}")
        print(f"Sample rate : {SAMPLE_RATE} Hz | channels: {CHANNELS}")

        gate = RmsGate(
            threshold=rms_threshold,
            block_size=BLOCKSIZE,
            sample_rate=SAMPLE_RATE,
            attack_ms=attack_ms,
            hold_ms=hold_ms,
        )

        distance_ref = ref_rms if ref_rms is not None else ref_energy

        if enable_spatial:
            spatial = SpatialMonitor(
                ctrl=ctrl,
                target_angle_deg=beam_deg,
                angle_tolerance_deg=angle_tolerance_deg,
                ref_rms=distance_ref,
                distance_ratio=distance_ratio,
                ratio_threshold=ratio_threshold,
            )
            spatial.start(interval=0.20)
            dist_msg = "disabled" if distance_ref is None else f">= {distance_ratio * 100:.0f}% of ref"
            ratio_msg = "disabled" if ratio_threshold <= 0.0 else f">= {ratio_threshold:.2f}"
            print(
                f"Voice filter : device VAD + DOA +/-{angle_tolerance_deg:.0f} deg | "
                f"distance RMS {dist_msg} | focus {ratio_msg}"
            )
        if offaxis_suppression:
            print(
                f"Off-axis suppression: enabled "
                f"(opposite beam reference, strength={offaxis_suppression_strength:.2f})"
            )

        block_seconds = BLOCKSIZE / SAMPLE_RATE
        spatial_attack_blocks = max(
            2,
            int(math.ceil((attack_ms / 1000.0) / block_seconds)),
        )
        spatial_hold_blocks = max(
            1,
            int(math.ceil((hold_ms / 1000.0) / block_seconds)),
        )
        pre_roll_chunks: deque[bytes] = deque(
            maxlen=max(1, int(math.ceil(0.5 / block_seconds)))
        )
        spatial_open = not enable_spatial
        spatial_pass_streak = 0
        spatial_fail_streak = 0

        if stop_event is None:
            stop_event = threading.Event()
        keyboard_stop_enabled = sys.stdin is not None and sys.stdin.isatty()
        if keyboard_stop_enabled:
            threading.Thread(target=_any_key_stop, args=(stop_event,), daemon=True).start()

        q: queue.Queue[bytes] = queue.Queue(maxsize=64)
        stats = RecordStats()
        t_start = time.monotonic()
        last_status_time = t_start

        def output_mono_bytes(stereo_chunk: bytes) -> bytes:
            if not offaxis_suppression:
                return extract_mono_bytes(stereo_chunk, channels=CHANNELS)
            mono, suppressed, ref_rms = suppress_offaxis_bytes(
                stereo_chunk,
                channels=CHANNELS,
                strength=offaxis_suppression_strength,
            )
            stats.offaxis_reference_peak_rms = max(stats.offaxis_reference_peak_rms, ref_rms)
            if suppressed:
                stats.offaxis_suppressed_chunks += 1
            return mono

        def callback(indata, frames, _time_info, status) -> None:
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            try:
                q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        def ensure_wave_open() -> wave.Wave_write:
            nonlocal wf
            if wf is None:
                wf = wave.open(output_path, "wb")
                wf.setnchannels(1)
                wf.setsampwidth(SAMPLE_WIDTH_BYTES)
                wf.setframerate(SAMPLE_RATE)
            return wf

        if not trigger_on_voice:
            ensure_wave_open()

        print(f"\nRecording -> {output_path}")
        print(f"Beam azimuth  : {beam_deg:.1f} deg")
        print(f"RMS floor     : {rms_threshold:.4f} (attack={attack_ms:.0f} ms, hold={hold_ms:.0f} ms)")
        if duration_sec > 0:
            print(f"Max duration  : {duration_sec:.1f} s")
        else:
            print("Max duration  : unlimited")
        if keyboard_stop_enabled:
            print("Press any key to stop early.\n")

        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCKSIZE,
            device=input_index,
            callback=callback,
        ):
            while True:
                if stop_event.is_set():
                    print("\nStopped by user request.")
                    break
                elapsed = time.monotonic() - t_start
                if duration_sec > 0 and elapsed >= duration_sec:
                    print(f"\nReached max duration ({duration_sec:.1f} s).")
                    break

                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    continue

                stats.received_chunks += 1
                stats.received_frames += len(chunk) // (SAMPLE_WIDTH_BYTES * CHANNELS)
                pre_roll_chunks.append(chunk)

                level = rms_int16_mono(chunk, channels=CHANNELS)
                stats.peak_rms = max(stats.peak_rms, level)
                rms_ok = gate.update(level)

                speech_ok = True
                doa_ok = True
                energy_ok = True
                ratio_ok = True
                doa_deg = None
                energy = None
                focus = None
                opened_this_chunk = False

                if spatial is not None:
                    speech_ok, doa_ok, energy_ok, ratio_ok, doa_deg, energy, focus = spatial.check(gate.smoothed_rms)
                    if speech_ok:
                        stats.speech_samples += 1
                    if doa_deg is not None:
                        stats.doa_samples.append(doa_deg)
                    if energy is not None:
                        stats.energy_samples.append(energy)

                    instant_spatial_ok = speech_ok and doa_ok and energy_ok and ratio_ok
                    if instant_spatial_ok:
                        spatial_pass_streak += 1
                        spatial_fail_streak = 0
                    else:
                        spatial_fail_streak += 1
                        spatial_pass_streak = 0

                    if not spatial_open and spatial_pass_streak >= spatial_attack_blocks:
                        spatial_open = True
                        opened_this_chunk = True
                        if trigger_on_voice and wf is None:
                            ensure_wave_open()
                            print("Detected qualifying 30 deg voice, starting WAV write.")
                        if wf is not None:
                            while pre_roll_chunks:
                                buffered = pre_roll_chunks.popleft()
                                wf.writeframesraw(output_mono_bytes(buffered))
                                stats.saved_chunks += 1
                                stats.saved_frames += len(buffered) // (SAMPLE_WIDTH_BYTES * CHANNELS)

                    elif spatial_open and spatial_fail_streak >= spatial_hold_blocks:
                        spatial_open = False

                    if rms_ok and not instant_spatial_ok:
                        if not speech_ok:
                            stats.speech_rejects += 1
                        if not doa_ok:
                            stats.doa_rejects += 1
                        if not energy_ok:
                            stats.energy_rejects += 1
                        if not ratio_ok:
                            stats.ratio_rejects += 1

                write_chunk = rms_ok and spatial_open

                if write_chunk and not opened_this_chunk:
                    if trigger_on_voice and wf is None:
                        ensure_wave_open()
                        if spatial is None:
                            print("Detected RMS-qualified audio, starting WAV write.")
                    mono = output_mono_bytes(chunk)
                    if wf is not None:
                        wf.writeframesraw(mono)
                        stats.saved_chunks += 1
                        stats.saved_frames += len(mono) // SAMPLE_WIDTH_BYTES

                now = time.monotonic()
                if now - last_status_time >= 2.0:
                    state = "OPEN" if gate.open else "CLOSED"
                    parts = [
                        f"  [{elapsed:5.1f}s] RMS={level:.4f} smoothed={gate.smoothed_rms:.4f}",
                        f" rms_gate={state}",
                    ]
                    if spatial is not None:
                        parts.append(f" speech={'YES' if speech_ok else 'no'}")
                        parts.append(f" DOA={doa_deg:.1f}deg {'OK' if doa_ok else 'OFF'}" if doa_deg is not None else " DOA=--")
                        if focus is not None and ratio_threshold > 0.0:
                            parts.append(f" focus={focus:.2f} {'OK' if ratio_ok else 'LOW'}")
                        parts.append(f" spatial={'OPEN' if spatial_open else 'WAIT'}")
                    parts.append(f" saved={stats.saved_frames / SAMPLE_RATE:.1f}s")
                    print("".join(parts))
                    last_status_time = now

    except sd.PortAudioError as exc:
        raise RuntimeError(f"Audio device error: {exc}") from exc
    finally:
        if spatial is not None:
            spatial.stop()
        if wf is not None:
            wf.close()
        ctrl.close()

    if denoise and stats.saved_frames > 0:
        result = light_denoise_wav(output_path, strength=denoise_strength)
        stats.denoise_applied = result.applied
        stats.denoise_noise_rms = result.noise_rms
        stats.denoise_peak_rms = result.peak_rms_after
        if result.applied:
            print(
                f"Light denoise applied: noise_rms={result.noise_rms:.4f}, "
                f"peak_after={result.peak_rms_after:.4f}"
            )

    if trigger_on_voice and stats.saved_frames == 0:
        print("No qualifying 30 deg voice detected; WAV was not created.", file=sys.stderr)

    return stats


# ======================================================================
# CLI
# ======================================================================

def parse_args():
    import argparse

    p = argparse.ArgumentParser(
        description="Record XVF3800 beamformed audio to WAV"
    )
    p.add_argument(
        "-o", "--output", default="respeaker_30deg_close.wav",
        help="Output WAV path (default: respeaker_30deg_close.wav)",
    )
    p.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION_SEC,
        help=f"Max recording time in seconds (default: {DEFAULT_DURATION_SEC})",
    )
    p.add_argument(
        "--rms-threshold", type=float, default=DEFAULT_RMS_THRESHOLD,
        help=(
            "Normalised RMS threshold 0–1 (default: %.3f). "
            "Lower values let more quiet audio through."
            % DEFAULT_RMS_THRESHOLD
        ),
    )
    p.add_argument(
        "--device-hint", default=None,
        help="Substring to match the USB audio input device name",
    )
    p.add_argument(
        "--beam-deg", type=float, default=DEFAULT_BEAM_AZIMUTH_DEG,
        help=f"Fixed beam azimuth in degrees (default: {DEFAULT_BEAM_AZIMUTH_DEG})",
    )
    p.add_argument(
        "--attack-ms", type=float, default=DEFAULT_ATTACK_MS,
        help=f"Gate attack time in ms (default: {DEFAULT_ATTACK_MS})",
    )
    p.add_argument(
        "--hold-ms", type=float, default=DEFAULT_HOLD_MS,
        help=f"Gate hold time in ms (default: {DEFAULT_HOLD_MS})",
    )
    p.add_argument(
        "--angle-tolerance", type=float, default=25.0,
        help="Allowed DOA deviation from beam angle in degrees (default: 25)",
    )
    p.add_argument(
        "--calibrate", action="store_true",
        help="Run calibration to measure reference energy at the target position",
    )
    p.add_argument(
        "--cal-json", default="calibration.json",
        help="Calibration file path (default: calibration.json)",
    )
    p.add_argument(
        "--cal-duration", type=float, default=3.0,
        help="Calibration recording duration in seconds (default: 3.0)",
    )
    p.add_argument(
        "--ref-energy", type=float, default=None,
        help="Reference speech energy override (bypasses calibration file)",
    )
    p.add_argument(
        "--energy-tolerance", type=float, default=0.70,
        help="Allowed energy drop below reference as fraction 0–1 (default: 0.7)",
    )
    p.add_argument(
        "--ratio-threshold", type=float, default=DEFAULT_RATIO_THRESHOLD,
        help="Optional focus ratio E0/sum(E0..E3); 0 disables it (default: 0)",
    )
    p.add_argument(
        "--trigger-on-voice", action="store_true", default=True,
        help="Create the WAV only after the first RMS/DOA/energy-approved voice chunk",
    )
    p.add_argument(
        "--no-denoise", action="store_true",
        help="Disable the lightweight post-recording denoise pass",
    )
    p.add_argument(
        "--denoise-strength", type=float, default=DEFAULT_DENOISE_STRENGTH,
        help=f"Light denoise strength 0-1 (default: {DEFAULT_DENOISE_STRENGTH})",
    )
    p.add_argument(
        "--no-offaxis-suppression", action="store_true",
        help="Disable opposite-beam sidechain suppression for off-axis speech",
    )
    p.add_argument(
        "--offaxis-strength", type=float, default=DEFAULT_OFFAXIS_SUPPRESSION_STRENGTH,
        help=(
            "Opposite-beam suppression strength 0-1 "
            f"(default: {DEFAULT_OFFAXIS_SUPPRESSION_STRENGTH})"
        ),
    )
    p.add_argument(
        "--target-distance", type=float, default=0.5,
        help="Target distance in meters for calibration record (default: 0.5)",
    )
    p.add_argument(
        "--no-spatial", action="store_true",
        help="Disable DOA and energy spatial filtering (RMS gate only)",
    )
    return p.parse_args()


# ======================================================================
# Entry point
# ======================================================================

def main() -> int:
    args = parse_args()

    # Early validation
    if not 0.0 <= args.rms_threshold <= 1.0:
        print("ERROR: --rms-threshold must be between 0.0 and 1.0", file=sys.stderr)
        return 1
    if not 0.0 <= args.energy_tolerance <= 1.0:
        print("ERROR: --energy-tolerance must be between 0.0 and 1.0", file=sys.stderr)
        return 1
    if not 0.0 <= args.denoise_strength <= 1.0:
        print("ERROR: --denoise-strength must be between 0.0 and 1.0", file=sys.stderr)
        return 1
    if not 0.0 <= args.offaxis_strength <= 1.0:
        print("ERROR: --offaxis-strength must be between 0.0 and 1.0", file=sys.stderr)
        return 1

    print("=" * 58)
    print("  ReSpeaker XVF3800 – Beamforming Audio Recorder")
    print("=" * 58)

    # -- resolve reference RMS ----------------------------------------------
    ref_rms: Optional[float] = args.ref_energy

    if ref_rms is None and not args.calibrate and not args.no_spatial:
        try:
            cal = _load_calibration(args.cal_json)
            ref_rms = cal.get("ref_rms", cal.get("ref_energy"))
            if ref_rms is not None:
                print(f"Loaded calibration: {args.cal_json}")
                print(f"  Target angle : {cal.get('target_angle_deg', '?')}°")
                print(f"  Target dist  : {cal.get('target_distance_m', '?')} m")
                print(f"  Ref RMS      : {ref_rms:.6f}")
        except (FileNotFoundError, KeyError, ValueError):
            pass  # no calibration file – RMS gate will be disabled

    try:
        # -- calibration mode -----------------------------------------------
        if args.calibrate:
            # Enable spatial so we can collect RMS samples at the target position.
            # DOA and energy gates are effectively disabled (wide tolerance,
            # no ref_rms) so all RMS-gated audio passes through while the
            # background thread samples the reference level.
            import statistics
            stats = record(
                output_path=args.output,
                duration_sec=args.cal_duration,
                rms_threshold=args.rms_threshold,
                device_hint=args.device_hint,
                beam_deg=args.beam_deg,
                attack_ms=args.attack_ms,
                hold_ms=args.hold_ms,
                angle_tolerance_deg=180.0,     # disable DOA gate
                ref_rms=None,                   # disable RMS distance gate
                enable_spatial=True,            # collect speech energy samples
            )
            if stats.doa_samples:
                print(f"\n  DOA samples    : {len(stats.doa_samples)}")
                print(f"  DOA mean       : {statistics.mean(stats.doa_samples):.1f}°")
            if stats.energy_samples:
                avg_rms = statistics.mean(stats.energy_samples)
                print(f"  RMS samples    : {len(stats.energy_samples)}")
                print(f"  RMS mean       : {avg_rms:.6f}")
            else:
                print("\n  WARNING: No RMS samples collected. "
                      "Is the sound source active?")
                avg_rms = 0.0
            _save_calibration(
                args.cal_json,
                target_angle_deg=args.beam_deg,
                target_distance_m=args.target_distance,
                ref_energy=avg_rms,
                ref_rms=stats.peak_rms,
            )
            print("\nCalibration complete. Now run without --calibrate to use "
                  "the spatial filter with the saved reference.")
            return 0

        # -- normal recording mode ------------------------------------------
        stats = record(
            output_path=args.output,
            duration_sec=args.duration,
            rms_threshold=args.rms_threshold,
            device_hint=args.device_hint,
            beam_deg=args.beam_deg,
            attack_ms=args.attack_ms,
            hold_ms=args.hold_ms,
            angle_tolerance_deg=args.angle_tolerance,
            ref_rms=ref_rms,
            distance_ratio=1.0 - args.energy_tolerance,
            ratio_threshold=args.ratio_threshold,
            enable_spatial=not args.no_spatial,
            trigger_on_voice=args.trigger_on_voice,
            denoise=not args.no_denoise,
            denoise_strength=args.denoise_strength,
            offaxis_suppression=not args.no_offaxis_suppression,
            offaxis_suppression_strength=args.offaxis_strength,
        )
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nUNEXPECTED ERROR: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # ---- summary -----------------------------------------------------------
    recv_sec = stats.received_frames / SAMPLE_RATE
    saved_sec = stats.saved_frames / SAMPLE_RATE
    ratio = (saved_sec / recv_sec * 100) if recv_sec > 0 else 0.0

    print(f"\n{'─' * 40}")
    print(f"  Recording finished.")
    print(f"  Captured  : {recv_sec:.2f} s  ({stats.received_chunks} chunks)")
    print(f"  Saved     : {saved_sec:.2f} s  ({stats.saved_chunks} chunks, {ratio:.1f}%)")
    print(f"  Peak RMS  : {stats.peak_rms:.4f}")
    if stats.denoise_applied:
        print(
            f"  Denoise   : applied "
            f"(noise_rms={stats.denoise_noise_rms:.4f}, "
            f"peak_after={stats.denoise_peak_rms:.4f})"
        )
    elif not args.no_denoise and saved_sec > 0.0:
        print("  Denoise   : skipped")
    if stats.offaxis_reference_peak_rms > 0.0:
        print(
            f"  Off-axis  : suppressed {stats.offaxis_suppressed_chunks} chunks "
            f"(ref_peak_rms={stats.offaxis_reference_peak_rms:.4f})"
        )
    if stats.speech_rejects or stats.doa_rejects or stats.energy_rejects or stats.ratio_rejects:
        print(f"  Speech rejects: {stats.speech_rejects} chunks")
        print(f"  DOA rejects   : {stats.doa_rejects} chunks")
        print(f"  Energy rejects: {stats.energy_rejects} chunks")
        print(f"  Ratio rejects : {stats.ratio_rejects} chunks")
    print(f"  Speech samples: {stats.speech_samples}")
    if stats.doa_samples:
        import statistics
        doa_arr = stats.doa_samples
        print(f"  DOA mean  : {statistics.mean(doa_arr):.1f}°  "
              f"(min={min(doa_arr):.1f}°, max={max(doa_arr):.1f}°)")
    if args.trigger_on_voice and not pathlib.Path(args.output).exists():
        print("  Output    : not created (no qualifying voice detected)")
    else:
        print(f"  Output    : {args.output}")
    print(f"{'─' * 40}")

    if saved_sec == 0.0:
        print(
            "\nWARNING: No audio passed the RMS gate. "
            "Try lowering --rms-threshold (current: %.4f) "
            "or reducing --attack-ms / --hold-ms." % args.rms_threshold,
            file=sys.stderr,
        )
    if stats.doa_rejects > stats.saved_chunks and saved_sec > 0:
        print(
            "\nWARNING: Spatial filter rejected most chunks. "
            "The sound source may be at the wrong angle or distance. "
            "Check DOA mean above vs. target beam angle.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
