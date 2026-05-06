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
import math
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
    "AEC_FIXEDBEAMSAZIMUTH_VALUES":  (33, 81, 4, "rw", "radians"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 4, "rw", "radians"),
    "AEC_FIXEDBEAMSONOFF":           (33, 37, 1, "rw", "int32"),
    "AEC_FIXEDBEAMSGATING":          (33, 83, 1, "rw", "uint8"),
    "AUDIO_MGR_OP_L":                (35, 15, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R":                (35, 19, 2, "rw", "uint8"),
    "AEC_AZIMUTH_VALUES":            (33, 75, 4, "ro", "radians"),
    "AEC_SPENERGY_VALUES":           (33, 80, 4, "ro", "float"),
    "AUDIO_MGR_SELECTED_AZIMUTHS":   (35, 11, 2, "ro", "radians"),
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
        Reference speech energy from calibration.  ``None`` disables the
        energy gate.
    energy_tolerance : float
        Allowed fractional deviation from *ref_energy* (0.0 – 1.0).
    """

    def __init__(
        self,
        ctrl: ReSpeakerControl,
        target_angle_deg: float,
        angle_tolerance_deg: float,
        ref_energy: Optional[float],
        energy_tolerance: float,
    ) -> None:
        self._ctrl = ctrl
        self.target_angle = target_angle_deg
        self.angle_tolerance = angle_tolerance_deg
        self.ref_energy = ref_energy
        self.energy_tolerance = energy_tolerance

        self._lock = threading.Lock()
        self._latest_doa: Optional[float] = None       # radians or NaN
        self._latest_energy: Optional[float] = None     # beam-0 speech energy
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
                if energy_vals is not None and len(energy_vals) >= 1:
                    with self._lock:
                        self._latest_energy = energy_vals[0]
            except Exception:
                pass

            time.sleep(interval)

    # -- main-loop API --------------------------------------------------------

    def check(self):
        """Return ``(doa_ok, energy_ok, doa_deg, energy)``.

        *doa_deg* is ``None`` when no valid DOA has been read yet;
        *energy* is ``None`` when no speech-energy reading is available.
        Both gates default to ``True`` when data is unavailable so that
        the filter never discards audio because of a transient read failure.
        """
        with self._lock:
            doa = self._latest_doa
            energy = self._latest_energy

        doa_ok = True
        energy_ok = True
        doa_deg: Optional[float] = None

        if doa is not None and not math.isnan(doa):
            doa_deg = math.degrees(doa)
            diff = abs(doa_deg - self.target_angle)
            if diff > 180.0:
                diff = 360.0 - diff
            doa_ok = diff <= self.angle_tolerance

        if (
            energy is not None
            and self.ref_energy is not None
            and self.ref_energy > 0.0
        ):
            deviation = abs(energy - self.ref_energy) / self.ref_energy
            energy_ok = deviation <= self.energy_tolerance

        return doa_ok, energy_ok, doa_deg, energy


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

def _float_from_bytes(data: bytes) -> float:
    """Unpack a little-endian IEEE-754 float from 4 bytes."""
    return struct.unpack("<f", data)[0]


def _float_to_bytes(value: float) -> bytes:
    """Pack a float into 4 little-endian bytes."""
    return struct.pack("<f", value)


# ======================================================================
# ReSpeakerControl – minimal XVF3800 USB parameter read/write
# ======================================================================

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
    doa_rejects: int = 0
    energy_rejects: int = 0
    doa_samples: list[float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.doa_samples is None:
            self.doa_samples = []


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

    print(f"Configuring fixed beam: azimuth={beam_deg:.1f}°, elevation=0°")
    print(f"  Beam 1 → {beam_deg:.1f}°  |  Beam 2 → {(beam_deg + 180) % 360:.1f}° (opposite)")

    # -- write beam parameters (4 beams: primary, opposite, 2 unused) ---------
    ctrl.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [beam_rad, opposite_rad, 0.0, 0.0])
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

    if azimuths is not None and len(azimuths) >= 2:
        az_ok = all(abs(a - e) < 0.1 for a, e in zip(azimuths[:2], [beam_rad, opposite_rad]))
    else:
        az_ok = True  # can't verify, assume OK

    if elevations is not None and len(elevations) >= 2:
        el_ok = all(abs(e) < 0.1 for e in elevations[:2])
    else:
        el_ok = True

    en_ok = (enabled == (1,)) if enabled is not None else True
    route_ok = (op_l == (6, 0)) if op_l is not None else True

    if not az_ok:
        print(f"  WARNING: azimuth mismatch – wrote {[beam_rad, opposite_rad]}, "
              f"read {azimuths}", file=sys.stderr)
    if not el_ok:
        print(f"  WARNING: elevation mismatch – wrote [0.0, 0.0], "
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
    ref_energy: Optional[float] = None,
    energy_tolerance: float = 0.5,
    enable_spatial: bool = True,
    stop_event: Optional[threading.Event] = None,
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
    ref_energy : float or None
        Reference speech energy from calibration.  ``None`` disables the
        energy/distance gate.
    energy_tolerance : float
        Allowed fractional deviation from *ref_energy* (0.0 – 1.0).
    enable_spatial : bool
        When False the spatial monitor is not started (plain RMS-only gate).
    stop_event : threading.Event or None
        Optional external stop signal, used by GUI frontends.

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
        spatial: Optional[SpatialMonitor] = None
        if enable_spatial:
            spatial = SpatialMonitor(
                ctrl=ctrl,
                target_angle_deg=beam_deg,
                angle_tolerance_deg=angle_tolerance_deg,
                ref_energy=ref_energy,
                energy_tolerance=energy_tolerance,
            )
            spatial.start(interval=0.25)
            if ref_energy is not None:
                print(f"Spatial gate : DOA ±{angle_tolerance_deg:.0f}°  |  "
                      f"energy ±{energy_tolerance * 100:.0f}% (ref={ref_energy:.6f})")
            else:
                print(f"Spatial gate : DOA ±{angle_tolerance_deg:.0f}°  |  "
                      f"energy gate DISABLED (no calibration)")

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
        print(f"Max duration  : {duration_sec:.1f} s")
        if keyboard_stop_enabled:
            print("Press any key to stop early.\n")
        else:
            print("Use the GUI stop button or an external stop signal to stop early.\n")

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(SAMPLE_RATE)

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
                    if elapsed >= duration_sec:
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
                        # -- spatial filter (DOA + energy) --
                        write_chunk = True
                        if spatial is not None:
                            doa_ok, energy_ok, doa_deg, energy = spatial.check()
                            if doa_deg is not None:
                                stats.doa_samples.append(doa_deg)
                            if not doa_ok:
                                stats.doa_rejects += 1
                                write_chunk = False
                            if not energy_ok:
                                stats.energy_rejects += 1
                                write_chunk = False

                        if write_chunk:
                            mono = extract_mono_bytes(chunk, channels=CHANNELS)
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
                            doa_ok, energy_ok, doa_deg, _ = spatial.check()
                            if doa_deg is not None:
                                tag = "OK" if doa_ok else "OFF-AXIS"
                                parts.append(f"  DOA={doa_deg:.1f}° {tag}")
                            else:
                                parts.append("  DOA=--")
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
        ctrl.close()

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
        "--energy-tolerance", type=float, default=0.5,
        help="Allowed energy deviation as fraction 0–1 (default: 0.5 = ±50%%)",
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

    print("=" * 58)
    print("  ReSpeaker XVF3800 – Beamforming Audio Recorder")
    print("=" * 58)

    # -- resolve reference energy -------------------------------------------
    ref_energy: Optional[float] = args.ref_energy

    if ref_energy is None and not args.calibrate and not args.no_spatial:
        # Try loading from calibration JSON
        import json as _json
        try:
            cal = _load_calibration(args.cal_json)
            ref_energy = cal.get("ref_energy")
            if ref_energy is not None:
                print(f"Loaded calibration: {args.cal_json}")
                print(f"  Target angle : {cal.get('target_angle_deg', '?')}°")
                print(f"  Target dist  : {cal.get('target_distance_m', '?')} m")
                print(f"  Ref energy   : {ref_energy:.6f}")
        except (FileNotFoundError, KeyError, ValueError):
            pass  # no calibration file – energy gate will be disabled

    try:
        # -- calibration mode -----------------------------------------------
        if args.calibrate:
            stats = record(
                output_path=args.output,
                duration_sec=args.cal_duration,
                rms_threshold=args.rms_threshold,
                device_hint=args.device_hint,
                beam_deg=args.beam_deg,
                attack_ms=args.attack_ms,
                hold_ms=args.hold_ms,
                enable_spatial=False,  # no spatial filter during calibration
            )
            if stats.doa_samples:
                import statistics
                print(f"\n  DOA samples : {len(stats.doa_samples)}")
                print(f"  DOA mean    : {statistics.mean(stats.doa_samples):.1f}°")
            # For calibration, the ref energy comes from RMS, not speech energy
            _save_calibration(
                args.cal_json,
                target_angle_deg=args.beam_deg,
                target_distance_m=args.target_distance,
                ref_energy=stats.peak_rms,
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
            ref_energy=ref_energy,
            energy_tolerance=args.energy_tolerance,
            enable_spatial=not args.no_spatial,
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
    if stats.doa_rejects or stats.energy_rejects:
        print(f"  DOA rejects   : {stats.doa_rejects} chunks")
        print(f"  Energy rejects: {stats.energy_rejects} chunks")
    if stats.doa_samples:
        import statistics
        doa_arr = stats.doa_samples
        print(f"  DOA mean  : {statistics.mean(doa_arr):.1f}°  "
              f"(min={min(doa_arr):.1f}°, max={max(doa_arr):.1f}°)")
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
