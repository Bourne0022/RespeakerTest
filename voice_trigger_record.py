"""Voice-triggered spatial recording for XVF3800 — no calibration required.

Only records audio when ALL of these conditions are met:
  1. Sound direction (DOA) is within angle_tolerance of the target beam
  2. Device's auto-select beam agrees on the direction
  3. Beam energy ratio confirms the sound is in the front hemisphere
  4. Audio RMS level indicates a close-range source (~0.5 m)

Recording auto-starts when voice is detected within the spatial window,
and auto-stops after a configurable silence period.

Usage:
  python voice_trigger_record.py --angle 30 --duration 0

Set --duration 0 for unlimited recording (stop with Ctrl+C or auto-silence).
"""

from __future__ import annotations

import array
import math
import pathlib
import platform
import queue
import struct
import sys
import threading
import time
import wave
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
try:
    import sounddevice as sd
except ImportError:
    raise SystemExit("pip install sounddevice")

try:
    import usb.core
    import usb.util
except ImportError:
    raise SystemExit("pip install pyusb")

try:
    import libusb_package
except Exception:
    libusb_package = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VID, PID = 0x2886, 0x001A
SAMPLE_RATE = 16_000
CHANNELS = 2
SAMPLE_WIDTH = 2
BLOCKSIZE = 1024

# USB parameter table (resid, cmdid, count, access, type)
PARAMS = {
    "AEC_FIXEDBEAMSAZIMUTH_VALUES":  (33, 81, 4, "rw", "radians"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 4, "rw", "radians"),
    "AEC_FIXEDBEAMSONOFF":           (33, 37, 1, "rw", "int32"),
    "AEC_FIXEDBEAMSGATING":          (33, 83, 1, "rw", "uint8"),
    "AUDIO_MGR_OP_L":                (35, 15, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R":                (35, 19, 2, "rw", "uint8"),
    "AEC_SPENERGY_VALUES":           (33, 80, 4, "ro", "float"),
    "AUDIO_MGR_SELECTED_AZIMUTHS":   (35, 11, 2, "ro", "radians"),
    "AEC_AZIMUTH_VALUES":            (33, 75, 4, "ro", "radians"),
}


# ---------------------------------------------------------------------------
# USB helpers
# ---------------------------------------------------------------------------
def _f32_from_bytes(data: bytes) -> float:
    return struct.unpack("<f", data)[0]

def _f32_to_bytes(value: float) -> bytes:
    return struct.pack("<f", value)

def _rad(deg: float) -> float:
    return deg * math.pi / 180.0


# ---------------------------------------------------------------------------
# XVF3800 USB control
# ---------------------------------------------------------------------------
class XVF3800:
    TIMEOUT = 100_000

    def __init__(self, dev: usb.core.Device) -> None:
        self.dev = dev
        self._claim()

    def _claim(self) -> None:
        iface = 3
        try:
            if self.dev.is_kernel_driver_active(iface):
                self.dev.detach_kernel_driver(iface)
        except (usb.core.USBError, NotImplementedError):
            pass
        try:
            usb.util.claim_interface(self.dev, iface)
        except usb.core.USBError as e:
            raise RuntimeError(f"Cannot claim USB interface 3: {e}") from e

    def write(self, name: str, values: list) -> None:
        resid, cmdid, count, access, vtype = PARAMS[name]
        if access == "ro":
            raise ValueError(f"'{name}' is read-only")
        if len(values) != count:
            raise ValueError(f"'{name}' expects {count} values, got {len(values)}")
        payload = bytearray()
        if vtype in ("float", "radians"):
            for v in values:
                payload += _f32_to_bytes(float(v))
        elif vtype == "uint8":
            for v in values:
                payload += int(v).to_bytes(1, "little")
        elif vtype == "int32":
            for v in values:
                payload += int(v).to_bytes(4, "little", signed=True)
        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0, cmdid, resid, bytes(payload), self.TIMEOUT,
        )

    def read(self, name: str, retries: int = 30) -> Optional[tuple]:
        resid, cmdid, count, access, vtype = PARAMS[name]
        if access == "wo":
            raise ValueError(f"'{name}' is write-only")
        read_cmdid = 0x80 | cmdid
        if vtype in ("float", "radians", "int32"):
            data_len = count * 4 + 1
        elif vtype == "uint8":
            data_len = count + 1
        else:
            data_len = count + 1

        for _ in range(retries):
            try:
                resp = self.dev.ctrl_transfer(
                    usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                    0, read_cmdid, resid, data_len, self.TIMEOUT,
                )
            except usb.core.USBError:
                time.sleep(0.01)
                continue
            if resp[0] == 0:  # CONTROL_SUCCESS
                raw = resp.tobytes()
                if vtype in ("float", "radians"):
                    return tuple(_f32_from_bytes(raw[i:i+4]) for i in range(1, len(raw), 4))
                if vtype == "uint8":
                    return tuple(resp.tolist()[1:])
                if vtype == "int32":
                    return tuple(int.from_bytes(raw[i:i+4], "little", signed=True) for i in range(1, len(raw), 4))
            time.sleep(0.01)
        return None

    def close(self) -> None:
        try:
            usb.util.release_interface(self.dev, 3)
        except usb.core.USBError:
            pass
        usb.util.dispose_resources(self.dev)


# ---------------------------------------------------------------------------
# Device discovery
# ---------------------------------------------------------------------------
def find_device() -> Optional[usb.core.Device]:
    if platform.system().lower().startswith("win") and libusb_package is not None:
        return libusb_package.find(idVendor=VID, idProduct=PID)
    return usb.core.find(idVendor=VID, idProduct=PID)


def pick_input(hint: Optional[str] = None) -> Optional[int]:
    devices = sd.query_devices()
    hints = (hint,) if hint else ("reSpeaker", "respeaker", "XVF3800", "xvf3800",
                                   "USB Audio", "USB Audio CODEC")
    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        if any(h.lower() in dev["name"].lower() for h in hints):
            return idx
    return None


# ---------------------------------------------------------------------------
# Configure device beams
# ---------------------------------------------------------------------------
def configure(ctrl: XVF3800, beam_deg: float) -> None:
    """Set up 2 fixed beams at target and opposite angles.

    XVF3800 beam architecture (from XMOS docs):
      - 2 fixed beams (slow-update, what we configure)
      - 1 free-running scan beam (fast, scans for new sources)
      - 1 auto-select beam (picks best-energy output)

    AEC_SPENERGY_VALUES indices:
      [0] = fixed beam 1 (our target)
      [1] = fixed beam 2 (our opposite)
      [2] = free-running scan beam
      [3] = auto-select beam
    """
    beam_rad = _rad(beam_deg)
    opp_rad = _rad((beam_deg + 180) % 360)

    print(f"Beam config:")
    print(f"  Fixed beam 0 → {beam_deg:.0f}° (target, audio output)")
    print(f"  Fixed beam 1 → {(beam_deg + 180) % 360:.0f}° (opposite)")

    # Write 2 fixed beams (remaining 2 slots unused)
    ctrl.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [beam_rad, opp_rad, 0.0, 0.0])
    ctrl.write("AEC_FIXEDBEAMSELEVATION_VALUES", [0.0, 0.0, 0.0, 0.0])
    ctrl.write("AEC_FIXEDBEAMSGATING", [0])
    ctrl.write("AEC_FIXEDBEAMSONOFF", [1])

    # Route fixed beam 0 output to left audio channel
    ctrl.write("AUDIO_MGR_OP_L", [6, 0])
    ctrl.write("AUDIO_MGR_OP_R", [0, 0])

    # Verify
    try:
        az = ctrl.read("AEC_FIXEDBEAMSAZIMUTH_VALUES", retries=10)
    except Exception:
        try:
            az = ctrl.read("AEC_AZIMUTH_VALUES", retries=10)
        except Exception:
            az = None

    if az is not None and len(az) >= 2:
        ok = all(abs(a - e) < 0.1 for a, e in zip(az[:2], [beam_rad, opp_rad]))
        status = "OK" if ok else f"MISMATCH (read {az})"
        print(f"  Verification: {status}")
    else:
        print("  Verification: skipped (firmware busy)")

    # Diagnostics
    try:
        sp = ctrl.read("AEC_SPENERGY_VALUES", retries=5)
        if sp:
            print(f"  Energy [E0 E1 E_scan E_auto]: [{sp[0]:.1f}, {sp[1]:.1f}, "
                  f"{sp[2]:.1f}, {sp[3]:.1f}]")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# RMS gate (smoothed, with attack/hold)
# ---------------------------------------------------------------------------
class RmsGate:
    def __init__(self, threshold: float, attack_ms: float = 40, hold_ms: float = 400,
                 block_size: int = BLOCKSIZE, sample_rate: int = SAMPLE_RATE,
                 alpha: float = 0.15):
        sec_per_block = block_size / sample_rate
        self.attack_blocks = max(1, int(attack_ms / 1000 / sec_per_block))
        self.hold_blocks = max(1, int(hold_ms / 1000 / sec_per_block))
        self.threshold = threshold
        self.alpha = alpha
        self._smoothed = 0.0
        self._above = 0
        self._below = 0
        self._open = False

    @property
    def open(self) -> bool:
        return self._open

    @property
    def smoothed(self) -> float:
        return self._smoothed

    def update(self, rms: float) -> bool:
        self._smoothed = self.alpha * rms + (1 - self.alpha) * self._smoothed
        if self._smoothed >= self.threshold:
            self._above += 1
            self._below = 0
        else:
            self._below += 1
            self._above = 0
        if not self._open and self._above >= self.attack_blocks:
            self._open = True
        elif self._open and self._below >= self.hold_blocks:
            self._open = False
        return self._open


# ---------------------------------------------------------------------------
# Spatial monitor – background thread polling DOA and energy
# ---------------------------------------------------------------------------
class SpatialMonitor:
    """Polls AUDIO_MGR_SELECTED_AZIMUTHS and AEC_SPENERGY_VALUES.

    check() returns a verdict based on:
      - Speaker DOA vs target angle
      - Auto-select beam DOA vs target angle (device must agree)
      - Front/back energy ratio E0/(E0+E1)
    """

    def __init__(self, ctrl: XVF3800, target_deg: float,
                 angle_tol_deg: float, ratio_min: float = 0.6):
        self._ctrl = ctrl
        self.target = target_deg
        self.angle_tol = angle_tol_deg
        self.ratio_min = ratio_min

        self._lock = threading.Lock()
        self._doa: Optional[float] = None        # speaker DOA (radians or NaN)
        self._auto_doa: Optional[float] = None    # auto-select beam DOA
        self._e0: Optional[float] = None          # fixed beam 0 energy
        self._e1: Optional[float] = None          # fixed beam 1 energy
        self._running = False

    def start(self, interval: float = 0.2) -> None:
        self._running = True
        t = threading.Thread(target=self._poll, args=(interval,), daemon=True)
        t.start()

    def stop(self) -> None:
        self._running = False

    def _poll(self, interval: float) -> None:
        while self._running:
            try:
                doa_vals = self._ctrl.read("AUDIO_MGR_SELECTED_AZIMUTHS", retries=5)
                if doa_vals and len(doa_vals) >= 2:
                    with self._lock:
                        self._doa = doa_vals[0]       # processed speaker DOA
                        self._auto_doa = doa_vals[1]   # auto-select beam DOA
            except Exception:
                pass
            try:
                ev = self._ctrl.read("AEC_SPENERGY_VALUES", retries=5)
                if ev and len(ev) >= 2:
                    with self._lock:
                        self._e0 = ev[0]  # fixed beam 0 (target)
                        self._e1 = ev[1]  # fixed beam 1 (opposite)
            except Exception:
                pass
            time.sleep(interval)

    def check(self):
        """Return (doa_ok, auto_ok, ratio_ok, doa_deg, auto_deg, ratio).

        All gates default to True when data is unavailable.
        """
        with self._lock:
            doa = self._doa
            auto = self._auto_doa
            e0 = self._e0
            e1 = self._e1

        doa_ok, auto_ok, ratio_ok = True, True, True
        doa_deg: Optional[float] = None
        auto_deg: Optional[float] = None
        ratio: Optional[float] = None

        # DOA check: processed speaker direction
        if doa is not None and not math.isnan(doa):
            doa_deg = math.degrees(doa)
            diff = abs(doa_deg - self.target)
            if diff > 180:
                diff = 360 - diff
            doa_ok = diff <= self.angle_tol

        # Auto-select beam check: device's best-beam must agree
        if auto is not None and not math.isnan(auto):
            auto_deg = math.degrees(auto)
            diff = abs(auto_deg - self.target)
            if diff > 180:
                diff = 360 - diff
            auto_ok = diff <= self.angle_tol

        # Front/back energy ratio
        if e0 is not None and e1 is not None and (e0 + e1) > 0:
            ratio = e0 / (e0 + e1)
            ratio_ok = ratio >= self.ratio_min

        return doa_ok, auto_ok, ratio_ok, doa_deg, auto_deg, ratio


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def rms_int16_mono(raw: bytes, channels: int = CHANNELS) -> float:
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    mono = samples[0::channels] if channels > 1 else samples
    n = len(mono)
    if n == 0:
        return 0.0
    acc = sum(s * s for s in mono)
    return math.sqrt(acc / n) / 32768.0


def extract_mono(raw: bytes, channels: int = CHANNELS) -> bytes:
    samples = array.array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    mono = samples[0::channels] if channels > 1 else samples
    return mono.tobytes()


# ---------------------------------------------------------------------------
# Keyboard stop (cross-platform)
# ---------------------------------------------------------------------------
def _key_stop(stop_event: threading.Event) -> None:
    if sys.stdin is None or not sys.stdin.isatty():
        return
    if platform.system().lower().startswith("win"):
        import msvcrt
        msvcrt.getch()
        stop_event.set()
        return
    import select, termios, tty
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
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        except termios.error:
            pass


# ---------------------------------------------------------------------------
# Voice-activated recording
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    received_s: float = 0.0
    saved_s: float = 0.0
    peak_rms: float = 0.0
    doa_rejects: int = 0
    auto_rejects: int = 0
    ratio_rejects: int = 0
    rms_rejects: int = 0
    trigger_count: int = 0
    doa_samples: list[float] = field(default_factory=list)


def voice_record(
    output_dir: str = ".",
    beam_deg: float = 30.0,
    angle_tolerance_deg: float = 25.0,
    ratio_threshold: float = 0.6,
    rms_gate_threshold: float = 0.005,
    dist_rms_threshold: float = 0.01,
    attack_ms: float = 0.0,
    hold_ms: float = 400.0,
    trigger_ms: float = 300.0,
    release_ms: float = 1500.0,
    max_duration_sec: float = 0.0,
    device_hint: Optional[str] = None,
    stop_event: Optional[threading.Event] = None,
) -> Stats:
    """Voice-activated spatial recording.

    State machine:
      IDLE → (N consecutive "good" frames) → RECORDING
      RECORDING → (M consecutive "bad" frames) → IDLE

    A frame is "good" when ALL pass:
      - RMS gate (audio present)
      - Speaker DOA near target angle
      - Auto-select beam DOA near target angle
      - Front/back energy ratio >= ratio_threshold
      - Audio RMS >= dist_rms_threshold (close-range check)
    """
    stats = Stats()

    # --- Ensure output directory ---
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Connect device ---
    dev = find_device()
    if dev is None:
        raise RuntimeError("XVF3800 not found (VID=0x2886 PID=0x001A)")
    ctrl = XVF3800(dev)

    try:
        # --- Configure beams ---
        configure(ctrl, beam_deg)

        # --- Audio input ---
        input_idx = pick_input(device_hint)
        if input_idx is None:
            raise RuntimeError("No USB audio input found. Use --device-hint.")

        dev_info = sd.query_devices(input_idx)
        print(f"Audio input: [{input_idx}] {dev_info['name']}\n")

        # --- Gates and monitor ---
        rms_gate = RmsGate(rms_gate_threshold, attack_ms=attack_ms, hold_ms=hold_ms)
        spatial = SpatialMonitor(ctrl, beam_deg, angle_tolerance_deg, ratio_threshold)
        spatial.start(interval=0.2)

        # --- State machine parameters ---
        sec_per_block = BLOCKSIZE / SAMPLE_RATE
        trigger_blocks = max(1, int(trigger_ms / 1000 / sec_per_block))
        release_blocks = max(1, int(release_ms / 1000 / sec_per_block))
        print(f"Voice trigger: {trigger_ms:.0f}ms ({trigger_blocks} blocks)  |  "
              f"release: {release_ms:.0f}ms ({release_blocks} blocks)")
        print(f"Spatial gate: DOA ±{angle_tolerance_deg:.0f}°  |  "
              f"E0/(E0+E1) >= {ratio_threshold:.1f}  |  RMS >= {dist_rms_threshold:.3f}")
        print(f"Output dir : {output_dir}")
        print(f"Press Ctrl+C to stop.\n")

        # --- Keyboard listener ---
        if stop_event is None:
            stop_event = threading.Event()
        if sys.stdin is not None and sys.stdin.isatty():
            threading.Thread(target=_key_stop, args=(stop_event,), daemon=True).start()

        # --- Audio queue ---
        q: queue.Queue = queue.Queue(maxsize=64)
        t_start = time.monotonic()

        def callback(indata, frames, ti, status):
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            try:
                q.put_nowait(bytes(indata))
            except queue.Full:
                pass

        # --- State machine state ---
        state = "IDLE"  # IDLE | RECORDING
        good_streak = 0
        bad_streak = 0
        wf: Optional[wave.Wave_write] = None
        current_output: str = ""
        session_count = 0
        last_status = t_start
        file_open_time: Optional[float] = None
        last_good_time = t_start

        def _open_wav() -> str:
            nonlocal session_count
            session_count += 1
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = str(pathlib.Path(output_dir) / f"voice_{ts}_{session_count:03d}.wav")
            return path

        def _close_wav() -> None:
            nonlocal wf
            if wf is not None:
                wf.close()
                wf = None

        with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                                dtype="int16", blocksize=BLOCKSIZE,
                                device=input_idx, callback=callback):
            while True:
                if stop_event.is_set():
                    print("\nStopped by user.")
                    break

                elapsed = time.monotonic() - t_start
                if max_duration_sec > 0 and elapsed >= max_duration_sec:
                    print(f"\nMax duration reached ({max_duration_sec:.0f}s).")
                    break

                # If in RECORDING with no file open (file was closed), also check
                # for session timeout to exit
                if state == "RECORDING" and wf is None:
                    # Session already ended, go back to IDLE
                    state = "IDLE"

                try:
                    chunk = q.get(timeout=0.1)
                except queue.Empty:
                    # If recording and no audio for too long, trigger release
                    if state == "RECORDING" and wf is not None:
                        silence_elapsed = time.monotonic() - last_good_time
                        if silence_elapsed > release_ms / 1000:
                            bad_streak = release_blocks  # force release
                    continue

                stats.received_s += BLOCKSIZE / SAMPLE_RATE
                level = rms_int16_mono(chunk)
                if level > stats.peak_rms:
                    stats.peak_rms = level

                # --- Evaluate all gates ---
                rms_open = rms_gate.update(level)
                doa_ok, auto_ok, ratio_ok, doa_deg, auto_deg, ratio = spatial.check()
                dist_ok = level >= dist_rms_threshold if rms_open else True

                if doa_deg is not None:
                    stats.doa_samples.append(doa_deg)

                # A frame is "spatially good" when all checks pass
                frame_good = rms_open and doa_ok and auto_ok and ratio_ok and dist_ok

                if not doa_ok:
                    stats.doa_rejects += 1
                if not auto_ok:
                    stats.auto_rejects += 1
                if not ratio_ok:
                    stats.ratio_rejects += 1
                if not dist_ok and rms_open:
                    stats.rms_rejects += 1

                # --- State machine ---
                if frame_good:
                    good_streak += 1
                    bad_streak = 0
                    last_good_time = time.monotonic()
                else:
                    bad_streak += 1
                    good_streak = 0

                if state == "IDLE":
                    if good_streak >= trigger_blocks:
                        state = "RECORDING"
                        current_output = _open_wav()
                        wf = wave.open(current_output, "wb")
                        wf.setnchannels(1)
                        wf.setsampwidth(SAMPLE_WIDTH)
                        wf.setframerate(SAMPLE_RATE)
                        file_open_time = time.monotonic()
                        stats.trigger_count += 1
                        print(f"\n>>> VOICE DETECTED — recording → {pathlib.Path(current_output).name}")
                        print(f"    DOA={doa_deg:.1f}°  auto={auto_deg:.1f}°  "
                              f"ratio={ratio:.2f}  RMS={level:.4f}")
                        good_streak = 0

                elif state == "RECORDING":
                    if bad_streak >= release_blocks:
                        state = "IDLE"
                        if wf is not None:
                            dur = time.monotonic() - (file_open_time or t_start)
                            _close_wav()
                            print(f"<<< SILENCE — saved {dur:.1f}s → {pathlib.Path(current_output).name}")
                            stats.saved_s += dur
                        bad_streak = 0

                # --- Write audio ---
                if state == "RECORDING" and wf is not None:
                    mono = extract_mono(chunk)
                    wf.writeframesraw(mono)

                # --- Periodic status ---
                now = time.monotonic()
                if now - last_status >= 2.0:
                    parts = [f"  [{elapsed:5.1f}s]  state={state}  RMS={level:.4f}"]
                    if doa_deg is not None:
                        parts.append(f"DOA={doa_deg:.0f}°")
                    else:
                        parts.append("DOA=--")
                    if auto_deg is not None:
                        parts.append(f"auto={auto_deg:.0f}°")
                    if ratio is not None:
                        parts.append(f"ratio={ratio:.2f}")
                    parts.append(f"good={good_streak}/{trigger_blocks}"
                                 if state == "IDLE" else f"bad={bad_streak}/{release_blocks}")
                    print("  ".join(parts))
                    last_status = now

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        spatial.stop()
        _close_wav()
        ctrl.close()

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="XVF3800 voice-triggered spatial recording (no calibration needed)")
    p.add_argument("--angle", type=float, default=30.0,
                   help="Target beam azimuth in degrees (default: 30)")
    p.add_argument("--angle-tolerance", type=float, default=25.0,
                   help="Allowed DOA deviation ±degrees (default: 25)")
    p.add_argument("--ratio-threshold", type=float, default=0.6,
                   help="Min front/back energy ratio E0/(E0+E1) (default: 0.6)")
    p.add_argument("--rms-gate", type=float, default=0.005,
                   help="RMS noise gate threshold 0-1 (default: 0.005)")
    p.add_argument("--dist-rms", type=float, default=0.01,
                   help="Min RMS for close-range voice 0-1 (default: 0.01)")
    p.add_argument("--attack-ms", type=float, default=0.0,
                   help="Gate attack time ms (default: 0)")
    p.add_argument("--hold-ms", type=float, default=400.0,
                   help="Gate hold time ms (default: 400)")
    p.add_argument("--trigger-ms", type=float, default=300.0,
                   help="Voice presence required before recording starts (default: 300ms)")
    p.add_argument("--release-ms", type=float, default=1500.0,
                   help="Silence required before recording stops (default: 1500ms)")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Max total duration in seconds (0=unlimited)")
    p.add_argument("--output-dir", default=".",
                   help="Output directory for WAV files (default: current dir)")
    p.add_argument("--device-hint", default=None,
                   help="Substring to match audio input device name")
    args = p.parse_args()

    print("=" * 60)
    print("  XVF3800 Voice-Triggered Spatial Recording")
    print(f"  Target: {args.angle:.0f}° ± {args.angle_tolerance:.0f}°")
    print(f"  Beam ratio: E0/(E0+E1) >= {args.ratio_threshold:.1f}")
    print(f"  Close-range RMS: >= {args.dist_rms:.3f}")
    print("=" * 60)

    try:
        stats = voice_record(
            output_dir=args.output_dir,
            beam_deg=args.angle,
            angle_tolerance_deg=args.angle_tolerance,
            ratio_threshold=args.ratio_threshold,
            rms_gate_threshold=args.rms_gate,
            dist_rms_threshold=args.dist_rms,
            attack_ms=args.attack_ms,
            hold_ms=args.hold_ms,
            trigger_ms=args.trigger_ms,
            release_ms=args.release_ms,
            max_duration_sec=args.duration,
            device_hint=args.device_hint,
        )
    except RuntimeError as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1

    print(f"\n{'─' * 40}")
    print(f"  Received : {stats.received_s:.1f}s")
    print(f"  Saved    : {stats.saved_s:.1f}s  ({stats.trigger_count} sessions)")
    print(f"  Peak RMS : {stats.peak_rms:.4f}")
    if stats.doa_rejects or stats.auto_rejects or stats.ratio_rejects or stats.rms_rejects:
        print(f"  Rejects: DOA={stats.doa_rejects}  auto={stats.auto_rejects}  "
              f"ratio={stats.ratio_rejects}  RMSdist={stats.rms_rejects}")
    if stats.doa_samples:
        import statistics
        d = stats.doa_samples
        print(f"  DOA mean : {statistics.mean(d):.1f}°  "
              f"(min={min(d):.1f}°, max={max(d):.1f}°)")
    print(f"{'─' * 40}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
