"""Record beamformed audio from a ReSpeaker XVF3800.

This script:
1. Configures the XVF3800 fixed beam to point at 30 degrees.
2. Routes the processed beamformed output to the left USB audio channel.
3. Records up to 30 seconds, or stops earlier when the user presses any key.
4. Keeps only chunks whose RMS is above a configurable threshold.
5. Writes the kept audio as a 16 kHz mono WAV file.

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
import sys
import threading
import time
import wave
from dataclasses import dataclass
from typing import Optional

try:
    import sounddevice as sd
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        "Missing dependency: sounddevice. Install it with: pip install sounddevice"
    ) from exc

try:
    import usb.core
    import usb.util
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        "Missing dependency: pyusb. Install it with: pip install pyusb"
    ) from exc

try:
    import libusb_package  # type: ignore
except Exception:  # pragma: no cover - optional on non-Windows systems
    libusb_package = None


VID = 0x2886
PID = 0x001A
SAMPLE_RATE = 16_000
CHANNELS = 2
SAMPLE_WIDTH_BYTES = 2  # int16 PCM
DEFAULT_DURATION_SEC = 30.0
DEFAULT_BEAM_AZIMUTH_DEG = 30.0


# Commands taken from the official ReSpeaker XVF3800 Python host control SDK.
# See: https://github.com/respeaker/reSpeaker_XVF3800_USB_4MIC_ARRAY/blob/master/python_control/xvf_host.py
PARAMETERS = {
    "AEC_FIXEDBEAMSAZIMUTH_VALUES": (33, 81, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSELEVATION_VALUES": (33, 82, 2, "rw", "radians"),
    "AEC_FIXEDBEAMSONOFF": (33, 37, 1, "rw", "int32"),
    "AEC_FIXEDBEAMSGATING": (33, 83, 1, "rw", "uint8"),
    "AUDIO_MGR_OP_L": (35, 15, 2, "rw", "uint8"),
    "AUDIO_MGR_OP_R": (35, 19, 2, "rw", "uint8"),
    "AEC_AZIMUTH_VALUES": (33, 75, 4, "ro", "radians"),
    "AEC_SPENERGY_VALUES": (33, 80, 4, "ro", "float"),
}


class ReSpeakerControl:
    """Minimal USB control wrapper for XVF3800.

    This is a small subset of the official xvf_host.py behavior, enough for
    setting the beam and routing the processed output.
    """

    TIMEOUT_MS = 100_000

    def __init__(self, dev: usb.core.Device):
        self.dev = dev

    def write(self, name: str, values: list[float | int]) -> None:
        resid, cmdid, count, access, value_type = PARAMETERS[name]
        if access == "ro":
            raise ValueError(f"{name} is read-only")
        if len(values) != count:
            raise ValueError(f"{name} expects {count} values, got {len(values)}")

        payload = bytearray()
        if value_type in ("float", "radians"):
            for value in values:
                payload += struct_pack_float(float(value))
        elif value_type == "uint8":
            for value in values:
                payload += int(value).to_bytes(1, "little", signed=False)
        elif value_type == "int32":
            for value in values:
                payload += int(value).to_bytes(4, "little", signed=True)
        else:
            raise ValueError(f"Unsupported type: {value_type}")

        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            cmdid,
            resid,
            payload,
            self.TIMEOUT_MS,
        )

    def read(self, name: str):
        resid, cmdid, count, access, value_type = PARAMETERS[name]
        if access == "wo":
            raise ValueError(f"{name} is write-only")

        read_cmdid = 0x80 | cmdid
        if value_type in ("uint8", "char"):
            length = count + 1
        elif value_type in ("float", "radians", "uint32", "int32"):
            length = count * 4 + 1
        elif value_type == "uint16":
            length = count * 2 + 1
        else:
            raise ValueError(f"Unsupported type: {value_type}")

        response = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            read_cmdid,
            resid,
            length,
            self.TIMEOUT_MS,
        )

        if value_type == "uint16":
            raw = response.tobytes()[1:]
            return tuple(int.from_bytes(raw[i : i + 2], "little") for i in range(0, len(raw), 2))
        if value_type in ("float", "radians"):
            raw = response.tobytes()[1:]
            return tuple(float_from_bytes(raw[i : i + 4]) for i in range(0, len(raw), 4))
        if value_type == "uint8":
            return tuple(response.tolist()[1:])

        raise ValueError(f"Unsupported read type: {value_type}")

    def close(self) -> None:
        usb.util.dispose_resources(self.dev)


def float_from_bytes(data: bytes) -> float:
    import struct

    return struct.unpack("<f", data)[0]


def struct_pack_float(value: float) -> bytes:
    import struct

    return struct.pack("<f", value)


def find_device(vid: int = VID, pid: int = PID) -> Optional[usb.core.Device]:
    if platform.system().lower().startswith("win") and libusb_package is not None:
        return libusb_package.find(idVendor=vid, idProduct=pid)
    return usb.core.find(idVendor=vid, idProduct=pid)


def radians(deg: float) -> float:
    return deg * math.pi / 180.0


def pick_input_device(device_hint: Optional[str]) -> Optional[int]:
    """Return a sounddevice input device index matching the hint."""

    devices = sd.query_devices()
    if device_hint is None:
        hints = ("reSpeaker", "XVF3800", "USB Audio", "USB Audio CODEC")
    else:
        hints = (device_hint,)

    for idx, dev in enumerate(devices):
        if dev["max_input_channels"] <= 0:
            continue
        name = dev["name"]
        if any(h.lower() in name.lower() for h in hints):
            return idx
    return None


def rms_int16_mono(raw_bytes: bytes, channels: int = CHANNELS) -> float:
    """Compute normalized RMS from interleaved int16 PCM."""

    samples = array.array("h")
    samples.frombytes(raw_bytes)

    if sys.byteorder != "little":
        samples.byteswap()

    if channels > 1:
        mono = samples[0::channels]
    else:
        mono = samples

    if len(mono) == 0:
        return 0.0

    acc = 0
    for sample in mono:
        acc += sample * sample
    return math.sqrt(acc / len(mono)) / 32768.0


def extract_mono_bytes(raw_bytes: bytes, channels: int = CHANNELS) -> bytes:
    samples = array.array("h")
    samples.frombytes(raw_bytes)
    if sys.byteorder != "little":
        samples.byteswap()
    if channels > 1:
        mono = samples[0::channels]
    else:
        mono = samples
    return mono.tobytes()


def any_key_stop(stop_event: threading.Event) -> None:
    """Set stop_event when the user presses any key."""

    if not sys.stdin.isatty():
        return

    if platform.system().lower().startswith("win"):
        import msvcrt

        msvcrt.getch()
        stop_event.set()
        return

    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if ready:
                sys.stdin.read(1)
                stop_event.set()
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


@dataclass
class RecordStats:
    received_chunks: int = 0
    saved_chunks: int = 0
    received_frames: int = 0
    saved_frames: int = 0


def configure_device(ctrl: ReSpeakerControl, beam_deg: float) -> None:
    """Set fixed beam mode and route the beamformed output to the left channel."""

    beam_rad = radians(beam_deg)

    # Beam 1 points at the requested angle. Beam 2 is placed opposite it so the
    # device still has a meaningful second fixed beam if the firmware consults it.
    ctrl.write("AEC_FIXEDBEAMSAZIMUTH_VALUES", [beam_rad, radians((beam_deg + 180.0) % 360.0)])
    ctrl.write("AEC_FIXEDBEAMSELEVATION_VALUES", [0.0, 0.0])
    ctrl.write("AEC_FIXEDBEAMSGATING", [0])
    ctrl.write("AEC_FIXEDBEAMSONOFF", [1])

    # Route the left USB channel to the first processed beam output.
    # This is the clean beamformed stream we write to file.
    ctrl.write("AUDIO_MGR_OP_L", [6, 0])

    # Silence the right channel to keep the host-side capture simple.
    ctrl.write("AUDIO_MGR_OP_R", [0, 0])


def record(
    output_path: str,
    duration_sec: float = DEFAULT_DURATION_SEC,
    rms_threshold: float = 0.02,
    device_hint: Optional[str] = None,
    beam_deg: float = DEFAULT_BEAM_AZIMUTH_DEG,
) -> RecordStats:
    dev = find_device()
    if not dev:
        raise RuntimeError(
            "No XVF3800 control device found. "
            "Check that the microphone is connected and the libusb driver/permissions are correct."
        )

    ctrl = ReSpeakerControl(dev)
    try:
        configure_device(ctrl, beam_deg)
        print("AEC_AZIMUTH_VALUES:", ctrl.read("AEC_AZIMUTH_VALUES"))
        print("AEC_SPENERGY_VALUES:", ctrl.read("AEC_SPENERGY_VALUES"))

        input_index = pick_input_device(device_hint)
        if input_index is None:
            devices = sd.query_devices()
            details = "\n".join(
                f"{i}: {d['name']} (in={d['max_input_channels']}, default_sr={d['default_samplerate']})"
                for i, d in enumerate(devices)
                if d["max_input_channels"] > 0
            )
            raise RuntimeError(
                "No matching USB audio input device found.\n"
                "Pass --device-hint with a device name fragment, or pick an index manually.\n"
                f"Available input devices:\n{details}"
            )

        stop_event = threading.Event()
        keyboard_thread = threading.Thread(target=any_key_stop, args=(stop_event,), daemon=True)
        keyboard_thread.start()

        q: queue.Queue[bytes] = queue.Queue(maxsize=32)
        stats = RecordStats()
        start = time.monotonic()

        def callback(indata, frames, _time_info, status) -> None:
            if status:
                print(f"Audio status: {status}", file=sys.stderr)
            try:
                q.put_nowait(bytes(indata))
            except queue.Full:
                # Drop very old audio if the main loop falls behind.
                pass

        print(f"Recording to {output_path}")
        print(f"Beam: {beam_deg:.1f} degrees")
        print(f"RMS threshold: {rms_threshold:.4f} (normalized int16 RMS)")
        print("Press any key to stop.")

        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(SAMPLE_WIDTH_BYTES)
            wf.setframerate(SAMPLE_RATE)

            with sd.RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=1024,
                device=input_index,
            ):
                while True:
                    if stop_event.is_set():
                        break
                    if time.monotonic() - start >= duration_sec:
                        break
                    try:
                        chunk = q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    stats.received_chunks += 1
                    stats.received_frames += len(chunk) // (SAMPLE_WIDTH_BYTES * CHANNELS)

                    level = rms_int16_mono(chunk, channels=CHANNELS)
                    if level >= rms_threshold:
                        mono = extract_mono_bytes(chunk, channels=CHANNELS)
                        wf.writeframesraw(mono)
                        stats.saved_chunks += 1
                        stats.saved_frames += len(mono) // SAMPLE_WIDTH_BYTES

        return stats
    finally:
        ctrl.close()


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Record XVF3800 beamformed audio to WAV")
    parser.add_argument("-o", "--output", default="respeaker_30deg_close.wav", help="output WAV path")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC, help="max record time in seconds")
    parser.add_argument(
        "--rms-threshold",
        type=float,
        default=0.02,
        help="normalized RMS threshold; keep chunks at or above this level",
    )
    parser.add_argument(
        "--device-hint",
        default=None,
        help="substring used to choose the USB audio input device automatically",
    )
    parser.add_argument(
        "--beam-deg",
        type=float,
        default=DEFAULT_BEAM_AZIMUTH_DEG,
        help="fixed beam azimuth in degrees",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = record(
        output_path=args.output,
        duration_sec=args.duration,
        rms_threshold=args.rms_threshold,
        device_hint=args.device_hint,
        beam_deg=args.beam_deg,
    )

    print(
        f"Done. Saved {stats.saved_frames / SAMPLE_RATE:.2f}s "
        f"from {stats.received_frames / SAMPLE_RATE:.2f}s of captured audio."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
