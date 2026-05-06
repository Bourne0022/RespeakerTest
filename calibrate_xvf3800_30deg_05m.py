"""Batch calibration recorder for ReSpeaker XVF3800.

This script reuses the existing `respeaker_xvf3800_beam_record.record()`
function without modifying the original file.

It performs multiple short recordings, stores one WAV per run, collects the
Peak RMS reported by the recorder, and writes a CSV summary.

Example:
    python calibrate_xvf3800_30deg_05m.py --count 5 --duration 5 --playback
"""

from __future__ import annotations

import csv
import contextlib
import io
import pathlib
import sys
import time
import wave
from dataclasses import dataclass

import respeaker_xvf3800_beam_record as recorder

try:
    import winsound
except ImportError:  # pragma: no cover
    winsound = None


DEFAULT_COUNT = 5
DEFAULT_DURATION_SEC = 5.0
DEFAULT_BEAM_DEG = 30.0
DEFAULT_RMS_THRESHOLD = 0.03
DEFAULT_ATTACK_MS = 40.0
DEFAULT_HOLD_MS = 400.0
DEFAULT_DEVICE_HINT = "XVF3800"
DEFAULT_NOTE = "距离约 0.5m"
DEFAULT_OUTPUT_PREFIX = "calibration_30deg_0p5m"
DEFAULT_CSV = "calibration_30deg_0.5m.csv"


@dataclass
class RunResult:
    index: int
    filename: str
    wav_path: pathlib.Path
    peak_rms: float
    wav_duration_sec: float
    beam_deg: float
    note: str


def safe_capture_record(
    output_path: str,
    duration_sec: float,
    rms_threshold: float,
    beam_deg: float,
    attack_ms: float,
    hold_ms: float,
    device_hint: str,
):
    """Run the existing recorder while suppressing its verbose console output."""

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        stats = recorder.record(
            output_path=output_path,
            duration_sec=duration_sec,
            rms_threshold=rms_threshold,
            device_hint=device_hint,
            beam_deg=beam_deg,
            attack_ms=attack_ms,
            hold_ms=hold_ms,
        )
    return stats


def wav_duration(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def play_wav(path: pathlib.Path) -> str:
    if winsound is None:
        return "SKIPPED: winsound unavailable"
    try:
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return "OK"
    except Exception as exc:  # pragma: no cover
        return f"FAILED: {exc}"


def write_csv(csv_path: pathlib.Path, rows: list[RunResult]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["文件名", "Peak RMS", "测试序号", "波束角度", "备注"])
        for row in rows:
            writer.writerow(
                [
                    row.filename,
                    f"{row.peak_rms:.4f}",
                    row.index,
                    f"{row.beam_deg:.0f}°",
                    row.note,
                ]
            )


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="XVF3800 calibration recorder")
    p.add_argument("--count", type=int, default=DEFAULT_COUNT, help="number of recordings")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION_SEC, help="seconds per recording")
    p.add_argument("--beam-deg", type=float, default=DEFAULT_BEAM_DEG, help="fixed beam angle in degrees")
    p.add_argument("--rms-threshold", type=float, default=DEFAULT_RMS_THRESHOLD, help="RMS gate threshold")
    p.add_argument("--attack-ms", type=float, default=DEFAULT_ATTACK_MS, help="gate attack time")
    p.add_argument("--hold-ms", type=float, default=DEFAULT_HOLD_MS, help="gate hold time")
    p.add_argument("--device-hint", default=DEFAULT_DEVICE_HINT, help="substring used to pick the XVF3800 input device")
    p.add_argument("--note", default=DEFAULT_NOTE, help="text written into the CSV note column")
    p.add_argument("--prefix", default=DEFAULT_OUTPUT_PREFIX, help="WAV filename prefix")
    p.add_argument("--csv", default=DEFAULT_CSV, help="output CSV path")
    p.add_argument("--playback", action="store_true", help="play each WAV after recording")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        print("ERROR: --count must be positive", file=sys.stderr)
        return 1
    if args.duration < 5.0:
        print("ERROR: --duration should be at least 5 seconds", file=sys.stderr)
        return 1
    if not 0.0 <= args.rms_threshold <= 1.0:
        print("ERROR: --rms-threshold must be between 0.0 and 1.0", file=sys.stderr)
        return 1

    results: list[RunResult] = []
    csv_path = pathlib.Path(args.csv).resolve()

    try:
        for i in range(1, args.count + 1):
            wav_name = f"{args.prefix}_{i:02d}.wav"
            wav_path = pathlib.Path(wav_name).resolve()

            stats = safe_capture_record(
                output_path=str(wav_path),
                duration_sec=args.duration,
                rms_threshold=args.rms_threshold,
                beam_deg=args.beam_deg,
                attack_ms=args.attack_ms,
                hold_ms=args.hold_ms,
                device_hint=args.device_hint,
            )

            if not wav_path.exists():
                raise RuntimeError(f"WAV file was not created: {wav_path.name}")

            duration = wav_duration(wav_path)
            result = RunResult(
                index=i,
                filename=wav_path.name,
                wav_path=wav_path,
                peak_rms=stats.peak_rms,
                wav_duration_sec=duration,
                beam_deg=args.beam_deg,
                note=args.note,
            )
            results.append(result)

            if args.playback:
                playback_status = play_wav(wav_path)
                print(f"[{i}/{args.count}] {wav_path.name} playback: {playback_status}")
            else:
                print(f"[{i}/{args.count}] {wav_path.name} recorded")

        write_csv(csv_path, results)

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nCSV path: {csv_path}")
    print("Measured Peak RMS values:")
    for row in results:
        print(f"  {row.index:02d}: {row.filename} -> {row.peak_rms:.4f} ({row.wav_duration_sec:.2f} s)")

    print("\nCSV rows:")
    for row in results:
        print(f"  {row.index:02d}, {row.filename}, {row.peak_rms:.4f}, {row.beam_deg:.0f}°, {row.note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
