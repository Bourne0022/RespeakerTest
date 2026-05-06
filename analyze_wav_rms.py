"""Analyze WAV RMS in fixed time blocks.

Features:
- Prints per-block RMS using 100 ms blocks by default.
- Marks each block as PASS/DROP against a threshold.
- Summarizes total duration, peak RMS, and gated-out spans.
- Supports multiple WAV files in one run.
- Optionally extracts angle/distance labels from filenames with a regex.

Examples:
    python analyze_wav_rms.py test.wav --threshold 0.06
    python analyze_wav_rms.py *.wav --threshold 0.06 --label-regex "(?P<angle>\\d+)deg_(?P<distance>[0-9.]+)m"
"""

from __future__ import annotations

import argparse
import array
import glob
import math
import pathlib
import re
import sys
import wave
from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass
class BlockStat:
    index: int
    start_sec: float
    end_sec: float
    rms: float
    passed: bool


@dataclass
class FileSummary:
    path: pathlib.Path
    label: str
    duration_sec: float
    peak_rms: float
    mean_rms: float
    pass_ratio: float
    blocks: list[BlockStat]
    truncated_spans: list[tuple[int, int]]


def expand_inputs(items: Sequence[str]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for item in items:
        matches = [pathlib.Path(p) for p in glob.glob(item)]
        if matches:
            paths.extend(matches)
        else:
            paths.append(pathlib.Path(item))
    # Keep unique paths in order.
    seen = set()
    unique: list[pathlib.Path] = []
    for p in paths:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def read_pcm_samples(wf: wave.Wave_read, frame_count: int) -> bytes:
    return wf.readframes(frame_count)


def rms_from_pcm(raw: bytes, sampwidth: int) -> float:
    """Return normalized RMS in [0, 1] for common PCM WAV sample widths."""

    if not raw:
        return 0.0

    if sampwidth == 1:
        # 8-bit PCM is unsigned.
        samples = [(b - 128) / 128.0 for b in raw]
        return math.sqrt(sum(s * s for s in samples) / len(samples))

    if sampwidth == 2:
        samples = array.array("h")
        samples.frombytes(raw)
        if sys.byteorder != "little":
            samples.byteswap()
        return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768.0

    if sampwidth == 3:
        values = []
        for i in range(0, len(raw) - (len(raw) % 3), 3):
            b = raw[i : i + 3]
            sign = 0xFF if b[2] & 0x80 else 0x00
            v = int.from_bytes(b + bytes([sign]), "little", signed=True)
            values.append(v / 8388608.0)
        return math.sqrt(sum(s * s for s in values) / len(values)) if values else 0.0

    if sampwidth == 4:
        values = array.array("i")
        values.frombytes(raw)
        if sys.byteorder != "little":
            values.byteswap()
        return math.sqrt(sum(s * s for s in values) / len(values)) / 2147483648.0

    raise ValueError(f"Unsupported sample width: {sampwidth} bytes")


def span_string(blocks: list[BlockStat], start_idx: int, end_idx: int) -> str:
    start = blocks[start_idx].start_sec
    end = blocks[end_idx].end_sec
    return f"{start:.2f}-{end:.2f}s (blocks {blocks[start_idx].index}-{blocks[end_idx].index})"


def merge_truncated_spans(blocks: list[BlockStat]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    run_start = None
    for i, block in enumerate(blocks):
        if not block.passed:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            spans.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        spans.append((run_start, len(blocks) - 1))
    return spans


def analyze_file(path: pathlib.Path, threshold: float, block_ms: int) -> FileSummary:
    if not path.exists():
        raise FileNotFoundError(path)

    with wave.open(str(path), "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        total_frames = wf.getnframes()

        block_frames = max(1, int(round(framerate * block_ms / 1000.0)))
        blocks: list[BlockStat] = []
        all_rms: list[float] = []

        index = 0
        while True:
            raw = read_pcm_samples(wf, block_frames)
            if not raw:
                break
            index += 1
            frames = len(raw) // (sampwidth * nchannels)
            start_sec = (index - 1) * block_frames / framerate
            end_sec = min(total_frames / framerate, start_sec + frames / framerate)
            rms = rms_from_pcm(raw, sampwidth)
            passed = rms >= threshold
            blocks.append(BlockStat(index, start_sec, end_sec, rms, passed))
            all_rms.append(rms)

    peak_rms = max(all_rms) if all_rms else 0.0
    mean_rms = sum(all_rms) / len(all_rms) if all_rms else 0.0
    pass_count = sum(1 for b in blocks if b.passed)
    pass_ratio = pass_count / len(blocks) if blocks else 0.0
    truncated_spans = merge_truncated_spans(blocks)
    label = path.stem

    return FileSummary(
        path=path,
        label=label,
        duration_sec=total_frames / framerate,
        peak_rms=peak_rms,
        mean_rms=mean_rms,
        pass_ratio=pass_ratio,
        blocks=blocks,
        truncated_spans=truncated_spans,
    )


def apply_label_regex(label: str, regex: str | None) -> str:
    if not regex:
        return label
    m = re.search(regex, label)
    if not m:
        return label
    if m.groupdict():
        parts = [f"{k}={v}" for k, v in m.groupdict().items() if v is not None]
        return ", ".join(parts) if parts else label
    return m.group(0)


def print_summary(summary: FileSummary, threshold: float, block_ms: int, label_regex: str | None) -> None:
    label = apply_label_regex(summary.label, label_regex)
    print(f"\n== {summary.path.name} ==")
    if label != summary.label:
        print(f"Label: {label}")
    print(f"Total duration: {summary.duration_sec:.3f} s")
    print(f"Peak RMS: {summary.peak_rms:.4f}")
    print(f"Mean RMS: {summary.mean_rms:.4f}")
    print(f"Gate threshold: {threshold:.4f}")
    print(f"Block size: {block_ms} ms")
    print(f"Pass ratio: {summary.pass_ratio * 100:.1f}%")

    print("\nPer-block RMS:")
    for b in summary.blocks:
        status = "PASS" if b.passed else "DROP"
        print(f"{b.index:04d}  {b.start_sec:7.2f}-{b.end_sec:7.2f}s  RMS={b.rms:.4f}  {status}")

    print("\nTruncated spans:")
    if summary.truncated_spans:
        for start_idx, end_idx in summary.truncated_spans:
            print(f"- {span_string(summary.blocks, start_idx, end_idx)}")
    else:
        print("- none")


def print_multi_file_table(summaries: list[FileSummary], label_regex: str | None) -> None:
    print("\n== Multi-file summary ==")
    header = f"{'File':<28} {'Label':<24} {'Dur(s)':>8} {'PeakRMS':>10} {'MeanRMS':>10} {'Pass%':>8}"
    print(header)
    print("-" * len(header))
    for s in summaries:
        label = apply_label_regex(s.label, label_regex)
        print(
            f"{s.path.name:<28} {label[:24]:<24} "
            f"{s.duration_sec:8.3f} {s.peak_rms:10.4f} {s.mean_rms:10.4f} {s.pass_ratio * 100:8.1f}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze WAV RMS in 100 ms blocks")
    p.add_argument(
        "files",
        nargs="+",
        help="One or more WAV files, or glob patterns such as *.wav",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.06,
        help="Normalized RMS threshold used for pass/drop gating (default: 0.06)",
    )
    p.add_argument(
        "--block-ms",
        type=int,
        default=100,
        help="Block size in milliseconds (default: 100)",
    )
    p.add_argument(
        "--label-regex",
        default=None,
        help=(
            "Optional regex to extract angle/distance labels from filenames. "
            "Use named groups like (?P<angle>...) and (?P<distance>...)"
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        print("ERROR: --threshold must be between 0.0 and 1.0", file=sys.stderr)
        return 1
    if args.block_ms <= 0:
        print("ERROR: --block-ms must be positive", file=sys.stderr)
        return 1

    paths = expand_inputs(args.files)
    if not paths:
        print("ERROR: no input files found", file=sys.stderr)
        return 1

    summaries: list[FileSummary] = []
    for path in paths:
        try:
            summary = analyze_file(path, args.threshold, args.block_ms)
        except Exception as exc:
            print(f"ERROR: {path}: {exc}", file=sys.stderr)
            return 1
        summaries.append(summary)
        print_summary(summary, args.threshold, args.block_ms, args.label_regex)

    if len(summaries) > 1:
        print_multi_file_table(summaries, args.label_regex)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
