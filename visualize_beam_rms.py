"""Visualize XVF3800 beam RMS values on a polar plot.

Inputs:
  - CSV file produced by calibrate_xvf3800_30deg_05m.py
  - WAV directory containing the referenced WAV files

Behavior:
  - Reads Peak RMS values from the CSV.
  - Assigns angles uniformly across 0..360 degrees by row order.
  - Draws a polar plot and highlights the maximum RMS point.
  - Saves a PNG in the current working directory.

Example:
  python visualize_beam_rms.py --csv calibration_30deg_0.5m.csv --wav-dir .
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass
class BeamRow:
    index: int
    filename: str
    peak_rms: float
    angle_deg: float
    wav_path: pathlib.Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize beam RMS as a polar plot")
    parser.add_argument("--csv", required=True, help="CSV file with Peak RMS values")
    parser.add_argument("--wav-dir", required=True, help="Directory containing the WAV files")
    parser.add_argument(
        "--output",
        default=None,
        help="Output PNG path (default: derived from CSV name in current directory)",
    )
    return parser.parse_args()


def read_rows(csv_path: pathlib.Path, wav_dir: pathlib.Path) -> list[BeamRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not wav_dir.exists():
        raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

    rows_raw: list[dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header row")
        rows_raw = list(reader)

    if not rows_raw:
        raise ValueError("CSV contains no data rows")

    peak_key = None
    file_key = None
    index_key = None

    for candidate in ("Peak RMS", "peak_rms", "PeakRMS"):
        if candidate in reader.fieldnames:
            peak_key = candidate
            break
    for candidate in ("文件名", "filename", "File", "file"):
        if candidate in reader.fieldnames:
            file_key = candidate
            break
    for candidate in ("测试序号", "index", "Index", "No"):
        if candidate in reader.fieldnames:
            index_key = candidate
            break

    if peak_key is None:
        raise ValueError(f"Could not find Peak RMS column in CSV headers: {reader.fieldnames}")
    if file_key is None:
        raise ValueError(f"Could not find filename column in CSV headers: {reader.fieldnames}")

    count = len(rows_raw)
    angle_step = 360.0 / count
    rows: list[BeamRow] = []

    for i, row in enumerate(rows_raw):
        filename = row[file_key].strip()
        wav_path = wav_dir / filename
        if not wav_path.exists():
            raise FileNotFoundError(f"WAV file listed in CSV not found: {wav_path}")

        try:
            peak_rms = float(row[peak_key])
        except ValueError as exc:
            raise ValueError(f"Invalid Peak RMS value in row {i + 1}: {row[peak_key]}") from exc

        if index_key and row.get(index_key):
            try:
                idx = int(float(row[index_key]))
            except ValueError:
                idx = i + 1
        else:
            idx = i + 1

        angle_deg = (i * angle_step) % 360.0
        rows.append(
            BeamRow(
                index=idx,
                filename=filename,
                peak_rms=peak_rms,
                angle_deg=angle_deg,
                wav_path=wav_path,
            )
        )

    return rows


def plot_polar(rows: list[BeamRow], csv_path: pathlib.Path, output_path: pathlib.Path) -> None:
    angles_rad = [math.radians(r.angle_deg) for r in rows]
    rms_values = [r.peak_rms for r in rows]

    max_idx = max(range(len(rows)), key=lambda i: rows[i].peak_rms)
    max_row = rows[max_idx]

    fig = plt.figure(figsize=(9, 8), dpi=160)
    ax = fig.add_subplot(111, projection="polar")

    # Make the polar chart start at 0 degrees on the right and go clockwise.
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    # Plot the RMS curve and markers.
    ax.plot(angles_rad, rms_values, linewidth=2.2, color="#1f77b4")
    ax.scatter(angles_rad, rms_values, s=45, color="#1f77b4", zorder=3)

    # Highlight the maximum point.
    ax.scatter([math.radians(max_row.angle_deg)], [max_row.peak_rms], s=120, color="#d62728", zorder=4)
    ax.annotate(
        f"Max RMS\n{max_row.peak_rms:.4f}\n{max_row.angle_deg:.1f}°",
        xy=(math.radians(max_row.angle_deg), max_row.peak_rms),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=10,
        color="#d62728",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
    )

    # Reference line at 30 degrees for quick verification.
    ax.plot([math.radians(30.0), math.radians(30.0)], [0, max(rms_values) * 1.05], linestyle="--", color="#2ca02c", linewidth=1.5)
    ax.text(math.radians(30.0), max(rms_values) * 1.08, "30° target", color="#2ca02c", fontsize=9, ha="left", va="bottom")

    ax.set_title(f"XVF3800 Beam RMS Polar Plot\n{csv_path.name}", pad=18)
    ax.set_rlabel_position(135)
    ax.set_ylim(0, max(rms_values) * 1.15 if max(rms_values) > 0 else 1.0)
    ax.grid(True, alpha=0.35)

    # Show every recording angle on the theta grid.
    theta_ticks_deg = [r.angle_deg for r in rows]
    theta_tick_labels = [f"{int(round(a))}°" for a in theta_ticks_deg]
    ax.set_xticks([math.radians(a) for a in theta_ticks_deg])
    ax.set_xticklabels(theta_tick_labels)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    csv_path = pathlib.Path(args.csv).resolve()
    wav_dir = pathlib.Path(args.wav_dir).resolve()
    output_path = pathlib.Path(args.output).resolve() if args.output else pathlib.Path(f"{csv_path.stem}_polar.png").resolve()

    try:
        rows = read_rows(csv_path, wav_dir)
        plot_polar(rows, csv_path, output_path)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    max_row = max(rows, key=lambda r: r.peak_rms)
    print(f"PNG saved: {output_path}")
    print(f"Max RMS: {max_row.peak_rms:.4f} at {max_row.angle_deg:.1f}° ({max_row.filename})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
