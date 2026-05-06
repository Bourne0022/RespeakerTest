"""Test script for XVF3800 beamformed recording with spatial filtering.

Quick start:
  1. Calibrate (place sound source at 0.5m, 30°):
     python test.py calibrate --distance 0.5 --angle 30

  2. Record with spatial filter:
     python test.py record --duration 5

  3. Analyze existing WAV:
     python test.py analyze test.wav --threshold 0.05
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run calibration to measure reference speech energy at the target position."""
    import respeaker_xvf3800_beam_record as r

    print("=" * 58)
    print("  CALIBRATION MODE")
    print(f"  Place sound source at {args.distance:.1f}m, {args.angle:.0f}°")
    print(f"  Recording for {args.duration}s ...")
    print("=" * 58)

    stats = r.record(
        output_path="calibration_temp.wav",
        duration_sec=args.duration,
        rms_threshold=0.005,
        device_hint=args.device_hint,
        beam_deg=args.angle,
        angle_tolerance_deg=180.0,     # disable DOA gate
        ref_energy=None,                # disable energy gate
        enable_spatial=True,            # collect speech energy samples
    )

    import statistics
    if stats.energy_samples:
        ref_energy = statistics.mean(stats.energy_samples)
        print(f"\nEnergy samples: {len(stats.energy_samples)}")
        print(f"Energy mean: {ref_energy:.1f}")
    else:
        print("\nWARNING: No speech energy samples collected. Is the sound source active?")
        ref_energy = 0.0

    if stats.doa_samples:
        print(f"DOA mean: {statistics.mean(stats.doa_samples):.1f}°")

    r._save_calibration(
        args.cal_json,
        target_angle_deg=args.angle,
        target_distance_m=args.distance,
        ref_energy=ref_energy,
        ref_rms=stats.peak_rms,
    )
    print(f"\nReference energy: {ref_energy:.1f}  (peak RMS: {stats.peak_rms:.6f})")
    print("Calibration complete. Run 'python test.py record' to use it.")

    # Clean up temp file
    pathlib.Path("calibration_temp.wav").unlink(missing_ok=True)
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Record with spatial filtering enabled."""
    import respeaker_xvf3800_beam_record as r

    ref_energy = args.ref_energy
    if ref_energy is None and args.distance_gate and not args.no_spatial:
        try:
            cal = r._load_calibration(args.cal_json)
            ref_energy = cal.get("ref_energy")
            print(f"Calibration loaded: ref_energy={ref_energy:.6f}")
        except (FileNotFoundError, KeyError):
            print("No calibration file found – distance gate disabled.")
    elif not args.distance_gate:
        print("Distance gate disabled; using angle gate only.")

    stats = r.record(
        output_path=args.output,
        duration_sec=args.duration,
        rms_threshold=args.rms_threshold,
        device_hint=args.device_hint,
        beam_deg=args.angle,
        attack_ms=args.attack_ms,
        hold_ms=args.hold_ms,
        angle_tolerance_deg=args.angle_tolerance,
        ref_energy=ref_energy,
        energy_tolerance=args.energy_tolerance,
        ratio_threshold=args.ratio_threshold,
        enable_spatial=not args.no_spatial,
        trigger_on_voice=args.trigger_on_voice,
    )

    recv_sec = stats.received_frames / 16000
    saved_sec = stats.saved_frames / 16000
    ratio = (saved_sec / recv_sec * 100) if recv_sec > 0 else 0.0

    print(f"\n{'─' * 40}")
    print(f"  Captured : {recv_sec:.2f}s  Saved: {saved_sec:.2f}s ({ratio:.1f}%)")
    print(f"  Peak RMS : {stats.peak_rms:.4f}")
    if stats.speech_rejects or stats.doa_rejects or stats.energy_rejects or stats.ratio_rejects:
        print(f"  Speech rejects: {stats.speech_rejects}")
        print(f"  DOA rejects   : {stats.doa_rejects}")
        print(f"  Energy rejects: {stats.energy_rejects}")
        print(f"  Ratio rejects : {stats.ratio_rejects}")
    print(f"  Speech samples: {stats.speech_samples}")
    if stats.doa_samples:
        import statistics
        print(f"  DOA mean  : {statistics.mean(stats.doa_samples):.1f}°")
    print(f"{'─' * 40}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Analyze a WAV file RMS in 100ms blocks."""
    import analyze_wav_rms as an

    paths = an.expand_inputs(args.files)
    if not paths:
        print("ERROR: no input files found", file=sys.stderr)
        return 1

    label_regex = None
    if args.label_regex:
        label_regex = args.label_regex

    for path in paths:
        summary = an.analyze_file(path, args.threshold, args.block_ms)
        an.print_summary(summary, args.threshold, args.block_ms, label_regex)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="XVF3800 30° + 0.5m spatial recording test"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- calibrate ----------------------------------------------------------
    cal = sub.add_parser("calibrate", help="Calibrate reference energy at target position")
    cal.add_argument("--distance", type=float, default=0.5,
                     help="Target distance in meters (default: 0.5)")
    cal.add_argument("--angle", type=float, default=30.0,
                     help="Target angle in degrees (default: 30)")
    cal.add_argument("--duration", type=float, default=3.0,
                     help="Calibration duration in seconds (default: 3)")
    cal.add_argument("--device-hint", default=None)
    cal.add_argument("--cal-json", default="calibration.json")

    # -- record -------------------------------------------------------------
    rec = sub.add_parser("record", help="Record with spatial filtering")
    rec.add_argument("-o", "--output", default="test.wav")
    rec.add_argument("--duration", type=float, default=5.0)
    rec.add_argument("--rms-threshold", type=float, default=0.02)
    rec.add_argument("--device-hint", default=None)
    rec.add_argument("--angle", type=float, default=30.0,
                     help="Beam azimuth in degrees (default: 30)")
    rec.add_argument("--angle-tolerance", type=float, default=25.0,
                     help="Allowed DOA deviation ±degrees (default: 25)")
    rec.add_argument("--attack-ms", type=float, default=40.0)
    rec.add_argument("--hold-ms", type=float, default=400.0)
    rec.add_argument("--ref-energy", type=float, default=None,
                     help="Reference energy override")
    rec.add_argument("--energy-tolerance", type=float, default=0.5,
                     help="Allowed energy deviation fraction (default: 0.5)")
    rec.add_argument("--cal-json", default="calibration.json")
    rec.add_argument("--no-spatial", action="store_true",
                     help="Disable spatial filter")
    rec.add_argument("--distance-gate", action="store_true",
                     help="Enable distance gate by loading calibration.json")
    rec.add_argument("--trigger-on-voice", action="store_true", default=True,
                     help="Only create WAV after voice is detected")
    rec.add_argument("--ratio-threshold", type=float, default=0.0,
                     help="Optional front/back focus ratio E0/(E0+E1); 0 disables it (default: 0)")

    # -- analyze ------------------------------------------------------------
    ana = sub.add_parser("analyze", help="Analyze WAV file RMS blocks")
    ana.add_argument("files", nargs="+", help="WAV file(s) or glob")
    ana.add_argument("--threshold", type=float, default=0.05)
    ana.add_argument("--block-ms", type=int, default=100)
    ana.add_argument("--label-regex", default=None)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "calibrate":
        return cmd_calibrate(args)
    elif args.command == "record":
        return cmd_record(args)
    elif args.command == "analyze":
        return cmd_analyze(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
