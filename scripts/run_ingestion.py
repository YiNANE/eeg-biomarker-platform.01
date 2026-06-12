#!/usr/bin/env python3
"""
Ingestion pipeline: load, resample, segment by state, and save.

Usage:
    python scripts/run_ingestion.py --subject sub-032301
    python scripts/run_ingestion.py --subject sub-032301 --resample 250
    python scripts/run_ingestion.py --subject sub-032301 --output ./my_outputs
    python scripts/run_ingestion.py --subject sub-032301 --skip_validation
    python scripts/run_ingestion.py --subject sub-032301 --save_plots

Batch mode:
    python scripts/run_ingestion.py --subject_list sub-032301 sub-032302 sub-032303
    python scripts/run_ingestion.py --all  # process all available subjects
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_subject, segment_by_state, validate_raw, load_batch
from src.utils.config import load_config, load_paths


def save_segmented(raw_eo, raw_ec, subject_id, output_dir, session="rest"):
    """Save segmented EO/EC data as FIF files.

    Parameters
    ----------
    raw_eo : mne.io.Raw
        Eyes Open segmented data.
    raw_ec : mne.io.Raw
        Eyes Closed segmented data.
    subject_id : str
        Subject identifier (e.g. "sub-032301").
    output_dir : str or Path
        Root output directory. Files saved to <output_dir>/segmented/.
    session : str
        Session label (default "rest").

    Returns
    -------
    tuple of Path
        (eo_path, ec_path) — paths to saved FIF files.
    """
    seg_dir = Path(output_dir) / "segmented"
    seg_dir.mkdir(parents=True, exist_ok=True)

    eo_path = seg_dir / f"{subject_id}_ses-{session}_eo_raw.fif"
    ec_path = seg_dir / f"{subject_id}_ses-{session}_ec_raw.fif"

    raw_eo.save(str(eo_path), overwrite=True)
    raw_ec.save(str(ec_path), overwrite=True)

    return eo_path, ec_path


def print_summary(subject_id, raw, raw_eo, raw_ec, report):
    """Print a formatted summary of the ingestion results."""
    print()
    print("=" * 60)
    print(f"  Ingestion Summary — {subject_id}")
    print("=" * 60)
    print(f"  Original:  {raw.times[-1]:8.1f}s  @ {raw.info['sfreq']:.0f} Hz  "
          f"({raw.n_times} samples, {len(raw.ch_names)} ch)")
    print(f"  EO:        {raw_eo.times[-1]:8.1f}s  @ {raw_eo.info['sfreq']:.0f} Hz  "
          f"({raw_eo.n_times} samples)")
    print(f"  EC:        {raw_ec.times[-1]:8.1f}s  @ {raw_ec.info['sfreq']:.0f} Hz  "
          f"({raw_ec.n_times} samples)")
    print(f"  Total:     {raw_eo.times[-1] + raw_ec.times[-1]:8.1f}s  "
          f"(+ {raw.times[-1] - (raw_eo.times[-1] + raw_ec.times[-1]):.1f}s setup)")
    print(f"  Validation: {'PASSED' if report['passed'] else 'ISSUES FOUND'}")
    if not report['passed']:
        for issue in report['issues']:
            print(f"    - {issue}")
    print("=" * 60)


def process_single_subject(
    subject_id, lemon_root, config, output_dir, session="rest",
    resample_freq=250.0, skip_validation=False, save_plots=False,
):
    """Load, segment, validate, and save one subject."""
    # --- Step 1: Load with resampling ---
    print(f"\n[{subject_id}] Loading...")
    raw = load_subject(
        subject_id,
        lemon_root=lemon_root,
        session=session,
        resample_freq=resample_freq,
    )

    # --- Step 2: Segment by state ---
    print(f"[{subject_id}] Segmenting by state (EO/EC)...")
    raw_eo, raw_ec = segment_by_state(raw)

    # --- Step 3: Validate ---
    if not skip_validation:
        print(f"[{subject_id}] Validating...")
        report_eo = validate_raw(
            raw_eo, subject_id=f"{subject_id}_EO",
            expected_sfreq=resample_freq,
        )
        report_ec = validate_raw(
            raw_ec, subject_id=f"{subject_id}_EC",
            expected_sfreq=resample_freq,
        )
        # Combine reports
        report = {
            "passed": report_eo["passed"] and report_ec["passed"],
            "subject": subject_id,
            "checks": {"EO": report_eo["checks"], "EC": report_ec["checks"]},
            "issues": report_eo["issues"] + report_ec["issues"],
        }
    else:
        report = {"passed": True, "subject": subject_id, "checks": {}, "issues": []}

    # --- Step 4: Save ---
    eo_path, ec_path = save_segmented(raw_eo, raw_ec, subject_id, output_dir, session)
    print(f"[{subject_id}] Saved EO -> {eo_path}")
    print(f"[{subject_id}] Saved EC -> {ec_path}")

    # --- Step 5: Optional plots ---
    if save_plots:
        _save_plots(raw_eo, raw_ec, subject_id, output_dir)

    # --- Summary ---
    print_summary(subject_id, raw, raw_eo, raw_ec, report)

    return {
        "subject": subject_id,
        "status": "ok" if report["passed"] else "validation_issues",
        "raw": raw,
        "raw_eo": raw_eo,
        "raw_ec": raw_ec,
        "report": report,
        "eo_path": str(eo_path),
        "ec_path": str(ec_path),
    }


def _save_plots(raw_eo, raw_ec, subject_id, output_dir):
    """Save diagnostic plots for EO vs EC comparison."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plot_dir = Path(output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # PSD comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, raw_state, label in [
            (axes[0], raw_eo, "EO"),
            (axes[1], raw_ec, "EC"),
        ]:
            psd = raw_state.compute_psd(fmax=45)
            psd.plot(axes=ax, show=False)
            ax.set_title(f"{subject_id} — {label}")

        plt.tight_layout()
        plt.savefig(str(plot_dir / f"{subject_id}_psd_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  [PLOT] Saved PSD comparison -> {plot_dir / f'{subject_id}_psd_comparison.png'}")

        # Raw trace comparison (first 10s of each)
        fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        duration = min(10.0, raw_eo.times[-1], raw_ec.times[-1])
        for ax, raw_state, label in [
            (axes[0], raw_eo, "EO"),
            (axes[1], raw_ec, "EC"),
        ]:
            raw_state.copy().crop(tmax=duration).plot(
                n_channels=10, scalings="auto", show=False, axes=ax
            )
            ax.set_title(f"{subject_id} — {label} (first {duration:.0f}s)")

        plt.tight_layout()
        plt.savefig(str(plot_dir / f"{subject_id}_raw_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  [PLOT] Saved raw trace -> {plot_dir / f'{subject_id}_raw_comparison.png'}")

    except Exception as e:
        print(f"  [WARN] Failed to save plots: {e}")


def find_available_subjects(lemon_root):
    """Find all subject IDs available in the LEMON dataset."""
    root = Path(lemon_root)
    if not root.exists():
        return []
    return sorted([
        d.name for d in root.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Ingest and segment EEG data by state (EO/EC).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subject sub-032301
  %(prog)s --subject sub-032301 --resample 250 --save_plots
  %(prog)s --subject_list sub-032301 sub-032302 sub-032303
  %(prog)s --all
  %(prog)s --all --skip_validation
        """,
    )
    parser.add_argument(
        "--config",
        default="configs/preprocessing.yaml",
        help="Path to preprocessing config (default: configs/preprocessing.yaml)",
    )
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to local paths config (default: configs/paths.local.yaml)",
    )
    parser.add_argument(
        "--subject",
        help="Single subject ID, e.g. sub-032301",
    )
    parser.add_argument(
        "--subject_list",
        nargs="+",
        help="List of subject IDs, e.g. sub-032301 sub-032302",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all available subjects in lemon_root",
    )
    parser.add_argument(
        "--session",
        default="rest",
        help="Session label (default: rest)",
    )
    parser.add_argument(
        "--resample",
        type=float,
        default=250.0,
        help="Target sampling frequency in Hz (default: 250.0)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: from paths.local.yaml -> outputs_root)",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Skip data validation step",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save diagnostic PSD and raw trace plots",
    )

    args = parser.parse_args()

    # --- Resolve config paths relative to project root ---
    config_path = args.config
    if not Path(config_path).is_absolute():
        config_path = str(PROJECT_ROOT / config_path)
    paths_path = args.paths
    if not Path(paths_path).is_absolute():
        paths_path = str(PROJECT_ROOT / paths_path)

    config = load_config(config_path)
    paths = load_paths(paths_path)
    lemon_root = paths["lemon_root"]
    output_dir = args.output or paths.get("outputs_root", "outputs")

    # --- Determine subjects to process ---
    subject_ids = []
    if args.subject:
        subject_ids = [args.subject]
    elif args.subject_list:
        subject_ids = args.subject_list
    elif args.all:
        subject_ids = find_available_subjects(lemon_root)
        if not subject_ids:
            print(f"[ERROR] No subjects found in {lemon_root}")
            return 1
        print(f"Found {len(subject_ids)} subject(s) in {lemon_root}")
    else:
        parser.print_help()
        print("\n[ERROR] Specify --subject, --subject_list, or --all")
        return 1

    # --- Override resample_freq from config if not explicitly set ---
    resample_freq = args.resample
    if args.resample == 250.0 and "resample_freq" in config:
        resample_freq = config["resample_freq"]

    # --- Print header ---
    print("=" * 60)
    print("  EEG Ingestion Pipeline")
    print("=" * 60)
    print(f"  Subjects:     {len(subject_ids)}")
    print(f"  Lemon root:   {lemon_root}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Resample:     {resample_freq} Hz")
    print(f"  Session:      {args.session}")
    print(f"  Validation:   {'OFF' if args.skip_validation else 'ON'}")
    print(f"  Save plots:   {'YES' if args.save_plots else 'NO'}")
    print("=" * 60)

    # --- Process subjects ---
    results = []
    errors = []

    for sid in subject_ids:
        try:
            result = process_single_subject(
                sid,
                lemon_root=lemon_root,
                config=config,
                output_dir=output_dir,
                session=args.session,
                resample_freq=resample_freq,
                skip_validation=args.skip_validation,
                save_plots=args.save_plots,
            )
            results.append(result)
        except Exception as e:
            print(f"\n[FAIL] {sid}: {e}")
            errors.append({"subject": sid, "error": str(e)})

    # --- Final summary ---
    print()
    print("=" * 60)
    print("  Final Summary")
    print("=" * 60)
    print(f"  Total:     {len(subject_ids)}")
    print(f"  Succeeded: {len(results)}")
    print(f"  Failed:    {len(errors)}")
    if errors:
        print("  Errors:")
        for e in errors:
            print(f"    - {e['subject']}: {e['error']}")
    print("=" * 60)

    return 0 if not errors else 1


if __name__ == "__main__":
    exit(main())
