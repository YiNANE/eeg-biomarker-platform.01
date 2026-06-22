#!/usr/bin/env python3
"""
Run ICA artifact rejection on preprocessed EEG data.

This script applies ICA-based artifact removal to remove eye blink,
eye movement, and muscle artifacts from EEG data. It can work with
raw data directly or with EO/EC segmented data from the ingestion pipeline.

Usage:
    # Run on a single subject (uses full raw data):
    python scripts/run_ica.py --subject sub-032301

    # Run on already-segmented EO data:
    python scripts/run_ica.py --subject sub-032301 --state EO

    # Run with custom ICA config:
    python scripts/run_ica.py --subject sub-032301 --ica_config configs/ica.yaml

    # Run and save diagnostic plots:
    python scripts/run_ica.py --subject sub-032301 --save_plots

    # Batch mode:
    python scripts/run_ica.py --subject_list sub-032301 sub-032302 sub-032303
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_subject, segment_by_state
from src.preprocessing.ica import (
    run_ica_artifact_removal,
    compute_ica_qc,
    save_ica_report,
    classify_components,
    DEFAULT_ICA_CONFIG,
)
from src.utils.config import load_config, load_paths


def print_header(subject_id, state_label):
    """Print a formatted header for ICA processing."""
    print()
    print("=" * 60)
    print(f"  ICA Artifact Rejection — {subject_id}" +
          (f" ({state_label})" if state_label else ""))
    print("=" * 60)


def print_summary(subject_id, report, qc):
    """Print a formatted summary of ICA results."""
    print()
    print("=" * 60)
    print(f"  ICA Summary — {subject_id}")
    print("=" * 60)
    print(f"  ICA method:         {report.detection_method}")
    print(f"  Components total:   {report.n_components_total}")
    print(f"  Components removed: {report.n_components_removed}")
    print(f"  Removed indices:    {report.removed_indices}")
    print(f"  Removed labels:     {report.removed_labels}")
    print(f"  EOG channel used:   {report.eog_channel_used or 'none'}")
    print(f"  Convergence:        {'Yes' if report.converged else 'No'}")
    if report.converged:
        print(f"  Iterations:         {report.n_iter}")
    print(f"  Variance explained: {report.variance_explained_removed:.2%}")
    print(f"  Global variance:    {qc['global_variance_change_pct']:+.1f}%")
    print(f"  Frontal variance:   {qc['frontal_variance_change_pct']:+.1f}%")
    if report.warnings:
        print(f"  Warnings:           {len(report.warnings)}")
        for w in report.warnings:
            print(f"    ⚠ {w}")
    print("=" * 60)


def process_subject(
    subject_id,
    lemon_root,
    ica_config,
    output_dir,
    session="rest",
    state=None,
    resample_freq=250.0,
    save_plots=False,
):
    """
    Run ICA artifact rejection on a single subject.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    lemon_root : str
        Path to LEMON dataset root.
    ica_config : dict
        ICA configuration.
    output_dir : str
        Output directory for results.
    session : str
        Session label.
    state : str, optional
        If "EO" or "EC", run on segmented state data.
        If None, run on the full concatenated raw data.
    resample_freq : float
        Target sampling frequency.
    save_plots : bool
        Whether to save diagnostic plots.

    Returns
    -------
    dict with keys: subject_id, status, report, qc, raw_clean_path
    """
    # --- Load data ---
    if state:
        print(f"\n[{subject_id}] Loading and segmenting (state={state})...")
        raw_full = load_subject(
            subject_id,
            lemon_root=lemon_root,
            session=session,
            resample_freq=resample_freq,
        )
        raw_eo, raw_ec = segment_by_state(raw_full)
        if state.upper() == "EO":
            raw = raw_eo
        elif state.upper() == "EC":
            raw = raw_ec
        else:
            raise ValueError(f"Invalid state: {state}. Must be 'EO' or 'EC'.")
        state_label = state.upper()
    else:
        print(f"\n[{subject_id}] Loading raw data...")
        raw = load_subject(
            subject_id,
            lemon_root=lemon_root,
            session=session,
            resample_freq=resample_freq,
        )
        state_label = "full"

    print(
        f"[{subject_id}] Loaded: {raw.times[-1]:.1f}s "
        f"@ {raw.info['sfreq']:.0f} Hz, {len(raw.ch_names)} channels"
    )

    # --- Pre-filter for ICA (1 Hz high-pass strongly recommended) ---
    print(f"[{subject_id}] Applying pre-ICA bandpass filter (1–40 Hz)...")
    raw_filtered = raw.copy().filter(1.0, 40.0, fir_design="firwin")

    # --- Run ICA ---
    ica_config["output"]["save_plots"] = save_plots or ica_config.get(
        "output", {}
    ).get("save_plots", False)

    subject_label = f"{subject_id}_{state_label}" if state else subject_id
    raw_clean, ica, report = run_ica_artifact_removal(
        raw_filtered,
        config=ica_config,
        output_dir=output_dir,
        subject_id=subject_label,
    )

    # --- QC comparison ---
    qc = compute_ica_qc(
        raw_filtered, raw_clean, ica,
        removed_indices=report.removed_indices,
        subject_id=subject_label,
    )

    # --- Save cleaned data ---
    ica_dir = Path(output_dir) / "ica_cleaned"
    ica_dir.mkdir(parents=True, exist_ok=True)
    clean_path = ica_dir / f"{subject_label}_ica_clean_raw.fif"
    raw_clean.save(str(clean_path), overwrite=True)
    print(f"[{subject_id}] Saved cleaned data -> {clean_path}")

    # --- Save QC report ---
    if ica_config.get("output", {}).get("save_qc", True):
        save_ica_report(report, qc, output_dir, subject_label)

    # --- Component classification (for information) ---
    try:
        classification = classify_components(
            ica, raw_filtered, ica_config, subject_label
        )
        n_blink = sum(1 for v in classification.values() if v == "eye_blink")
        n_movement = sum(1 for v in classification.values() if v == "eye_movement")
        n_muscle = sum(1 for v in classification.values() if v == "muscle")
        n_neural = sum(1 for v in classification.values() if v == "neural")
        print(
            f"[{subject_id}] Classification: "
            f"{n_blink} blink, {n_movement} movement, "
            f"{n_muscle} muscle, {n_neural} neural"
        )
    except Exception as e:
        print(f"[{subject_id}] Classification skipped: {e}")

    return {
        "subject": subject_id,
        "state": state_label if state else "full",
        "status": "ok",
        "report": report,
        "qc": qc,
        "raw_clean_path": str(clean_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ICA artifact rejection for EEG data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subject sub-032301
  %(prog)s --subject sub-032301 --state EO --save_plots
  %(prog)s --subject_list sub-032301 sub-032302 sub-032303
  %(prog)s --subject sub-032301 --ica_config configs/ica.yaml
        """,
    )
    parser.add_argument(
        "--subject",
        help="Single subject ID, e.g. sub-032301",
    )
    parser.add_argument(
        "--subject_list",
        nargs="+",
        help="List of subject IDs",
    )
    parser.add_argument(
        "--state",
        choices=["EO", "EC"],
        help="Run ICA on EO or EC segmented data (default: full raw)",
    )
    parser.add_argument(
        "--ica_config",
        default="configs/ica.yaml",
        help="Path to ICA config YAML (default: configs/ica.yaml)",
    )
    parser.add_argument(
        "--paths",
        default="configs/paths.local.yaml",
        help="Path to local paths config",
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
        help="Output directory (default: from paths config)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Save diagnostic ICA plots",
    )
    parser.add_argument(
        "--method",
        choices=["correlation", "kurtosis", "topography", "combined"],
        default=None,
        help="Override EOG detection method from config",
    )

    args = parser.parse_args()

    # --- Resolve config paths ---
    ica_config_path = args.ica_config
    if not Path(ica_config_path).is_absolute():
        ica_config_path = str(PROJECT_ROOT / ica_config_path)
    paths_path = args.paths
    if not Path(paths_path).is_absolute():
        paths_path = str(PROJECT_ROOT / paths_path)

    # --- Load configs ---
    try:
        ica_config = load_config(ica_config_path)
    except FileNotFoundError:
        print(f"[WARN] ICA config not found: {ica_config_path}, using defaults")
        ica_config = DEFAULT_ICA_CONFIG.copy()

    # Handle nested "ica" key vs flat structure
    if "ica" in ica_config:
        ica_config = {**ica_config["ica"], **{k: v for k, v in ica_config.items() if k != "ica"}}

    paths = load_paths(paths_path)
    lemon_root = paths["lemon_root"]
    output_dir = args.output or paths.get("outputs_root", "outputs")

    # --- Override detection method if specified ---
    if args.method:
        ica_config.setdefault("eog_detection", {})
        ica_config["eog_detection"]["method"] = args.method

    # --- Determine subjects ---
    subject_ids = []
    if args.subject:
        subject_ids = [args.subject]
    elif args.subject_list:
        subject_ids = args.subject_list
    else:
        parser.print_help()
        print("\n[ERROR] Specify --subject or --subject_list")
        return 1

    # --- Print header ---
    print("=" * 60)
    print("  ICA Artifact Rejection Pipeline")
    print("=" * 60)
    print(f"  Subjects:       {len(subject_ids)}")
    print(f"  State:          {args.state or 'full raw'}")
    print(f"  ICA config:     {ica_config_path}")
    print(f"  Detection:      {ica_config.get('eog_detection', {}).get('method', 'combined')}")
    print(f"  Save plots:     {'YES' if args.save_plots else 'NO'}")
    print(f"  Output dir:     {output_dir}")
    print("=" * 60)

    # --- Process subjects ---
    results = []
    errors = []

    for sid in subject_ids:
        try:
            print_header(sid, args.state)
            result = process_subject(
                sid,
                lemon_root=lemon_root,
                ica_config=ica_config,
                output_dir=output_dir,
                session=args.session,
                state=args.state,
                resample_freq=args.resample,
                save_plots=args.save_plots,
            )
            print_summary(sid, result["report"], result["qc"])
            results.append(result)
        except Exception as e:
            print(f"\n[FAIL] {sid}: {e}")
            import traceback
            traceback.print_exc()
            errors.append({"subject": sid, "error": str(e)})

    # --- Final summary ---
    print()
    print("=" * 60)
    print("  ICA Pipeline — Final Summary")
    print("=" * 60)
    print(f"  Total:     {len(subject_ids)}")
    print(f"  Succeeded: {len(results)}")
    print(f"  Failed:    {len(errors)}")
    for r in results:
        qc = r["qc"]
        print(
            f"    ✅ {r['subject']} ({r['state']}): "
            f"removed {qc['n_components_removed']} components, "
            f"frontal var {qc['frontal_variance_change_pct']:+.1f}%"
        )
    if errors:
        print("  Errors:")
        for e in errors:
            print(f"    ❌ {e['subject']}: {e['error']}")
    print("=" * 60)

    return 0 if not errors else 1


if __name__ == "__main__":
    exit(main())
