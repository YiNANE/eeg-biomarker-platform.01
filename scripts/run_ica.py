#!/usr/bin/env python3
"""
Run ICA artifact rejection on preprocessed EEG data.

Applies ICA-based artifact removal to remove eye blink and eye movement
artifacts from EEG data, purifying frontal channels.

Usage:
    python scripts/run_ica.py --subject sub-032301
    python scripts/run_ica.py --subject sub-032301 --state EO
    python scripts/run_ica.py --subject_list sub-032301 sub-032302
"""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_subject, segment_by_state
from src.preprocessing.ica import run_ica_artifact_removal
from src.utils.config import load_config, load_paths


def process_subject(
    subject_id,
    lemon_root,
    ica_config,
    output_dir,
    session="rest",
    state=None,
    resample_freq=250.0,
):
    """Load data, run ICA, and save cleaned result for one subject."""
    # --- Load ---
    if state:
        raw_full = load_subject(
            subject_id, lemon_root=lemon_root, session=session,
            resample_freq=resample_freq,
        )
        raw_eo, raw_ec = segment_by_state(raw_full)
        raw = raw_eo if state.upper() == "EO" else raw_ec
    else:
        raw = load_subject(
            subject_id, lemon_root=lemon_root, session=session,
            resample_freq=resample_freq,
        )

    print(f"[{subject_id}] Loaded: {raw.times[-1]:.1f}s "
          f"@ {raw.info['sfreq']:.0f} Hz, {len(raw.ch_names)} channels")

    # --- Pre-filter for ICA ---
    raw_filt = raw.copy().filter(1.0, 40.0, fir_design="firwin")

    # --- Run ICA ---
    state_label = f"_{state}" if state else ""
    label = f"{subject_id}{state_label}"
    raw_clean, ica, info = run_ica_artifact_removal(
        raw_filt, config=ica_config, subject_id=label,
    )

    # --- Save ---
    ica_dir = Path(output_dir) / "ica_cleaned"
    ica_dir.mkdir(parents=True, exist_ok=True)
    clean_path = ica_dir / f"{label}_ica_clean_raw.fif"
    raw_clean.save(str(clean_path), overwrite=True)

    # --- Summary ---
    print(f"[{subject_id}] Removed {info['n_components_removed']}/"
          f"{info['n_components_total']} components: {info['removed_indices']}")
    print(f"[{subject_id}] Labels: {info['removed_labels']}")
    print(f"[{subject_id}] Saved -> {clean_path}")

    return {
        "subject": subject_id,
        "state": state or "full",
        "status": "ok",
        "info": info,
        "clean_path": str(clean_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description="ICA artifact rejection for EEG data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subject sub-032301
  %(prog)s --subject sub-032301 --state EO
  %(prog)s --subject_list sub-032301 sub-032302
        """,
    )
    parser.add_argument("--subject", help="Single subject ID")
    parser.add_argument("--subject_list", nargs="+", help="List of subject IDs")
    parser.add_argument("--state", choices=["EO", "EC"],
                        help="Run on EO or EC segmented data (default: full raw)")
    parser.add_argument("--ica_config", default="configs/ica.yaml",
                        help="Path to ICA config YAML")
    parser.add_argument("--paths", default="configs/paths.local.yaml",
                        help="Path to local paths config")
    parser.add_argument("--session", default="rest", help="Session label")
    parser.add_argument("--resample", type=float, default=250.0,
                        help="Target sampling frequency in Hz")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: from paths config)")

    args = parser.parse_args()

    if not Path(args.ica_config).is_absolute():
        args.ica_config = str(PROJECT_ROOT / args.ica_config)
    if not Path(args.paths).is_absolute():
        args.paths = str(PROJECT_ROOT / args.paths)

    try:
        ica_config = load_config(args.ica_config)
    except FileNotFoundError:
        from src.preprocessing.ica import DEFAULT_ICA_CONFIG
        print(f"[WARN] Config not found: {args.ica_config}, using defaults")
        ica_config = DEFAULT_ICA_CONFIG.copy()

    if "ica" in ica_config:
        ica_config = {**ica_config["ica"],
                      **{k: v for k, v in ica_config.items() if k != "ica"}}

    paths = load_paths(args.paths)
    lemon_root = paths["lemon_root"]
    output_dir = args.output or paths.get("outputs_root", "outputs")

    subject_ids = []
    if args.subject:
        subject_ids = [args.subject]
    elif args.subject_list:
        subject_ids = args.subject_list
    else:
        parser.print_help()
        return 1

    print(f"Subjects: {len(subject_ids)} | State: {args.state or 'full'} | "
          f"Detection: {ica_config.get('eog_detection', {}).get('method', 'combined')}")

    results, errors = [], []
    for sid in subject_ids:
        try:
            results.append(process_subject(
                sid, lemon_root, ica_config, output_dir,
                session=args.session, state=args.state, resample_freq=args.resample,
            ))
        except Exception as e:
            print(f"[FAIL] {sid}: {e}")
            errors.append({"subject": sid, "error": str(e)})

    print(f"\nDone. {len(results)} succeeded, {len(errors)} failed.")
    for r in results:
        info = r["info"]
        print(f"  {r['subject']} ({r['state']}): "
              f"removed {info['n_components_removed']} components")
    if errors:
        for e in errors:
            print(f"  FAIL {e['subject']}: {e['error']}")

    return 0 if not errors else 1


if __name__ == "__main__":
    exit(main())
