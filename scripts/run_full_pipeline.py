#!/usr/bin/env python3
"""
End-to-end pipeline: ingestion → state segmentation → preprocessing → PSD.

This script demonstrates the full pipeline from raw EEG to Alpha-blockade PSD plots,
integrating the ingestion module (Task 1) with the preprocessing module.

Usage:
    python scripts/run_full_pipeline.py --subject sub-032301
    python scripts/run_full_pipeline.py --subject sub-032301 --save_plots
    python scripts/run_full_pipeline.py --subject_list sub-032301 sub-032302
"""
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_subject, segment_by_state, validate_raw
from src.preprocessing.pipeline import run_preprocessing
from src.utils.config import load_config, load_paths


def run_pipeline_for_state(
    raw_state, state_label, subject_id, config, output_dir,
):
    """Run preprocessing on a single state (EO or EC) and return results.

    Parameters
    ----------
    raw_state : mne.io.Raw
        Segmented raw data for one state (EO or EC).
    state_label : str
        Label for the state, e.g. "EO" or "EC".
    subject_id : str
        Subject identifier (e.g. "sub-032301").
    config : dict
        Preprocessing configuration loaded from YAML.
    output_dir : str or Path
        Root output directory. Preprocessed data saved to
        <output_dir>/preprocessed/<state_label>/.

    Returns
    -------
    dict
        Result from run_preprocessing(), containing "epochs" and "qc" keys.
    """
    print(f"\n  [{subject_id}] Preprocessing {state_label}...")
    result = run_preprocessing(
        raw_state,
        config,
        output_dir=str(Path(output_dir) / "preprocessed" / state_label),
        subject_id=f"{subject_id}_{state_label}",
    )
    print(f"  [{subject_id}] {state_label}: {result['qc']['n_windows']} windows")
    return result


def compute_psd_and_plot(epochs_eo, epochs_ec, subject_id, output_dir):
    """Compute PSD for EO and EC, save comparison plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        plot_dir = Path(output_dir) / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Compute PSD for each state
        psd_eo = epochs_eo.compute_psd(fmin=1, fmax=45, n_fft=512)
        psd_ec = epochs_ec.compute_psd(fmin=1, fmax=45, n_fft=512)

        # Average across epochs and channels
        psd_eo_avg = psd_eo.average()
        psd_ec_avg = psd_ec.average()

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # PSD overlay (all channels)
        ax = axes[0]
        freqs = psd_eo_avg.freqs
        ax.plot(freqs, psd_eo_avg.data.T, alpha=0.3, color="steelblue", linewidth=0.5)
        ax.plot(freqs, psd_ec_avg.data.T, alpha=0.3, color="coral", linewidth=0.5)
        ax.plot(freqs, np.mean(psd_eo_avg.data, axis=0), color="steelblue", linewidth=2.5, label="EO Mean")
        ax.plot(freqs, np.mean(psd_ec_avg.data, axis=0), color="coral", linewidth=2.5, label="EC Mean")
        ax.axvspan(8, 12, color="yellow", alpha=0.15, label="Alpha (8-12 Hz)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density (dB)")
        ax.set_title(f"{subject_id}: EO vs EC — All Channels")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Alpha band bar plot
        ax = axes[1]
        alpha_mask = (freqs >= 8) & (freqs <= 12)
        alpha_eo = np.mean(psd_eo_avg.data[:, alpha_mask], axis=1)
        alpha_ec = np.mean(psd_ec_avg.data[:, alpha_mask], axis=1)

        ch_names = epochs_eo.ch_names[:20]  # Show first 20 channels
        x = np.arange(len(ch_names))
        width = 0.35
        ax.bar(x - width/2, alpha_eo[:20], width, label="EO", color="steelblue", alpha=0.8)
        ax.bar(x + width/2, alpha_ec[:20], width, label="EC", color="coral", alpha=0.8)
        ax.set_xlabel("Channel")
        ax.set_ylabel("Alpha Power (dB)")
        ax.set_title(f"{subject_id}: Alpha Power (8-12 Hz) — First 20 Channels")
        ax.set_xticks(x)
        ax.set_xticklabels(ch_names, rotation=45, ha="right", fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plot_path = plot_dir / f"{subject_id}_alpha_blockade.png"
        plt.savefig(str(plot_path), dpi=150)
        plt.close(fig)
        print(f"  [PLOT] Saved Alpha-blockade plot -> {plot_path}")

        # Check for Alpha blockade effect
        mean_alpha_eo = np.mean(alpha_eo)
        mean_alpha_ec = np.mean(alpha_ec)
        blockade_ratio = mean_alpha_ec / mean_alpha_eo if mean_alpha_eo > 0 else float("inf")
        print(f"  [ALPHA] EO mean alpha: {mean_alpha_eo:.2f} dB")
        print(f"  [ALPHA] EC mean alpha: {mean_alpha_ec:.2f} dB")
        print(f"  [ALPHA] EC/EO ratio: {blockade_ratio:.2f}x")
        if blockade_ratio > 1.2:
            print(f"  [ALPHA] ✅ Alpha blockade detected! EC > EO by {((blockade_ratio-1)*100):.0f}%")
        else:
            print(f"  [ALPHA] ⚠️ No clear Alpha blockade (ratio={blockade_ratio:.2f})")

        return {
            "mean_alpha_eo": mean_alpha_eo,
            "mean_alpha_ec": mean_alpha_ec,
            "blockade_ratio": blockade_ratio,
        }

    except Exception as e:
        print(f"  [WARN] Failed to compute/plot PSD: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: ingestion → preprocessing → PSD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --subject sub-032301
  %(prog)s --subject sub-032301 --save_plots
  %(prog)s --subject_list sub-032301 sub-032302
        """,
    )
    parser.add_argument("--config", default="configs/preprocessing.yaml")
    parser.add_argument("--paths", default="configs/paths.local.yaml")
    parser.add_argument("--subject", help="Single subject ID")
    parser.add_argument("--subject_list", nargs="+", help="List of subject IDs")
    parser.add_argument("--session", default="rest")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--save_plots", action="store_true", help="Save Alpha-blockade PSD plots")

    args = parser.parse_args()

    # --- Resolve paths ---
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
    print("=" * 70)
    print("  End-to-End Pipeline: Ingestion → Preprocessing → PSD")
    print("=" * 70)
    print(f"  Subjects:     {len(subject_ids)}")
    print(f"  Lemon root:   {lemon_root}")
    print(f"  Output dir:   {output_dir}")
    print(f"  Save plots:   {'YES' if args.save_plots else 'NO'}")
    print("=" * 70)

    # --- Process each subject ---
    all_results = []
    errors = []

    for sid in subject_ids:
        print(f"\n{'=' * 70}")
        print(f"  Processing: {sid}")
        print(f"{'=' * 70}")

        try:
            # === Step 1: Ingestion (load + resample) ===
            print(f"\n[{sid}] Step 1: Loading and resampling...")
            raw = load_subject(
                sid,
                lemon_root=lemon_root,
                session=args.session,
                resample_freq=config.get("resample_freq", 250.0),
            )
            print(f"  [{sid}] Loaded: {raw.times[-1]:.1f}s @ {raw.info['sfreq']:.0f} Hz, "
                  f"{len(raw.ch_names)} channels")

            # === Step 2: State segmentation ===
            print(f"[{sid}] Step 2: Segmenting by state (EO/EC)...")
            raw_eo, raw_ec = segment_by_state(raw)
            print(f"  [{sid}] EO: {raw_eo.times[-1]:.1f}s, EC: {raw_ec.times[-1]:.1f}s")

            # === Step 3: Preprocessing (filter → ICA → windowing) ===
            print(f"[{sid}] Step 3: Preprocessing...")
            result_eo = run_pipeline_for_state(raw_eo, "EO", sid, config, output_dir)
            result_ec = run_pipeline_for_state(raw_ec, "EC", sid, config, output_dir)

            # === Step 4: PSD and Alpha-blockade analysis ===
            alpha_result = None
            if args.save_plots:
                print(f"[{sid}] Step 4: Computing PSD and Alpha-blockade analysis...")
                alpha_result = compute_psd_and_plot(
                    result_eo["epochs"], result_ec["epochs"], sid, output_dir,
                )

            all_results.append({
                "subject": sid,
                "status": "ok",
                "eo_windows": result_eo["qc"]["n_windows"],
                "ec_windows": result_ec["qc"]["n_windows"],
                "alpha": alpha_result,
            })

        except Exception as e:
            print(f"\n[FAIL] {sid}: {e}")
            errors.append({"subject": sid, "error": str(e)})

    # --- Final summary ---
    print()
    print("=" * 70)
    print("  Final Summary")
    print("=" * 70)
    print(f"  Total:     {len(subject_ids)}")
    print(f"  Succeeded: {len(all_results)}")
    print(f"  Failed:    {len(errors)}")
    for r in all_results:
        alpha_str = ""
        if r["alpha"]:
            ratio = r["alpha"]["blockade_ratio"]
            alpha_str = f" | Alpha EC/EO={ratio:.2f}x {'✅' if ratio > 1.2 else '⚠️'}"
        print(f"    ✅ {r['subject']}: {r['eo_windows']} EO windows, "
              f"{r['ec_windows']} EC windows{alpha_str}")
    if errors:
        print("  Errors:")
        for e in errors:
            print(f"    ❌ {e['subject']}: {e['error']}")
    print("=" * 70)

    return 0 if not errors else 1


if __name__ == "__main__":
    exit(main())
