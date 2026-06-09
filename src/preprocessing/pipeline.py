"""Preprocessing pipeline: filtering, bad channel detection, ICA, epoching/windowing."""
import mne
import numpy as np
from pathlib import Path
from ..utils.logging import get_logger

logger = get_logger(__name__)


def run_preprocessing(raw: mne.io.Raw, config: dict, output_dir: str, subject_id: str) -> dict:
    """
    Run full preprocessing pipeline on a raw EEG recording.

    Parameters
    ----------
    raw : mne.io.Raw
    config : dict
        Loaded from configs/preprocessing.yaml
    output_dir : str
        Where to save preprocessed output and QC.
    subject_id : str

    Returns
    -------
    dict with keys: 'raw_clean', 'epochs' or 'windows', 'qc'
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    qc = {"subject_id": subject_id}

    # --- Filtering ---
    logger.info(f"[{subject_id}] Filtering {config['l_freq']}-{config['h_freq']} Hz")
    raw.filter(config["l_freq"], config["h_freq"])
    raw.notch_filter(config["notch_freq"])

    # --- Resampling ---
    logger.info(f"[{subject_id}] Resampling to {config['resample_freq']} Hz")
    raw.resample(config["resample_freq"])

    # --- Bad channel detection ---
    logger.info(f"[{subject_id}] Detecting bad channels")
    # Placeholder: implement RANSAC or other method
    qc["n_bad_channels"] = len(raw.info["bads"])
    qc["bad_channels"] = raw.info["bads"]

    # --- ICA ---
    if config.get("run_ica", True):
        logger.info(f"[{subject_id}] Running ICA")
        ica = mne.preprocessing.ICA(
            n_components=config["ica_n_components"],
            method=config["ica_method"],
            random_state=42,
        )
        ica.fit(raw)

        # ---------- Automatic detection of eye movement / blink components ----------
        # 1. Determine the channel to use for EOG detection
        eog_channel = config.get("eog_channel")
        if eog_channel == "auto" or eog_channel is None:
            # Auto-select from common frontal electrodes
            possible = ['Fp1', 'Fp2', 'Fpz', 'EOG', 'eog', 'VEOG', 'HEOG']
            eog_channel = next((ch for ch in possible if ch in raw.ch_names), None)
            if eog_channel:
                logger.info(f"[{subject_id}] Auto-selected EOG channel: {eog_channel}")
            else:
                logger.warning(f"[{subject_id}] No suitable EOG channel found. ICA will not remove any components.")
        
        # 2. If a valid channel is found, detect and remove EOG components
        if eog_channel and eog_channel in raw.ch_names:
            # find_bads_eog returns (components_indices, scores)
            eog_indices, _ = ica.find_bads_eog(raw, ch_name=eog_channel, threshold=3.0)
            if len(eog_indices) == 0:
                logger.info(f"[{subject_id}] No EOG-related components detected.")
            else:
                logger.info(f"[{subject_id}] Detected EOG components: {eog_indices}")
            
            # Set the components to be excluded
            ica.exclude = eog_indices
            qc["ica_components_excluded"] = eog_indices
            
            # 3. Apply ICA to reconstruct the signal (remove excluded components from raw)
            ica.apply(raw)
            logger.info(f"[{subject_id}] ICA applied. Removed components {eog_indices} from the signal.")
        else:
            qc["ica_components_excluded"] = []
            logger.info(f"[{subject_id}] No valid EOG channel configured. Skipping component rejection.")

    # --- Windowing (resting-state) ---
    logger.info(f"[{subject_id}] Creating windows")
    window_len = config["window_length"]
    overlap = config["window_overlap"]
    step = window_len * (1 - overlap)
    events = mne.make_fixed_length_events(raw, duration=window_len, overlap=window_len * overlap)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=window_len, baseline=None, preload=True)
    qc["n_windows"] = len(epochs)

    # --- Save ---
    epochs.save(out / f"{subject_id}_preprocessed-epo.fif", overwrite=config.get("overwrite", False))
    logger.info(f"[{subject_id}] Done. {len(epochs)} windows saved.")

    return {"epochs": epochs, "qc": qc}
