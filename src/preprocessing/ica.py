"""
Physiological Artifact Rejection using Independent Component Analysis (ICA).

Detects and removes eye blink and eye movement artifacts from EEG data,
purifying frontal channels contaminated by ocular activity.

Usage:
    from src.preprocessing.ica import run_ica_artifact_removal
    raw_clean, ica, info = run_ica_artifact_removal(raw)
"""

import numpy as np
import mne
from typing import Optional, Tuple, List, Dict

from ..utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Default configuration
# ============================================================================

DEFAULT_ICA_CONFIG = {
    "method": "fastica",
    "n_components": 20,
    "random_state": 42,
    "max_iter": 5000,
    "fit_params": {},
    "eog_detection": {
        "method": "combined",      # "correlation", "kurtosis", "topography", "combined"
        "eog_channel": "auto",     # "auto", specific channel name, or null to skip correlation
        "correlation_threshold": 3.0,
        "kurtosis_threshold": 3.0,
        "focal_threshold": 0.75,
    },
}


# ============================================================================
# Public API
# ============================================================================


def run_ica_artifact_removal(
    raw: mne.io.Raw,
    config: Optional[dict] = None,
    subject_id: str = "unknown",
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA, dict]:
    """
    Run ICA artifact removal to purify frontal channels.

    Detects eye blink and eye movement components, removes them,
    and reconstructs clean EEG signal.

    Parameters
    ----------
    raw : mne.io.Raw
        Preloaded, filtered raw EEG data. High-pass filtering at 1 Hz
        is strongly recommended before calling this function.
    config : dict, optional
        ICA configuration. If None, uses DEFAULT_ICA_CONFIG.
    subject_id : str
        Subject identifier for logging.

    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned data with eye artifact components removed.
    ica : mne.preprocessing.ICA
        Fitted ICA object (excludes artifact components).
    info : dict
        Summary with keys: n_components_total, n_components_removed,
        removed_indices, removed_labels, eog_channel_used, detection_method.
    """
    cfg = _merge_config(config)
    _validate_raw_for_ica(raw, subject_id)

    n_components = _resolve_n_components(cfg["n_components"], raw, subject_id)
    n_total = n_components

    # --- Fit ICA ---
    logger.info(f"[{subject_id}] Fitting ICA ({cfg['method']}, {n_components} components)...")
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=cfg["method"],
        random_state=cfg.get("random_state", 42),
        max_iter=cfg.get("max_iter", 5000),
        fit_params=cfg.get("fit_params", {}),
    )
    ica.fit(raw)
    logger.info(f"[{subject_id}] ICA converged in {ica.n_iter_ or 0} iterations")

    # --- Detect EOG components ---
    eog_indices, eog_labels = _detect_eog_components(ica, raw, cfg, subject_id)
    eog_channel = _resolve_eog_channel(raw, cfg)

    # --- Safety: never remove ALL components ---
    if len(eog_indices) >= n_components:
        logger.warning(
            f"[{subject_id}] All {n_components} components classified as artifact. "
            f"Removing only the top 50%."
        )
        eog_indices = eog_indices[: max(1, n_components // 2)]

    # --- Remove artifacts and reconstruct ---
    ica.exclude = eog_indices
    logger.info(
        f"[{subject_id}] Excluding {len(eog_indices)}/{n_components} components: "
        f"{eog_indices}"
    )

    raw_clean = raw.copy()
    ica.apply(raw_clean)

    logger.info(
        f"[{subject_id}] ICA done. Removed {len(eog_indices)} components "
        f"({eog_labels}). Clean signal reconstructed."
    )

    info = {
        "n_components_total": n_total,
        "n_components_removed": len(eog_indices),
        "removed_indices": eog_indices,
        "removed_labels": eog_labels,
        "eog_channel_used": eog_channel,
        "detection_method": cfg["eog_detection"]["method"],
    }

    return raw_clean, ica, info


def configure_ica(
    raw: mne.io.Raw,
    config: Optional[dict] = None,
    subject_id: str = "unknown",
) -> mne.preprocessing.ICA:
    """
    Configure and return an unfitted ICA object.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data (used to determine n_components if auto).
    config : dict, optional
        ICA configuration.
    subject_id : str
        Subject identifier for logging.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Configured (unfitted) ICA object.
    """
    cfg = _merge_config(config)
    n_components = _resolve_n_components(cfg["n_components"], raw, subject_id)

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=cfg["method"],
        random_state=cfg.get("random_state", 42),
        max_iter=cfg.get("max_iter", 5000),
        fit_params=cfg.get("fit_params", {}),
    )
    logger.info(
        f"[{subject_id}] ICA configured: method={cfg['method']}, "
        f"n_components={n_components}"
    )
    return ica


# ============================================================================
# EOG Detection — Strategy 1: Correlation with EOG channel
# ============================================================================


def detect_eog_by_correlation(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    eog_channel: Optional[str] = None,
    threshold: float = 3.0,
    subject_id: str = "unknown",
) -> Tuple[List[int], Dict[int, str]]:
    """
    Detect EOG-related components by correlating component time courses
    with an EOG channel signal (MNE's find_bads_eog).

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        Raw EEG data (must contain the EOG channel).
    eog_channel : str, optional
        EOG channel name. Auto-detected if None.
    threshold : float
        Z-score threshold for EOG correlation detection.
    subject_id : str
        Subject identifier for logging.

    Returns
    -------
    eog_indices : list of int
        Component indices identified as EOG artifacts.
    labels : dict
        {component_index: "eye_blink" or "eye_movement"}.
    """
    if eog_channel is None:
        eog_channel = _resolve_eog_channel(raw, {})

    if eog_channel is None or eog_channel not in raw.ch_names:
        logger.warning(
            f"[{subject_id}] No EOG channel available for correlation detection."
        )
        return [], {}

    try:
        eog_indices, scores = ica.find_bads_eog(
            raw, ch_name=eog_channel, threshold=threshold
        )
    except Exception as e:
        logger.warning(f"[{subject_id}] EOG correlation detection failed: {e}")
        return [], {}

    labels = {}
    for i, idx in enumerate(eog_indices):
        labels[idx] = "eye_blink" if scores[i] > threshold * 2 else "eye_movement"

    logger.info(
        f"[{subject_id}] Correlation: {len(eog_indices)} EOG components "
        f"(ch={eog_channel}, thr={threshold})"
    )
    return list(eog_indices), labels


# ============================================================================
# EOG Detection — Strategy 2: Temporal Kurtosis
# ============================================================================


def detect_eog_by_kurtosis(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    threshold: float = 3.0,
    subject_id: str = "unknown",
) -> Tuple[List[int], Dict[int, str]]:
    """
    Detect eye-blink components by temporal kurtosis.

    Eye blinks produce transient high-amplitude deflections, so their
    component time courses have high kurtosis. This method works WITHOUT
    an EOG channel.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        Raw EEG data.
    threshold : float
        Kurtosis z-score threshold.
    subject_id : str
        Subject identifier for logging.

    Returns
    -------
    blink_indices : list of int
        Component indices identified as eye blinks.
    labels : dict
        {component_index: "eye_blink"}.
    """
    sources = ica.get_sources(raw).get_data()
    n_components = sources.shape[0]

    kurt_vals = np.array([_kurtosis(sources[i]) for i in range(n_components)])

    kurt_mean = np.nanmean(kurt_vals)
    kurt_std = np.nanstd(kurt_vals)
    if kurt_std < 1e-10:
        logger.warning(f"[{subject_id}] All components have near-identical kurtosis")
        return [], {}

    kurt_z = (kurt_vals - kurt_mean) / kurt_std
    blink_indices = np.where(kurt_z > threshold)[0].tolist()
    labels = {int(idx): "eye_blink" for idx in blink_indices}

    logger.info(
        f"[{subject_id}] Kurtosis: {len(blink_indices)} blink components "
        f"(thr={threshold})"
    )
    return blink_indices, labels


# ============================================================================
# EOG Detection — Strategy 3: Spatial Topography
# ============================================================================


def detect_eog_by_topography(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    frontal_channels: Optional[List[str]] = None,
    focal_threshold: float = 0.75,
    subject_id: str = "unknown",
) -> Tuple[List[int], Dict[int, str]]:
    """
    Detect eye-blink components by spatial topography.

    Eye artifacts have characteristic spatial patterns: highly focal
    (concentrated in few channels) and strong frontal weighting.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        Raw EEG data.
    frontal_channels : list of str, optional
        Frontal channel names. Auto-detected if None.
    focal_threshold : float
        Threshold for focal score (0–1). Higher = stricter.
    subject_id : str
        Subject identifier for logging.

    Returns
    -------
    blink_indices : list of int
        Component indices identified as probable eye artifacts.
    labels : dict
        {component_index: "eye_blink"}.
    """
    if frontal_channels is None:
        frontal_channels = _find_frontal_channels(raw, None)

    if not frontal_channels:
        logger.warning(
            f"[{subject_id}] No frontal channels identified for topography detection"
        )
        return [], {}

    ica_ch_names = ica.ch_names
    frontal_idx = [
        ica_ch_names.index(ch) for ch in frontal_channels if ch in ica_ch_names
    ]
    if not frontal_idx:
        logger.warning(f"[{subject_id}] No frontal channels found in data")
        return [], {}

    n_components = ica.n_components_
    topographies = _get_channel_topographies(ica)  # (n_channels, n_components)

    focal_scores = np.zeros(n_components)
    frontal_ratios = np.zeros(n_components)

    for i in range(n_components):
        topo = np.abs(topographies[:, i])
        topo_norm = topo / (np.sum(topo) + 1e-12)
        focal_scores[i] = float(np.max(topo_norm))
        frontal_ratios[i] = float(np.sum(topo_norm[frontal_idx]))

    blink_indices = np.where(
        (focal_scores > focal_threshold) & (frontal_ratios > 0.4)
    )[0].tolist()

    labels = {int(idx): "eye_blink" for idx in blink_indices}

    logger.info(
        f"[{subject_id}] Topography: {len(blink_indices)} blink components "
        f"(focal_thr={focal_threshold})"
    )
    return blink_indices, labels


# ============================================================================
# Internal helpers
# ============================================================================


def _detect_eog_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    config: dict,
    subject_id: str,
) -> Tuple[List[int], Dict[int, str]]:
    """
    Detect EOG components using the configured strategy.

    Supports:
      - "correlation": MNE's find_bads_eog (needs EOG channel)
      - "kurtosis": temporal kurtosis (no EOG channel needed)
      - "topography": spatial focality + frontal power
      - "combined": vote-based consensus of all three methods
    """
    eog_cfg = config.get("eog_detection", {})
    method = eog_cfg.get("method", "combined")
    eog_channel = _resolve_eog_channel(raw, config)
    corr_thresh = eog_cfg.get("correlation_threshold", 3.0)
    kurt_thresh = eog_cfg.get("kurtosis_threshold", 3.0)
    focal_thresh = eog_cfg.get("focal_threshold", 0.75)
    frontal_chs = _find_frontal_channels(raw, config)

    # --- Method 1: Correlation ---
    corr_indices, corr_labels = [], {}
    if method in ("correlation", "combined") and eog_channel:
        corr_indices, corr_labels = detect_eog_by_correlation(
            ica, raw, eog_channel, corr_thresh, subject_id
        )

    # --- Method 2: Kurtosis ---
    kurt_indices, kurt_labels = [], {}
    if method in ("kurtosis", "combined"):
        kurt_indices, kurt_labels = detect_eog_by_kurtosis(
            ica, raw, kurt_thresh, subject_id
        )

    # --- Method 3: Topography ---
    topo_indices, topo_labels = [], {}
    if method in ("topography", "combined"):
        topo_indices, topo_labels = detect_eog_by_topography(
            ica, raw, frontal_chs, focal_thresh, subject_id
        )

    # --- Combine results ---
    if method == "correlation":
        return corr_indices, corr_labels
    elif method == "kurtosis":
        return kurt_indices, kurt_labels
    elif method == "topography":
        return topo_indices, topo_labels
    else:  # "combined" — vote-based consensus
        from collections import Counter

        all_votes = corr_indices + kurt_indices + topo_indices
        vote_counts = Counter(all_votes)
        min_votes = 2 if (eog_channel and len(kurt_indices) >= 0) else 1
        all_indices = [
            idx for idx, count in vote_counts.items() if count >= min_votes
        ]
        all_labels = {}
        for idx in all_indices:
            if idx in corr_labels:
                all_labels[idx] = corr_labels[idx]
            elif idx in kurt_labels:
                all_labels[idx] = kurt_labels[idx]
            else:
                all_labels[idx] = topo_labels.get(idx, "eye_blink")

        return all_indices, all_labels


def _merge_config(user_config: Optional[dict]) -> dict:
    """Merge user config with defaults. User values take precedence."""
    if user_config is None:
        return DEFAULT_ICA_CONFIG.copy()

    merged = DEFAULT_ICA_CONFIG.copy()
    for key, value in user_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _validate_raw_for_ica(raw: mne.io.Raw, subject_id: str) -> None:
    """Validate that raw data is suitable for ICA."""
    if raw.get_data().size == 0:
        raise ValueError(f"[{subject_id}] Raw data is empty")

    if len(raw.ch_names) < 3:
        raise ValueError(
            f"[{subject_id}] ICA requires at least 3 channels, "
            f"got {len(raw.ch_names)}"
        )

    if not raw.preload:
        raise ValueError(
            f"[{subject_id}] Raw data must be preloaded before ICA. "
            f"Call raw.load_data() first."
        )

    data = raw.get_data()
    if np.any(np.isnan(data)):
        raise ValueError(f"[{subject_id}] Raw data contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError(f"[{subject_id}] Raw data contains Inf values")


def _resolve_n_components(
    n_components_cfg,
    raw: mne.io.Raw,
    subject_id: str,
) -> int:
    """Resolve n_components from config (None = auto, int = fixed)."""
    n_eeg_channels = sum(
        1 for ch in raw.info["chs"]
        if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH
    )
    if n_eeg_channels == 0:
        n_eeg_channels = len(raw.ch_names)

    if n_components_cfg is None:
        try:
            rank = mne.compute_rank(raw)
            n_comp = max(1, min(rank, n_eeg_channels - 1, 30))
            logger.info(f"[{subject_id}] Auto n_components: {n_comp} (rank={rank})")
            return n_comp
        except Exception:
            n_comp = min(n_eeg_channels, 20)
            logger.info(f"[{subject_id}] Fallback n_components: {n_comp}")
            return n_comp
    elif isinstance(n_components_cfg, int) and n_components_cfg > 0:
        return min(n_components_cfg, n_eeg_channels)
    else:
        raise ValueError(
            f"Invalid n_components: {n_components_cfg}. "
            f"Must be a positive integer or None (auto)."
        )


def _resolve_eog_channel(
    raw: mne.io.Raw,
    config: Optional[dict],
) -> Optional[str]:
    """Resolve EOG channel from config or auto-detect."""
    cfg = _merge_config(config)
    eog_ch = cfg.get("eog_detection", {}).get("eog_channel", "auto")

    if eog_ch is None or eog_ch == "":
        return None

    if eog_ch != "auto":
        if eog_ch in raw.ch_names:
            return eog_ch
        logger.warning(f"EOG channel '{eog_ch}' not found. Trying auto...")

    candidates = ["VEOG", "HEOG", "EOG", "Fp1", "Fp2", "Fpz", "eog", "veog", "heog"]
    for ch in candidates:
        if ch in raw.ch_names:
            return ch
    return None


def _find_frontal_channels(
    raw: mne.io.Raw,
    config: Optional[dict],
) -> List[str]:
    """Find frontal channels in the data."""
    cfg = _merge_config(config)
    configured = cfg.get("eog_detection", {}).get("frontal_channels", [])

    if configured:
        return [ch for ch in configured if ch in raw.ch_names]

    frontal_patterns = ["Fp", "AF", "F"]
    frontal = []
    for ch in raw.ch_names:
        for pattern in frontal_patterns:
            if ch.startswith(pattern):
                frontal.append(ch)
                break
    return frontal


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis of a 1D array. Normal distribution = 0."""
    n = len(x)
    if n < 4:
        return 0.0
    x = x - np.mean(x)
    m2 = np.mean(x ** 2)
    m4 = np.mean(x ** 4)
    if m2 < 1e-20:
        return 0.0
    return float(m4 / (m2 ** 2) - 3)


def _get_channel_topographies(ica: mne.preprocessing.ICA) -> np.ndarray:
    """
    Reconstruct ICA component topographies in channel space.

    Returns (n_channels, n_components) array.
    """
    pca = ica.pca_components_  # (n_pca, n_channels)
    unmixing = ica.get_components()  # (n_pca, n_components)
    return pca.T @ unmixing  # (n_channels, n_components)
