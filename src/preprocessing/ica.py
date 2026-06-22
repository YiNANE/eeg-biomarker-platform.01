"""
Physiological Artifact Rejection using Independent Component Analysis (ICA).

This module provides comprehensive ICA-based artifact detection and removal for
EEG data, with special focus on eye blink and eye movement artifacts that
contaminate frontal channels.

Key capabilities:
  - Configure ICA with MNE (fastica, picard, infomax)
  - Detect EOG-related components via multiple strategies:
      1. Correlation with EOG channels (MNE's find_bads_eog)
      2. Temporal kurtosis (eye blinks produce high-kurtosis components)
      3. Topographic focal score (eye artifacts are focal in frontal regions)
      4. Combined multi-metric scoring
  - Detect muscle artifacts via spectral properties
  - Classify each component as: eye_blink, eye_movement, muscle, neural
  - Reconstruct clean EEG signal with artifacts removed
  - Generate diagnostic plots (component topographies, time courses, PSD)
  - Compute before/after QC metrics

Usage:
    from src.preprocessing.ica import run_ica_artifact_removal

    raw_clean, ica, qc = run_ica_artifact_removal(raw, config)
"""

import numpy as np
import mne
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field

from ..utils.logging import get_logger

logger = get_logger(__name__)


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class ComponentScores:
    """Per-component classification scores."""
    eog_correlation: np.ndarray     # correlation with EOG channel(s)
    kurtosis: np.ndarray            # temporal kurtosis (high = eye blink)
    focal_score: np.ndarray         # spatial focality (high = focal)
    frontal_power_ratio: np.ndarray # power ratio in frontal channels
    muscle_score: np.ndarray        # muscle artifact likelihood

    @property
    def n_components(self) -> int:
        return len(self.eog_correlation)


@dataclass
class ICAReport:
    """Complete ICA processing report."""
    subject_id: str
    n_components_total: int
    n_components_removed: int
    removed_indices: List[int]
    removed_labels: Dict[int, str]  # component index -> artifact type
    component_scores: Optional[ComponentScores] = None
    variance_explained_removed: float = 0.0
    eog_channel_used: Optional[str] = None
    detection_method: str = ""
    converged: bool = False
    n_iter: int = 0

    # Before/after comparison
    frontal_variance_before: float = 0.0
    frontal_variance_after: float = 0.0
    global_variance_before: float = 0.0
    global_variance_after: float = 0.0

    # Issues encountered
    warnings: List[str] = field(default_factory=list)


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
        "eog_channel": "auto",     # "auto", specific channel name, or null to skip
        "correlation_threshold": 3.0,
        "kurtosis_threshold": 3.0,
        "focal_threshold": 0.75,
    },
    "muscle_detection": {
        "enabled": True,
        "threshold": 0.6,
    },
    "classification": {
        "frontal_channels": ["Fp1", "Fp2", "Fpz", "AF3", "AF4", "AF7", "AF8"],
        "frontal_power_threshold": 0.5,
    },
    "output": {
        "save_plots": False,
        "overwrite": False,
    },
}


# ============================================================================
# Public API
# ============================================================================


def run_ica_artifact_removal(
    raw: mne.io.Raw,
    config: Optional[dict] = None,
    output_dir: Optional[str] = None,
    subject_id: str = "unknown",
) -> Tuple[mne.io.Raw, mne.preprocessing.ICA, ICAReport]:
    """
    Run full ICA artifact removal on raw EEG data.

    This is the main entry point. It:
      1. Configures and fits ICA
      2. Detects EOG/muscle artifact components
      3. Removes artifact components and reconstructs clean signal
      4. Computes QC metrics (before/after comparison)
      5. Optionally saves diagnostic plots

    Parameters
    ----------
    raw : mne.io.Raw
        Preloaded, filtered raw EEG data. Filtering should be applied BEFORE
        calling this function (e.g. 1–40 Hz bandpass).
    config : dict, optional
        ICA configuration. If None, uses DEFAULT_ICA_CONFIG.
        Expected structure matches configs/ica.yaml.
    output_dir : str, optional
        Directory for diagnostic plots and QC report. If None, plots are skipped.
    subject_id : str
        Subject identifier for logging and output files.

    Returns
    -------
    raw_clean : mne.io.Raw
        Cleaned raw data with artifact components removed.
    ica : mne.preprocessing.ICA
        Fitted ICA object (excludes artifact components).
    report : ICAReport
        Detailed report of components removed, scores, and QC metrics.

    Raises
    ------
    ValueError
        If raw has no data, too few channels, or ICA fails to converge.
    RuntimeError
        If ICA fitting fails catastrophically.

    Notes
    -----
    - The raw data MUST be preloaded (raw.load_data()) before calling.
    - High-pass filtering at 1 Hz is strongly recommended before ICA.
    - If no EOG channel is available, the module falls back to kurtosis +
      topography-based detection.
    """
    cfg = _merge_config(config)
    report = ICAReport(
        subject_id=subject_id,
        n_components_total=0,
        n_components_removed=0,
        removed_indices=[],
        removed_labels={},
    )

    # --- Validate input ---
    _validate_raw_for_ica(raw, subject_id)

    # --- Determine n_components ---
    n_components = _resolve_n_components(cfg["n_components"], raw, subject_id)
    report.n_components_total = n_components

    # --- Compute before metrics ---
    data_before = raw.get_data()
    report.global_variance_before = float(np.var(data_before))
    frontal_chs = _find_frontal_channels(raw, cfg)
    if frontal_chs:
        frontal_idx = [raw.ch_names.index(ch) for ch in frontal_chs]
        report.frontal_variance_before = float(np.var(data_before[frontal_idx, :]))

    # --- Fit ICA ---
    logger.info(f"[{subject_id}] Fitting ICA ({cfg['method']}, {n_components} components)...")
    try:
        ica = mne.preprocessing.ICA(
            n_components=n_components,
            method=cfg["method"],
            random_state=cfg.get("random_state", 42),
            max_iter=cfg.get("max_iter", 5000),
            fit_params=cfg.get("fit_params", {}),
        )
        ica.fit(raw)
        report.converged = True
        report.n_iter = ica.n_iter_ or 0
        logger.info(
            f"[{subject_id}] ICA converged in {report.n_iter} iterations"
        )
    except Exception as e:
        raise RuntimeError(
            f"[{subject_id}] ICA fitting failed: {e}. "
            f"Method: {cfg['method']}, n_components: {n_components}"
        ) from e

    # --- Detect EOG components ---
    eog_indices, eog_labels, scores = _detect_eog_components(ica, raw, cfg, subject_id)
    report.eog_channel_used = _resolve_eog_channel(raw, cfg)
    report.detection_method = cfg["eog_detection"]["method"]
    report.component_scores = scores

    # --- Detect muscle components ---
    muscle_indices = []
    if cfg.get("muscle_detection", {}).get("enabled", False):
        muscle_indices = _detect_muscle_components(
            ica, raw, cfg, subject_id
        )

    # --- Combine and deduplicate ---
    all_removed = sorted(set(eog_indices + muscle_indices))
    for idx in eog_indices:
        report.removed_labels[idx] = eog_labels.get(idx, "eog_artifact")
    for idx in muscle_indices:
        if idx not in report.removed_labels:
            report.removed_labels[idx] = "muscle_artifact"
    report.removed_indices = all_removed
    report.n_components_removed = len(all_removed)

    # --- Safety check: don't remove ALL components ---
    if len(all_removed) >= n_components:
        msg = (
            f"[{subject_id}] All {n_components} components classified as artifact. "
            f"This suggests a problem with the data or detection thresholds. "
            f"Removing only the top 50% of components. "
            f"Please inspect the data manually."
        )
        logger.warning(msg)
        report.warnings.append(msg)
        # Keep only the strongest half
        all_removed = sorted(
            all_removed,
            key=lambda i: scores.eog_correlation[i] if scores else 0,
            reverse=True,
        )[: max(1, n_components // 2)]
        report.removed_indices = all_removed
        report.n_components_removed = len(all_removed)
        report.removed_labels = {
            i: report.removed_labels.get(i, "eog_artifact")
            for i in all_removed
        }

    # --- Remove artifacts ---
    ica.exclude = all_removed
    logger.info(
        f"[{subject_id}] Excluding {len(all_removed)}/{n_components} components: "
        f"{all_removed}"
    )
    for idx in all_removed:
        logger.info(
            f"  Component {idx}: {report.removed_labels.get(idx, 'unknown')}"
        )

    # Apply ICA — reconstruct clean signal
    raw_clean = raw.copy()
    ica.apply(raw_clean)

    # --- Compute after metrics ---
    data_after = raw_clean.get_data()
    report.global_variance_after = float(np.var(data_after))
    if frontal_chs:
        report.frontal_variance_after = float(np.var(data_after[frontal_idx, :]))
    report.variance_explained_removed = float(
        np.sum(ica.pca_explained_variance_[all_removed])
        / np.sum(ica.pca_explained_variance_)
    ) if len(all_removed) > 0 and hasattr(ica, 'pca_explained_variance_') else 0.0

    # --- Diagnostic plots ---
    if output_dir and cfg.get("output", {}).get("save_plots", False):
        try:
            _save_diagnostic_plots(ica, raw, raw_clean, all_removed, output_dir, subject_id)
        except Exception as e:
            logger.warning(f"[{subject_id}] Failed to save diagnostic plots: {e}")

    # --- Log summary ---
    _log_summary(report, subject_id)

    return raw_clean, ica, report


# ============================================================================
# ICA Configuration
# ============================================================================


def configure_ica(
    raw: mne.io.Raw,
    config: Optional[dict] = None,
    subject_id: str = "unknown",
) -> mne.preprocessing.ICA:
    """
    Configure and return an ICA object (without fitting).

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
# EOG Component Detection — Strategy #1: Correlation
# ============================================================================


def detect_eog_by_correlation(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    eog_channel: Optional[str] = None,
    threshold: float = 3.0,
    subject_id: str = "unknown",
) -> Tuple[List[int], Dict[int, str]]:
    """
    Detect EOG-related components by correlating with EOG channel(s).

    Uses MNE's `find_bads_eog` which correlates each component's time course
    with the EOG channel signal. Components exceeding the z-score threshold
    are flagged as EOG artifacts.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        Raw EEG data (must have the EOG channel).
    eog_channel : str, optional
        EOG channel name. If None, auto-detected.
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
            f"[{subject_id}] No EOG channel available for correlation detection. "
            f"Available: {raw.ch_names[:10]}..."
        )
        return [], {}

    try:
        eog_indices, scores = ica.find_bads_eog(
            raw, ch_name=eog_channel, threshold=threshold
        )
    except Exception as e:
        logger.warning(
            f"[{subject_id}] EOG correlation detection failed: {e}"
        )
        return [], {}

    # Label components based on score magnitude
    labels = {}
    for i, idx in enumerate(eog_indices):
        if scores[i] > threshold * 2:
            labels[idx] = "eye_blink"  # strong correlation = blink
        else:
            labels[idx] = "eye_movement"  # moderate correlation = movement

    logger.info(
        f"[{subject_id}] Correlation detection: {len(eog_indices)} EOG components "
        f"found (ch={eog_channel}, threshold={threshold})"
    )
    return list(eog_indices), labels


# ============================================================================
# EOG Component Detection — Strategy #2: Kurtosis
# ============================================================================


def detect_eog_by_kurtosis(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    threshold: float = 3.0,
    subject_id: str = "unknown",
) -> Tuple[List[int], Dict[int, str], np.ndarray]:
    """
    Detect eye-blink components by temporal kurtosis.

    Eye blinks produce transient, high-amplitude deflections → component time
    courses have high kurtosis (peaked distribution with heavy tails).
    This method works WITHOUT an EOG channel.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        Raw EEG data.
    threshold : float
        Kurtosis z-score threshold (default 3.0).
    subject_id : str
        Subject identifier for logging.

    Returns
    -------
    blink_indices : list of int
        Component indices identified as eye blinks.
    labels : dict
        {component_index: "eye_blink"}.
    kurtosis_values : np.ndarray
        Kurtosis value for each component.
    """
    sources = ica.get_sources(raw).get_data()  # (n_components, n_times)
    n_components = sources.shape[0]

    # Compute kurtosis for each component
    # Use excess kurtosis (normal = 0)
    kurt_vals = np.zeros(n_components)
    for i in range(n_components):
        k = _kurtosis(sources[i, :])
        kurt_vals[i] = k

    # Z-score the kurtosis values
    kurt_mean = np.nanmean(kurt_vals)
    kurt_std = np.nanstd(kurt_vals)
    if kurt_std < 1e-10:
        logger.warning(f"[{subject_id}] All components have near-identical kurtosis")
        return [], {}, kurt_vals

    kurt_z = (kurt_vals - kurt_mean) / kurt_std

    blink_indices = np.where(kurt_z > threshold)[0].tolist()

    labels = {int(idx): "eye_blink" for idx in blink_indices}

    logger.info(
        f"[{subject_id}] Kurtosis detection: {len(blink_indices)} blink components "
        f"(threshold={threshold})"
    )
    return blink_indices, labels, kurt_vals


# ============================================================================
# EOG Component Detection — Strategy #3: Topography
# ============================================================================


def detect_eog_by_topography(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    frontal_channels: Optional[List[str]] = None,
    focal_threshold: float = 0.75,
    subject_id: str = "unknown",
) -> Tuple[List[int], Dict[int, str], np.ndarray, np.ndarray]:
    """
    Detect eye-blink components by topographic properties.

    Eye artifacts have characteristic spatial patterns:
      - Highly focal (concentrated in few frontal channels)
      - Strong frontal weighting (eyes are in front of the head)
      - Bilateral symmetry (both eyes blink together)

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        Raw EEG data.
    frontal_channels : list of str, optional
        List of frontal channel names. Auto-detected if None.
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
    focal_scores : np.ndarray
        Focal score for each component (0–1, higher = more focal).
    frontal_ratios : np.ndarray
        Frontal power ratio for each component (0–1).
    """
    if frontal_channels is None:
        frontal_channels = _find_frontal_channels(raw, None)

    if not frontal_channels:
        logger.warning(
            f"[{subject_id}] No frontal channels identified for topography detection"
        )
        return [], {}, np.array([]), np.array([])

    n_components = ica.n_components_
    focal_scores = np.zeros(n_components)
    frontal_ratios = np.zeros(n_components)

    # Get channel positions — use ICA's channel list (excludes non-EEG like EOG)
    ica_ch_names = ica.ch_names
    frontal_idx = [ica_ch_names.index(ch) for ch in frontal_channels if ch in ica_ch_names]

    if not frontal_idx:
        logger.warning(f"[{subject_id}] No frontal channels found in data")
        return [], {}, np.array([]), np.array([])

    # Get channel-space topographies: (n_channels, n_components)
    topographies = _get_channel_topographies(ica)  # (n_channels, n_components)

    for i in range(n_components):
        # Topography of component i
        topo = np.abs(topographies[:, i])
        topo_norm = topo / (np.sum(topo) + 1e-12)

        # Focal score: ratio of power in the strongest channel vs total
        focal_scores[i] = float(np.max(topo_norm))

        # Frontal ratio: power in frontal channels / total
        frontal_ratios[i] = float(np.sum(topo_norm[frontal_idx]))

    # Combine scores: components that are BOTH focal AND frontal
    combined = focal_scores * frontal_ratios
    threshold_val = np.percentile(combined, 85) if len(combined) > 5 else 0.5

    blink_indices = np.where(
        (focal_scores > focal_threshold) & (frontal_ratios > 0.4)
    )[0].tolist()

    labels = {int(idx): "eye_blink" for idx in blink_indices}

    logger.info(
        f"[{subject_id}] Topography detection: {len(blink_indices)} blink components "
        f"(focal_threshold={focal_threshold})"
    )
    return blink_indices, labels, focal_scores, frontal_ratios


# ============================================================================
# Muscle Artifact Detection
# ============================================================================


def _detect_muscle_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    config: dict,
    subject_id: str,
) -> List[int]:
    """
    Detect muscle artifact components by spectral properties.

    Muscle artifacts have broadband high-frequency power (>20 Hz).
    Components with unusually high spectral power above 20 Hz are flagged.
    """
    muscle_cfg = config.get("muscle_detection", {})
    threshold = muscle_cfg.get("threshold", 0.6)

    sources = ica.get_sources(raw).get_data()
    sfreq = raw.info["sfreq"]
    n_components = sources.shape[0]

    muscle_scores = np.zeros(n_components)

    for i in range(n_components):
        # Compute PSD
        psd = np.abs(np.fft.rfft(sources[i, :])) ** 2
        freqs = np.fft.rfftfreq(len(sources[i, :]), d=1.0 / sfreq)

        # High-frequency power ratio (>20 Hz vs total)
        hf_mask = freqs > 20
        if np.any(hf_mask):
            muscle_scores[i] = float(
                np.sum(psd[hf_mask]) / (np.sum(psd) + 1e-12)
            )

    # Flag components exceeding threshold
    muscle_indices = np.where(muscle_scores > threshold)[0].tolist()

    logger.info(
        f"[{subject_id}] Muscle detection: {len(muscle_indices)} components "
        f"(threshold={threshold})"
    )
    return muscle_indices


# ============================================================================
# Multi-method combined detection
# ============================================================================


def _detect_eog_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    config: dict,
    subject_id: str,
) -> Tuple[List[int], Dict[int, str], ComponentScores]:
    """
    Detect EOG components using the configured strategy.

    Supports:
      - "correlation": MNE's find_bads_eog (needs EOG channel)
      - "kurtosis": temporal kurtosis (no EOG channel needed)
      - "topography": spatial focality + frontal power
      - "combined": weighted vote across all methods
    """
    eog_cfg = config.get("eog_detection", {})
    method = eog_cfg.get("method", "combined")
    eog_channel = _resolve_eog_channel(raw, config)
    corr_thresh = eog_cfg.get("correlation_threshold", 3.0)
    kurt_thresh = eog_cfg.get("kurtosis_threshold", 3.0)
    focal_thresh = eog_cfg.get("focal_threshold", 0.75)
    frontal_chs = _find_frontal_channels(raw, config)

    n_components = ica.n_components_

    # Initialize scores
    eog_corr_scores = np.zeros(n_components)
    kurt_vals = np.zeros(n_components)
    focal_scores = np.zeros(n_components)
    frontal_ratios = np.zeros(n_components)
    muscle_scores = np.zeros(n_components)

    # --- Method 1: Correlation ---
    corr_indices, corr_labels = [], {}
    if method in ("correlation", "combined") and eog_channel:
        corr_indices, corr_labels = detect_eog_by_correlation(
            ica, raw, eog_channel, corr_thresh, subject_id
        )
        # Get individual scores from ICA object
        try:
            _, corr_scores_raw = ica.find_bads_eog(
                raw, ch_name=eog_channel, threshold=0  # get all scores
            )
            eog_corr_scores = np.array(corr_scores_raw)
        except Exception:
            eog_corr_scores = np.zeros(n_components)

    # --- Method 2: Kurtosis ---
    kurt_indices, kurt_labels = [], {}
    if method in ("kurtosis", "combined"):
        kurt_indices, kurt_labels, kurt_vals = detect_eog_by_kurtosis(
            ica, raw, kurt_thresh, subject_id
        )

    # --- Method 3: Topography ---
    topo_indices, topo_labels = [], {}
    if method in ("topography", "combined"):
        topo_indices, topo_labels, focal_scores, frontal_ratios = (
            detect_eog_by_topography(
                ica, raw, frontal_chs, focal_thresh, subject_id
            )
        )

    # --- Combine results ---
    if method == "correlation":
        all_indices = corr_indices
        all_labels = corr_labels
    elif method == "kurtosis":
        all_indices = kurt_indices
        all_labels = kurt_labels
    elif method == "topography":
        all_indices = topo_indices
        all_labels = topo_labels
    else:  # "combined" — vote-based consensus
        # Each method votes; component is artifact if ≥ 2 methods agree
        from collections import Counter

        all_votes = corr_indices + kurt_indices + topo_indices
        vote_counts = Counter(all_votes)
        # Need at least 2 votes, or 1 vote if only 1 method is applicable
        min_votes = 2 if (eog_channel and len(kurt_vals) > 0) else 1
        all_indices = [
            idx for idx, count in vote_counts.items() if count >= min_votes
        ]
        all_labels = {}
        for idx in all_indices:
            # Prefer correlation label over others
            if idx in corr_labels:
                all_labels[idx] = corr_labels[idx]
            elif idx in kurt_labels:
                all_labels[idx] = kurt_labels[idx]
            else:
                all_labels[idx] = topo_labels.get(idx, "eye_blink")

    # --- Compile scores ---
    # Normalize kurtosis to z-scores for ComponentScores
    if np.nanstd(kurt_vals) > 0:
        kurt_z = (kurt_vals - np.nanmean(kurt_vals)) / np.nanstd(kurt_vals)
    else:
        kurt_z = np.zeros_like(kurt_vals)

    scores = ComponentScores(
        eog_correlation=eog_corr_scores,
        kurtosis=kurt_z,
        focal_score=focal_scores,
        frontal_power_ratio=frontal_ratios,
        muscle_score=muscle_scores,
    )

    return all_indices, all_labels, scores


# ============================================================================
# Component Classification
# ============================================================================


def classify_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    config: Optional[dict] = None,
    subject_id: str = "unknown",
) -> Dict[int, str]:
    """
    Classify all ICA components into artifact types.

    Each component is classified as one of:
      - "eye_blink": Stereotyped blink artifact (high kurtosis + frontal)
      - "eye_movement": Sustained eye movement (moderate EOG correlation)
      - "muscle": High-frequency muscle artifact
      - "neural": Brain activity (none of the above)

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.
    raw : mne.io.Raw
        Raw EEG data.
    config : dict, optional
        ICA configuration.
    subject_id : str
        Subject identifier for logging.

    Returns
    -------
    classification : dict
        {component_index: artifact_type}
    """
    cfg = _merge_config(config)
    eog_cfg = cfg.get("eog_detection", {})
    muscle_cfg = cfg.get("muscle_detection", {})
    kurt_thresh = eog_cfg.get("kurtosis_threshold", 2.0)  # lower threshold for classification
    focal_thresh = eog_cfg.get("focal_threshold", 0.6)
    muscle_thresh = muscle_cfg.get("threshold", 0.5)

    n_components = ica.n_components_
    sources = ica.get_sources(raw).get_data()

    classification = {}
    eog_channel = _resolve_eog_channel(raw, cfg)

    # Compute per-component metrics
    # Pre-compute channel topographies and frontal info
    topographies = _get_channel_topographies(ica)  # (n_channels, n_components)
    frontal_chs = _find_frontal_channels(raw, cfg)
    ica_ch_names = ica.ch_names
    frontal_idx = []
    if frontal_chs:
        frontal_idx = [
            ica_ch_names.index(ch) for ch in frontal_chs if ch in ica_ch_names
        ]

    for i in range(n_components):
        source = sources[i, :]

        # Kurtosis
        k = _kurtosis(source)

        # Focal score from topography
        topo = np.abs(topographies[:, i])
        topo_norm = topo / (np.sum(topo) + 1e-12)
        focal = float(np.max(topo_norm))

        # Frontal ratio
        if frontal_idx:
            frontal_ratio = float(np.sum(topo_norm[frontal_idx]))
        else:
            frontal_ratio = 0.0

        # Muscle score
        sfreq = raw.info["sfreq"]
        psd = np.abs(np.fft.rfft(source)) ** 2
        freqs = np.fft.rfftfreq(len(source), d=1.0 / sfreq)
        hf_mask = freqs > 20
        muscle = float(np.sum(psd[hf_mask]) / (np.sum(psd) + 1e-12))

        # Classification logic
        if k > kurt_thresh and (focal > focal_thresh or frontal_ratio > 0.3):
            classification[i] = "eye_blink"
        elif muscle > muscle_thresh:
            classification[i] = "muscle"
        elif frontal_ratio > 0.3 and focal > focal_thresh:
            classification[i] = "eye_movement"
        else:
            classification[i] = "neural"

    # Log summary
    type_counts = {}
    for v in classification.values():
        type_counts[v] = type_counts.get(v, 0) + 1
    logger.info(
        f"[{subject_id}] Component classification: {type_counts}"
    )

    return classification


# ============================================================================
# Before/After QC comparison
# ============================================================================


def compute_ica_qc(
    raw_before: mne.io.Raw,
    raw_after: mne.io.Raw,
    ica: mne.preprocessing.ICA,
    removed_indices: List[int],
    subject_id: str = "unknown",
    frontal_channels: Optional[List[str]] = None,
) -> dict:
    """
    Compute before/after QC metrics for ICA artifact removal.

    Metrics computed:
      - Global variance change (%)
      - Frontal channel variance change (%)
      - Variance explained by removed components
      - Per-channel variance ratio (after/before)

    Parameters
    ----------
    raw_before : mne.io.Raw
        Raw data before ICA cleaning.
    raw_after : mne.io.Raw
        Raw data after ICA cleaning.
    ica : mne.preprocessing.ICA
        Fitted ICA object with exclude set.
    removed_indices : list of int
        Indices of removed components.
    subject_id : str
        Subject identifier.
    frontal_channels : list of str, optional
        Frontal channel names. Auto-detected if None.

    Returns
    -------
    qc : dict
        Dictionary of QC metrics.
    """
    if frontal_channels is None:
        frontal_channels = _find_frontal_channels(raw_before, None)

    data_before = raw_before.get_data()
    data_after = raw_after.get_data()
    ch_names = raw_before.ch_names

    # Global variance
    global_var_before = float(np.var(data_before))
    global_var_after = float(np.var(data_after))
    global_var_change_pct = (
        (global_var_after - global_var_before) / (global_var_before + 1e-12) * 100
    )

    # Frontal variance
    frontal_var_before = 0.0
    frontal_var_after = 0.0
    frontal_var_change_pct = 0.0
    if frontal_channels:
        frontal_idx = [ch_names.index(ch) for ch in frontal_channels if ch in ch_names]
        if frontal_idx:
            frontal_var_before = float(np.var(data_before[frontal_idx, :]))
            frontal_var_after = float(np.var(data_after[frontal_idx, :]))
            frontal_var_change_pct = (
                (frontal_var_after - frontal_var_before)
                / (frontal_var_before + 1e-12) * 100
            )

    # Per-channel variance ratio
    ch_var_before = np.var(data_before, axis=1)
    ch_var_after = np.var(data_after, axis=1)
    ch_var_ratio = ch_var_after / (ch_var_before + 1e-12)

    # Variance explained by removed components
    if hasattr(ica, "pca_explained_variance_"):
        total_var = np.sum(ica.pca_explained_variance_)
        removed_var = (
            np.sum(ica.pca_explained_variance_[removed_indices])
            if removed_indices
            else 0.0
        )
        var_explained = float(removed_var / total_var) if total_var > 0 else 0.0
    else:
        var_explained = 0.0

    qc = {
        "subject_id": subject_id,
        "n_components_removed": len(removed_indices),
        "removed_indices": removed_indices,
        "global_variance_before": global_var_before,
        "global_variance_after": global_var_after,
        "global_variance_change_pct": float(global_var_change_pct),
        "frontal_variance_before": frontal_var_before,
        "frontal_variance_after": frontal_var_after,
        "frontal_variance_change_pct": float(frontal_var_change_pct),
        "variance_explained_by_removed": var_explained,
        "max_channel_variance_increase": float(np.max(ch_var_ratio)),
        "channels_with_variance_decrease": int(np.sum(ch_var_ratio < 1.0)),
        "channels_with_variance_increase": int(np.sum(ch_var_ratio > 1.0)),
    }

    logger.info(
        f"[{subject_id}] ICA QC: "
        f"global variance {global_var_change_pct:+.1f}%, "
        f"frontal variance {frontal_var_change_pct:+.1f}%, "
        f"{var_explained:.1%} variance removed"
    )

    return qc


# ============================================================================
# Diagnostic visualization
# ============================================================================


def _save_diagnostic_plots(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
    raw_clean: mne.io.Raw,
    removed_indices: List[int],
    output_dir: str,
    subject_id: str,
) -> None:
    """Generate and save diagnostic plots for ICA artifact removal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(output_dir) / "ica_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    n_components = ica.n_components_

    # ---- Plot 1: Component topographies ----
    n_cols = min(8, n_components)
    n_rows = int(np.ceil(n_components / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.5, n_rows * 2.5),
        squeeze=False,
    )
    for i in range(n_components):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        try:
            ica.plot_components(
                picks=i, axes=ax, show=False,
                title=f"IC{i}" + (" [REMOVED]" if i in removed_indices else ""),
                colorbar=False,
            )
        except Exception:
            ax.text(0.5, 0.5, f"IC{i}", ha="center", va="center")
            ax.set_axis_off()

    # Hide unused subplots
    for i in range(n_components, n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row, col].set_axis_off()

    fig.suptitle(f"{subject_id} — ICA Component Topographies", fontsize=14)
    plt.tight_layout()
    fig.savefig(plot_dir / f"{subject_id}_ica_topographies.png", dpi=150)
    plt.close(fig)
    logger.info(f"[{subject_id}] Saved component topographies plot")

    # ---- Plot 2: Before/After PSD comparison (frontal channels) ----
    frontal_chs = _find_frontal_channels(raw, None)
    if frontal_chs:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for ax, r, title in [
            (axes[0], raw, "Before ICA"),
            (axes[1], raw_clean, "After ICA"),
        ]:
            psd = r.compute_psd(method="welch", fmin=1, fmax=45, picks=frontal_chs[:6])
            psd.plot(axes=ax, show=False, spatial_colors=False)
            ax.set_title(f"{subject_id} — Frontal Channels — {title}")

        plt.tight_layout()
        fig.savefig(plot_dir / f"{subject_id}_frontal_psd_before_after.png", dpi=150)
        plt.close(fig)
        logger.info(f"[{subject_id}] Saved frontal PSD before/after plot")

    # ---- Plot 3: Removed component time courses ----
    if removed_indices:
        sources = ica.get_sources(raw).get_data()
        n_removed = len(removed_indices)
        fig, axes = plt.subplots(
            n_removed, 1,
            figsize=(14, 1.5 * n_removed),
            squeeze=False,
        )
        times = raw.times[: len(sources[0])]
        for i, comp_idx in enumerate(removed_indices):
            ax = axes[i, 0]
            ax.plot(times, sources[comp_idx, :], linewidth=0.5, color="coral")
            ax.set_ylabel(f"IC{comp_idx}")
            ax.set_xlim(times[0], min(times[-1], 10))  # first 10s
            if i == n_removed - 1:
                ax.set_xlabel("Time (s)")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"{subject_id} — Removed Component Time Courses (first 10s)", fontsize=12)
        plt.tight_layout()
        fig.savefig(plot_dir / f"{subject_id}_removed_components.png", dpi=150)
        plt.close(fig)
        logger.info(f"[{subject_id}] Saved removed component time courses plot")

    # ---- Plot 4: Variance change by channel ----
    data_before = raw.get_data()
    data_after = raw_clean.get_data()
    ch_var_before = np.var(data_before, axis=1)
    ch_var_after = np.var(data_after, axis=1)
    ch_var_change = (ch_var_after - ch_var_before) / (ch_var_before + 1e-12) * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    colors = ["coral" if v < 0 else "steelblue" for v in ch_var_change]
    ax.bar(range(len(ch_var_change)), ch_var_change, color=colors, alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Channel Index")
    ax.set_ylabel("Variance Change (%)")
    ax.set_title(f"{subject_id} — Per-Channel Variance Change After ICA")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(plot_dir / f"{subject_id}_channel_variance_change.png", dpi=150)
    plt.close(fig)
    logger.info(f"[{subject_id}] Saved channel variance change plot")


# ============================================================================
# QC Report I/O
# ============================================================================


def save_ica_report(
    report: ICAReport,
    qc: dict,
    output_dir: str,
    subject_id: str,
) -> None:
    """
    Save ICA QC report as CSV.

    Parameters
    ----------
    report : ICAReport
        ICA processing report.
    qc : dict
        QC metrics from compute_ica_qc.
    output_dir : str
        Output directory.
    subject_id : str
        Subject identifier.
    """
    import pandas as pd

    out = Path(output_dir) / "ica_qc"
    out.mkdir(parents=True, exist_ok=True)

    # Flatten report + qc into a single-row DataFrame
    record = {
        "subject_id": subject_id,
        "n_components_total": report.n_components_total,
        "n_components_removed": report.n_components_removed,
        "removed_indices": str(report.removed_indices),
        "removed_labels": str(report.removed_labels),
        "detection_method": report.detection_method,
        "eog_channel_used": report.eog_channel_used or "none",
        "converged": report.converged,
        "n_iter": report.n_iter,
        **{f"qc_{k}": v for k, v in qc.items() if k != "subject_id"},
        "warnings": "; ".join(report.warnings),
    }

    df = pd.DataFrame([record])
    df.to_csv(out / f"{subject_id}_ica_report.csv", index=False)
    logger.info(f"[{subject_id}] Saved ICA report to {out}")


# ============================================================================
# Internal helpers
# ============================================================================


def _get_channel_topographies(ica: mne.preprocessing.ICA) -> np.ndarray:
    """
    Get ICA component topographies in channel space.

    MNE's ICA stores components in PCA-reduced space. This function
    reconstructs the full (n_channels × n_components) topography matrix
    by combining the PCA components with the ICA unmixing matrix.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA object.

    Returns
    -------
    topographies : np.ndarray
        Shape (n_channels, n_components). Each column is the spatial
        topography of one IC in channel space.
    """
    # pca_components_: (n_pca, n_channels) — PCA basis vectors
    # get_components(): (n_pca, n_components) — ICA unmixing in PCA space
    # Channel topographies = pca_components_.T @ get_components()
    pca = ica.pca_components_  # (n_pca, n_channels)
    unmixing = ica.get_components()  # (n_pca, n_components)
    topographies = pca.T @ unmixing  # (n_channels, n_components)
    return topographies


def _merge_config(user_config: Optional[dict]) -> dict:
    """Merge user config with defaults, user values take precedence."""
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

    # Check for NaN/Inf
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
    # ICA only uses EEG channels — count them for the upper bound
    n_eeg_channels = sum(
        1 for ch in raw.info["chs"] if ch["kind"] == mne.io.constants.FIFF.FIFFV_EEG_CH
    )
    if n_eeg_channels == 0:
        n_eeg_channels = len(raw.ch_names)  # fallback if no EEG type info

    if n_components_cfg is None:
        # Auto: use rank of data
        try:
            rank = mne.compute_rank(raw)
            n_comp = max(1, min(rank, n_eeg_channels - 1, 30))
            logger.info(
                f"[{subject_id}] Auto n_components: {n_comp} (rank={rank})"
            )
            return n_comp
        except Exception:
            n_comp = min(n_eeg_channels, 20)
            logger.info(
                f"[{subject_id}] Fallback n_components: {n_comp}"
            )
            return n_comp
    elif isinstance(n_components_cfg, int) and n_components_cfg > 0:
        actual = min(n_components_cfg, n_eeg_channels)
        if actual < n_components_cfg:
            logger.info(
                f"[{subject_id}] n_components reduced from {n_components_cfg} "
                f"to {actual} (limited by {n_eeg_channels} EEG channels)"
            )
        return actual
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
        logger.warning(
            f"Specified EOG channel '{eog_ch}' not found in data. "
            f"Trying auto-detection..."
        )

    # Auto-detect from common EOG channel names
    candidates = ["VEOG", "HEOG", "EOG", "Fp1", "Fp2", "Fpz", "eog", "veog", "heog"]
    for ch in candidates:
        if ch in raw.ch_names:
            return ch

    return None


def _find_frontal_channels(
    raw: mne.io.Raw,
    config: Optional[dict],
) -> List[str]:
    """
    Find frontal channels in the data.

    Uses the configured frontal channel list, or auto-detects channels
    with "Fp", "AF", or frontal "F" in their names.
    """
    cfg = _merge_config(config)
    configured = cfg.get("classification", {}).get("frontal_channels", [])

    if configured:
        return [ch for ch in configured if ch in raw.ch_names]

    # Auto-detect frontal channels
    frontal_patterns = ["Fp", "AF", "F"]
    frontal = []
    for ch in raw.ch_names:
        for pattern in frontal_patterns:
            if ch.startswith(pattern):
                frontal.append(ch)
                break
    return frontal


def _kurtosis(x: np.ndarray) -> float:
    """
    Compute excess kurtosis of a 1D array.

    Excess kurtosis = 0 for normal distribution.
    Higher values indicate heavier tails (more outliers).
    """
    n = len(x)
    if n < 4:
        return 0.0
    x = x - np.mean(x)
    m2 = np.mean(x ** 2)
    m4 = np.mean(x ** 4)
    if m2 < 1e-20:
        return 0.0
    return float(m4 / (m2 ** 2) - 3)


def _log_summary(report: ICAReport, subject_id: str) -> None:
    """Log a formatted summary of ICA results."""
    logger.info("=" * 60)
    logger.info(f"  ICA Artifact Removal Summary — {subject_id}")
    logger.info("=" * 60)
    logger.info(f"  Components total:   {report.n_components_total}")
    logger.info(f"  Components removed: {report.n_components_removed}")
    logger.info(f"  Detection method:   {report.detection_method}")
    logger.info(f"  EOG channel used:   {report.eog_channel_used or 'none'}")
    logger.info(f"  Converged:          {report.converged}")
    logger.info(f"  Iterations:         {report.n_iter}")
    logger.info(f"  Variance explained: {report.variance_explained_removed:.2%}")
    logger.info(
        f"  Frontal variance:   {report.frontal_variance_before:.4f} → "
        f"{report.frontal_variance_after:.4f} "
        f"({(report.frontal_variance_after / (report.frontal_variance_before + 1e-12) - 1) * 100:+.1f}%)"
    )
    logger.info(f"  Removed: {report.removed_indices}")
    logger.info(f"  Labels:  {report.removed_labels}")
    if report.warnings:
        logger.info(f"  Warnings: {report.warnings}")
    logger.info("=" * 60)
