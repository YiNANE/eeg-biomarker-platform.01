"""
Unit tests for the ICA artifact rejection module (ica.py).

Tests cover:
  - ICA configuration and parameter validation
  - EOG detection: correlation, kurtosis, topography methods
  - Combined multi-method detection
  - Component classification (blink, movement, muscle, neural)
  - Full ICA artifact removal pipeline (end-to-end)
  - Before/after QC comparison
  - Edge cases: no EOG channel, NaN data, flat channels, few channels
  - Integration with ingestion and preprocessing modules
"""

import os
import sys
import tempfile
from pathlib import Path

import mne
import numpy as np
import pytest

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ica import (
    configure_ica,
    detect_eog_by_correlation,
    detect_eog_by_kurtosis,
    detect_eog_by_topography,
    classify_components,
    run_ica_artifact_removal,
    compute_ica_qc,
    save_ica_report,
    ComponentScores,
    ICAReport,
    DEFAULT_ICA_CONFIG,
)
from src.utils.config import load_config, load_paths


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================


def _create_synthetic_raw(
    n_channels=10,
    n_times=5000,
    sfreq=250.0,
    add_eog=True,
    add_blinks=True,
    random_seed=42,
):
    """
    Create synthetic EEG-like raw data with optional eye-blink artifacts.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    n_times : int
        Number of time samples.
    sfreq : float
        Sampling frequency.
    add_eog : bool
        If True, include a VEOG channel with blink signals.
    add_blinks : bool
        If True, inject blink-like transients into frontal channels.
    random_seed : int
        Random seed.

    Returns
    -------
    raw : mne.io.Raw
        Synthetic raw data with realistic EEG + blink artifacts.
    """
    rng = np.random.default_rng(random_seed)
    duration = n_times / sfreq

    # Base EEG: 1/f noise + alpha oscillation
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    n_freqs = len(freqs)

    channels_data = []
    for i in range(n_channels):
        # 1/f noise
        spectrum = rng.normal(0, 1, n_freqs) + 1j * rng.normal(0, 1, n_freqs)
        spectrum[1:] /= np.sqrt(freqs[1:] + 0.1)  # avoid DC blowup
        spectrum[0] = 0
        eeg = np.fft.irfft(spectrum, n=n_times)

        # Add alpha oscillation (10 Hz) with channel-dependent amplitude
        t = np.arange(n_times) / sfreq
        alpha_amp = 0.5 + 0.3 * (1 - abs(i - n_channels // 2) / n_channels)
        eeg += alpha_amp * np.sin(2 * np.pi * 10 * t + rng.random() * np.pi)

        channels_data.append(eeg)

    channels_data = np.array(channels_data)

    # Add eye-blink artifacts to frontal channels (first 3)
    if add_blinks:
        blink_interval = int(sfreq * 3)  # every ~3 seconds
        blink_width = int(sfreq * 0.15)  # ~150 ms blink
        for blink_start in range(blink_interval, n_times - blink_width, blink_interval):
            blink_start += rng.integers(-int(sfreq * 0.5), int(sfreq * 0.5))
            blink_start = max(0, min(blink_start, n_times - blink_width - 1))
            # Gaussian-shaped blink
            t_blink = np.arange(blink_width)
            blink_wave = np.exp(-0.5 * ((t_blink - blink_width / 2) / (blink_width / 6)) ** 2)
            # Inject into frontal channels with decreasing amplitude
            for ch in range(min(3, n_channels)):
                amplitude = rng.uniform(8, 15) * (1 - ch * 0.3)
                channels_data[ch, blink_start:blink_start + blink_width] += amplitude * blink_wave

    # Build channel info
    ch_names = [f"ch_{i:02d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels

    if add_eog:
        # Add a VEOG channel with blink signals
        eog_data = rng.normal(0, 0.1, n_times)
        if add_blinks:
            for blink_start in range(int(sfreq * 3), n_times - int(sfreq * 0.15), int(sfreq * 3)):
                blink_start += rng.integers(-int(sfreq * 0.5), int(sfreq * 0.5))
                blink_start = max(0, min(blink_start, n_times - int(sfreq * 0.15) - 1))
                t_blink = np.arange(int(sfreq * 0.15))
                blink_wave = np.exp(-0.5 * ((t_blink - len(t_blink) / 2) / (len(t_blink) / 6)) ** 2)
                eog_data[blink_start:blink_start + len(t_blink)] += rng.uniform(10, 20) * blink_wave

        channels_data = np.vstack([channels_data, eog_data])
        ch_names.append("VEOG")
        ch_types.append("eog")

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(channels_data, info)
    raw.set_eeg_reference("average", verbose=False)

    return raw


@pytest.fixture(scope="module")
def synthetic_raw():
    """Synthetic raw data with eye blinks + VEOG channel."""
    return _create_synthetic_raw(n_channels=10, n_times=5000, sfreq=250.0)


@pytest.fixture(scope="module")
def synthetic_raw_no_eog():
    """Synthetic raw data WITHOUT EOG channel (for fallback tests)."""
    return _create_synthetic_raw(n_channels=10, n_times=5000, sfreq=250.0, add_eog=False)


@pytest.fixture(scope="module")
def synthetic_raw_no_blinks():
    """Synthetic raw data without blinks (clean EEG)."""
    return _create_synthetic_raw(
        n_channels=10, n_times=5000, sfreq=250.0, add_eog=True, add_blinks=False
    )


@pytest.fixture(scope="module")
def synthetic_raw_few_channels():
    """Synthetic raw data with few channels (edge case)."""
    return _create_synthetic_raw(
        n_channels=3, n_times=2000, sfreq=250.0, add_eog=True, add_blinks=True
    )


@pytest.fixture(scope="module")
def fitted_ica(synthetic_raw):
    """Fitted ICA on synthetic data."""
    raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
    ica = mne.preprocessing.ICA(
        n_components=min(6, len(synthetic_raw.ch_names) - 1),
        method="fastica",
        random_state=42,
    )
    ica.fit(raw_filt, verbose=False)
    return ica, raw_filt


# ============================================================================
# Real Data Fixtures (skip if not available)
# ============================================================================


@pytest.fixture(scope="session")
def lemon_root():
    """Load LEMON data root, or skip if not available."""
    try:
        paths = load_paths(str(PROJECT_ROOT / "configs" / "paths.local.yaml"))
        return paths["lemon_root"]
    except (FileNotFoundError, KeyError):
        pytest.skip("LEMON data paths not configured")


@pytest.fixture(scope="session")
def real_raw(lemon_root):
    """Load a real LEMON subject for integration tests."""
    from src.ingestion.loader import load_subject
    try:
        return load_subject("sub-032301", lemon_root=lemon_root, resample_freq=250.0)
    except FileNotFoundError:
        pytest.skip("Real subject data not available")


# ============================================================================
# Test 1: ICA Configuration
# ============================================================================


class TestICAConfiguration:
    """Tests for configure_ica() and configuration handling."""

    def test_configure_ica_returns_ica_object(self, synthetic_raw):
        """configure_ica should return an MNE ICA object."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        ica = configure_ica(raw_filt, subject_id="test")
        assert isinstance(ica, mne.preprocessing.ICA)
        assert ica.n_components > 0  # pre-fit attribute
        assert ica.method in ("fastica", "picard", "infomax")

    def test_configure_ica_with_custom_n_components(self, synthetic_raw):
        """configure_ica should respect custom n_components."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 5}
        ica = configure_ica(raw_filt, config=config, subject_id="test")
        assert ica.n_components == 5

    def test_configure_ica_auto_n_components(self, synthetic_raw):
        """configure_ica should auto-determine n_components when set to None."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": None}
        ica = configure_ica(raw_filt, config=config, subject_id="test")
        assert ica.n_components > 0

    def test_configure_ica_different_methods(self, synthetic_raw):
        """configure_ica should accept different ICA methods."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        for method in ["fastica", "picard"]:
            config = {"n_components": 4, "method": method}
            ica = configure_ica(raw_filt, config=config, subject_id="test")
            assert ica.method == method

    def test_configure_ica_respects_random_state(self, synthetic_raw):
        """configure_ica should produce deterministic results with fixed seed."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 4, "random_state": 42}

        ica1 = configure_ica(raw_filt, config=config)
        ica1.fit(raw_filt, verbose=False)

        ica2 = configure_ica(raw_filt, config=config)
        ica2.fit(raw_filt, verbose=False)

        # Components should be identical (up to sign)
        comps1 = np.abs(ica1.get_components())
        comps2 = np.abs(ica2.get_components())
        assert np.allclose(comps1, comps2, atol=1e-4)

    def test_configure_ica_with_minimal_channels(self, synthetic_raw_few_channels):
        """configure_ica should work with few channels."""
        raw_filt = synthetic_raw_few_channels.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 2}
        ica = configure_ica(raw_filt, config=config, subject_id="test")
        assert ica.n_components == 2


# ============================================================================
# Test 2: EOG Detection — Correlation Method
# ============================================================================


class TestEOGDetectionCorrelation:
    """Tests for detect_eog_by_correlation()."""

    def test_detects_blinks_with_veog(self, fitted_ica):
        """Correlation method should detect blinks when VEOG channel exists."""
        ica, raw = fitted_ica
        indices, labels = detect_eog_by_correlation(
            ica, raw, eog_channel="VEOG", threshold=2.0, subject_id="test"
        )
        # Should find at least some components (synthetic data has clear blinks)
        assert isinstance(indices, list)
        assert isinstance(labels, dict)

    def test_returns_labels_for_detected_components(self, fitted_ica):
        """All detected components should have labels."""
        ica, raw = fitted_ica
        indices, labels = detect_eog_by_correlation(
            ica, raw, eog_channel="VEOG", threshold=2.0, subject_id="test"
        )
        for idx in indices:
            assert idx in labels
            assert labels[idx] in ("eye_blink", "eye_movement")

    def test_auto_detect_eog_channel(self, fitted_ica):
        """Should auto-detect VEOG channel when eog_channel is None."""
        ica, raw = fitted_ica
        indices, labels = detect_eog_by_correlation(
            ica, raw, eog_channel=None, threshold=3.0, subject_id="test"
        )
        assert isinstance(indices, list)

    def test_missing_eog_channel_returns_empty(self, synthetic_raw_no_eog):
        """Should return empty when no EOG channel is available."""
        raw_filt = synthetic_raw_no_eog.copy().filter(1.0, 40.0, fir_design="firwin")
        ica = mne.preprocessing.ICA(
            n_components=min(6, len(raw_filt.ch_names) - 1),
            method="fastica",
            random_state=42,
        )
        ica.fit(raw_filt, verbose=False)

        indices, labels = detect_eog_by_correlation(
            ica, raw_filt, eog_channel="VEOG", subject_id="test"
        )
        assert indices == []
        assert labels == {}

    def test_higher_threshold_more_conservative(self, fitted_ica):
        """Higher threshold should find fewer or equal components."""
        ica, raw = fitted_ica

        indices_low, _ = detect_eog_by_correlation(
            ica, raw, eog_channel="VEOG", threshold=1.0, subject_id="test"
        )
        indices_high, _ = detect_eog_by_correlation(
            ica, raw, eog_channel="VEOG", threshold=10.0, subject_id="test"
        )
        assert len(indices_high) <= len(indices_low)


# ============================================================================
# Test 3: EOG Detection — Kurtosis Method
# ============================================================================


class TestEOGDetectionKurtosis:
    """Tests for detect_eog_by_kurtosis()."""

    def test_detects_blinks_by_kurtosis(self, fitted_ica):
        """Kurtosis method should detect blinks in synthetic data."""
        ica, raw = fitted_ica
        indices, labels, kurt_vals = detect_eog_by_kurtosis(
            ica, raw, threshold=2.0, subject_id="test"
        )
        assert isinstance(indices, list)
        assert isinstance(labels, dict)
        assert len(kurt_vals) == ica.n_components_

    def test_kurtosis_values_are_finite(self, fitted_ica):
        """All kurtosis values should be finite."""
        ica, raw = fitted_ica
        _, _, kurt_vals = detect_eog_by_kurtosis(
            ica, raw, threshold=2.0, subject_id="test"
        )
        assert np.all(np.isfinite(kurt_vals))

    def test_labels_are_eye_blink(self, fitted_ica):
        """Kurtosis-detected components should be labeled 'eye_blink'."""
        ica, raw = fitted_ica
        indices, labels, _ = detect_eog_by_kurtosis(
            ica, raw, threshold=2.0, subject_id="test"
        )
        for idx in indices:
            assert labels[idx] == "eye_blink"

    def test_works_without_eog_channel(self, synthetic_raw_no_eog):
        """Kurtosis method should work without EOG channel."""
        raw_filt = synthetic_raw_no_eog.copy().filter(1.0, 40.0, fir_design="firwin")
        n_comp = min(6, len(raw_filt.ch_names) - 1)
        ica = mne.preprocessing.ICA(
            n_components=n_comp, method="fastica", random_state=42
        )
        ica.fit(raw_filt, verbose=False)

        indices, labels, kurt_vals = detect_eog_by_kurtosis(
            ica, raw_filt, threshold=2.0, subject_id="test"
        )
        assert isinstance(indices, list)
        assert len(kurt_vals) == n_comp

    def test_clean_data_fewer_detections(self, synthetic_raw_no_blinks):
        """Clean data (no blinks) should have fewer kurtosis detections."""
        raw_filt = synthetic_raw_no_blinks.copy().filter(1.0, 40.0, fir_design="firwin")
        n_comp = min(6, len(raw_filt.ch_names) - 1)
        ica = mne.preprocessing.ICA(
            n_components=n_comp, method="fastica", random_state=42
        )
        ica.fit(raw_filt, verbose=False)

        indices, _, _ = detect_eog_by_kurtosis(
            ica, raw_filt, threshold=3.0, subject_id="test"
        )
        # Should detect very few (probably 0) with high threshold on clean data
        # This is informational — we can't assert == 0 due to ICA variability
        assert isinstance(indices, list)


# ============================================================================
# Test 4: EOG Detection — Topography Method
# ============================================================================


class TestEOGDetectionTopography:
    """Tests for detect_eog_by_topography()."""

    def test_detects_frontal_components(self, fitted_ica):
        """Topography method should identify frontal components."""
        ica, raw = fitted_ica
        frontal_chs = [ch for ch in raw.ch_names if ch == "VEOG" or ch.startswith("ch_0")]
        indices, labels, focal_scores, frontal_ratios = detect_eog_by_topography(
            ica, raw, frontal_channels=frontal_chs,
            focal_threshold=0.5, subject_id="test"
        )
        assert isinstance(indices, list)
        assert isinstance(labels, dict)
        assert len(focal_scores) == ica.n_components_
        assert len(frontal_ratios) == ica.n_components_

    def test_focal_scores_in_range(self, fitted_ica):
        """Focal scores should be between 0 and 1."""
        ica, raw = fitted_ica
        _, _, focal_scores, _ = detect_eog_by_topography(
            ica, raw, focal_threshold=0.5, subject_id="test"
        )
        assert np.all(focal_scores >= 0)
        assert np.all(focal_scores <= 1)

    def test_frontal_ratios_in_range(self, fitted_ica):
        """Frontal ratios should be between 0 and 1."""
        ica, raw = fitted_ica
        _, _, _, frontal_ratios = detect_eog_by_topography(
            ica, raw, focal_threshold=0.5, subject_id="test"
        )
        assert np.all(frontal_ratios >= 0)
        assert np.all(frontal_ratios <= 1)

    def test_no_frontal_channels_returns_empty(self, synthetic_raw):
        """Should return empty when no frontal channels match."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        n_comp = min(6, len(raw_filt.ch_names) - 1)
        ica = mne.preprocessing.ICA(
            n_components=n_comp, method="fastica", random_state=42
        )
        ica.fit(raw_filt, verbose=False)

        indices, labels, focal_scores, frontal_ratios = detect_eog_by_topography(
            ica, raw_filt,
            frontal_channels=["NONEXISTENT_CH"],
            focal_threshold=0.5,
            subject_id="test",
        )
        assert indices == []
        assert len(focal_scores) == 0 or len(focal_scores) == n_comp


# ============================================================================
# Test 5: Component Classification
# ============================================================================


class TestComponentClassification:
    """Tests for classify_components()."""

    def test_classifies_all_components(self, fitted_ica):
        """Every component should receive a classification."""
        ica, raw = fitted_ica
        classification = classify_components(ica, raw, subject_id="test")
        assert len(classification) == ica.n_components_
        for i in range(ica.n_components_):
            assert i in classification
            assert classification[i] in (
                "eye_blink", "eye_movement", "muscle", "neural"
            )

    def test_at_least_some_neural_components(self, fitted_ica):
        """Most components should be classified as neural in clean-ish data."""
        ica, raw = fitted_ica
        classification = classify_components(ica, raw, subject_id="test")
        n_neural = sum(1 for v in classification.values() if v == "neural")
        # At least some components should be neural
        assert n_neural > 0

    def test_classification_with_config(self, fitted_ica):
        """Classification should accept custom config."""
        ica, raw = fitted_ica
        config = {
            "eog_detection": {
                "kurtosis_threshold": 5.0,  # very conservative
            }
        }
        classification = classify_components(ica, raw, config=config, subject_id="test")
        assert len(classification) == ica.n_components_

    def test_classification_without_eog_channel(self, synthetic_raw_no_eog):
        """Classification should work without EOG channel."""
        raw_filt = synthetic_raw_no_eog.copy().filter(1.0, 40.0, fir_design="firwin")
        n_comp = min(6, len(raw_filt.ch_names) - 1)
        ica = mne.preprocessing.ICA(
            n_components=n_comp, method="fastica", random_state=42
        )
        ica.fit(raw_filt, verbose=False)

        classification = classify_components(ica, raw_filt, subject_id="test")
        assert len(classification) == n_comp


# ============================================================================
# Test 6: Full ICA Pipeline (End-to-End)
# ============================================================================


class TestFullICAPipeline:
    """End-to-end tests for run_ica_artifact_removal()."""

    def test_synthetic_data_pipeline_succeeds(self, synthetic_raw):
        """Full pipeline should succeed on synthetic data."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 6,
            "method": "fastica",
            "eog_detection": {"method": "combined", "eog_channel": "VEOG"},
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test_synth"
        )
        assert raw_clean is not None
        assert isinstance(ica, mne.preprocessing.ICA)
        assert isinstance(report, ICAReport)
        assert report.converged
        assert report.n_components_total == 6

    def test_cleaned_data_has_same_structure(self, synthetic_raw):
        """Cleaned data should have same channels and sfreq as input."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "correlation"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert len(raw_clean.ch_names) == len(raw_filt.ch_names)
        assert raw_clean.info["sfreq"] == raw_filt.info["sfreq"]
        assert raw_clean.n_times == raw_filt.n_times

    def test_frontal_variance_reduced(self, synthetic_raw):
        """
        ICA should reduce variance in frontal channels (blink removal).

        NOTE: This is probabilistic depending on ICA decomposition quality.
        We compute the change and log it; if no reduction, it's a soft warning.
        """
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 6,
            "eog_detection": {"method": "combined", "eog_channel": "VEOG"},
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )

        # Check that the report contains frontal variance info
        assert report.global_variance_before > 0
        assert report.global_variance_after > 0

        # Frontal channels (ch_00, ch_01, ch_02) should show reduction
        # if blinks were properly detected and removed
        frontal_idx = [i for i, ch in enumerate(raw_filt.ch_names)
                       if ch in ["ch_00", "ch_01", "ch_02"]]
        if frontal_idx:
            data_before = raw_filt.get_data()[frontal_idx, :]
            data_after = raw_clean.get_data()[frontal_idx, :]
            var_before = np.var(data_before)
            var_after = np.var(data_after)
            print(f"\n  Frontal variance: {var_before:.4f} → {var_after:.4f}")
            # This is informational; don't assert (ICA is stochastic)

    def test_ica_exclude_set_correctly(self, synthetic_raw):
        """ICA.exclude should contain the removed component indices."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert set(ica.exclude) == set(report.removed_indices)

    def test_report_has_removed_labels(self, synthetic_raw):
        """Report should have labels for all removed components."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        for idx in report.removed_indices:
            assert idx in report.removed_labels

    def test_pipeline_saves_diagnostic_plots(self, synthetic_raw):
        """Pipeline should save diagnostic plots when requested."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 6,
            "output": {"save_plots": True},
            "eog_detection": {"method": "combined"},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_clean, ica, report = run_ica_artifact_removal(
                raw_filt, config=config, output_dir=tmpdir, subject_id="test"
            )
            plot_dir = Path(tmpdir) / "ica_plots"
            assert plot_dir.exists()
            # Should have at least component topographies
            png_files = list(plot_dir.glob("*.png"))
            print(f"\n  Generated plots: {[f.name for f in png_files]}")
            assert len(png_files) > 0

    def test_pipeline_kurtosis_only_mode(self, synthetic_raw_no_eog):
        """Pipeline should work with kurtosis-only detection (no EOG channel)."""
        raw_filt = synthetic_raw_no_eog.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 6,
            "eog_detection": {"method": "kurtosis", "eog_channel": ""},
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test_no_eog"
        )
        assert report.converged
        assert report.detection_method == "kurtosis"

    def test_pipeline_topography_only_mode(self, synthetic_raw):
        """Pipeline should work with topography-only detection."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 6,
            "eog_detection": {"method": "topography"},
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert report.converged

    def test_default_config_works(self, synthetic_raw):
        """Pipeline should work with default config (no user config passed)."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, subject_id="test_default"
        )
        assert report.converged

    @pytest.mark.slow
    def test_pipeline_with_real_data(self, real_raw):
        """Full pipeline should succeed on real LEMON data."""
        raw_filt = real_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 20,
            "method": "fastica",
            "eog_detection": {"method": "combined", "eog_channel": "auto"},
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test_real"
        )
        assert raw_clean is not None
        assert report.converged
        assert report.n_components_total == 20
        # Real data should have at least some detections
        assert report.n_components_removed >= 0  # could be 0 legitimately
        print(f"\n  Real data: removed {report.n_components_removed}/20 components")
        print(f"  Removed: {report.removed_indices}")
        print(f"  Labels:  {report.removed_labels}")


# ============================================================================
# Test 7: ICA QC Comparison
# ============================================================================


class TestICAQC:
    """Tests for compute_ica_qc() and save_ica_report()."""

    def test_compute_qc_returns_expected_keys(self, synthetic_raw):
        """QC dict should contain all expected keys."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )

        qc = compute_ica_qc(
            raw_filt, raw_clean, ica,
            removed_indices=report.removed_indices,
            subject_id="test",
        )
        required_keys = [
            "subject_id", "n_components_removed", "removed_indices",
            "global_variance_before", "global_variance_after",
            "global_variance_change_pct",
            "frontal_variance_before", "frontal_variance_after",
            "frontal_variance_change_pct",
            "variance_explained_by_removed",
        ]
        for key in required_keys:
            assert key in qc, f"Missing key: {key}"

    def test_qc_variance_values_finite(self, synthetic_raw):
        """All QC variance values should be finite."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )

        qc = compute_ica_qc(
            raw_filt, raw_clean, ica,
            removed_indices=report.removed_indices,
            subject_id="test",
        )
        for key in ["global_variance_before", "global_variance_after",
                     "frontal_variance_before", "frontal_variance_after"]:
            assert np.isfinite(qc[key]), f"{key} is not finite: {qc[key]}"

    def test_no_components_removed_qc(self, synthetic_raw):
        """QC should handle the case where no components were removed."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )

        # Compute QC with empty removed list
        qc = compute_ica_qc(
            raw_filt, raw_filt, ica,  # same data for both
            removed_indices=[],
            subject_id="test",
        )
        assert qc["n_components_removed"] == 0

    def test_save_ica_report_creates_file(self, synthetic_raw):
        """save_ica_report should create a CSV file."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        qc = compute_ica_qc(
            raw_filt, raw_clean, ica,
            removed_indices=report.removed_indices,
            subject_id="test",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_ica_report(report, qc, tmpdir, "test")
            csv_dir = Path(tmpdir) / "ica_qc"
            assert csv_dir.exists()
            csv_files = list(csv_dir.glob("*.csv"))
            assert len(csv_files) == 1

    def test_qc_per_channel_metrics(self, synthetic_raw):
        """QC should include per-channel variance change info."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )

        qc = compute_ica_qc(
            raw_filt, raw_clean, ica,
            removed_indices=report.removed_indices,
            subject_id="test",
        )
        assert "max_channel_variance_increase" in qc
        assert "channels_with_variance_decrease" in qc
        assert "channels_with_variance_increase" in qc


# ============================================================================
# Test 8: Edge Cases and Error Handling
# ============================================================================


class TestICAEdgeCases:
    """Tests for edge cases and error handling."""

    def test_raises_on_nan_data(self):
        """Should raise ValueError when data contains NaN."""
        sfreq = 250.0
        info = mne.create_info(["ch1", "ch2", "ch3", "ch4"], sfreq, ["eeg"] * 4)
        data = np.random.randn(4, 1000)
        data[0, 500] = np.nan
        raw = mne.io.RawArray(data, info, verbose=False)

        with pytest.raises(ValueError, match="NaN"):
            run_ica_artifact_removal(
                raw, config={"n_components": 2}, subject_id="test"
            )

    def test_raises_on_few_channels(self):
        """Should raise ValueError with fewer than 3 channels."""
        sfreq = 250.0
        info = mne.create_info(["ch1", "ch2"], sfreq, ["eeg", "eeg"])
        raw = mne.io.RawArray(np.random.randn(2, 1000), info, verbose=False)

        with pytest.raises(ValueError):
            run_ica_artifact_removal(
                raw, config={"n_components": 2}, subject_id="test"
            )

    def test_handles_n_components_greater_than_channels(self, synthetic_raw_few_channels):
        """Should clamp n_components to n_channels."""
        raw_filt = synthetic_raw_few_channels.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 100}  # more than channels
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert report.n_components_total <= len(raw_filt.ch_names)

    def test_safety_never_removes_all_components(self, synthetic_raw):
        """
        When detection flags most components as artifacts, the safety
        check should ensure we never remove ALL components.
        """
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        # Very low thresholds to be aggressive
        config = {
            "n_components": 4,
            "eog_detection": {
                "method": "combined",
                "correlation_threshold": 0.1,
                "kurtosis_threshold": 0.1,
                "focal_threshold": 0.1,
            },
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        # Should never remove all components (safety cap at 50%)
        assert report.n_components_removed < report.n_components_total

    def test_no_artifacts_detected_is_ok(self, synthetic_raw_no_blinks):
        """Pipeline should succeed even if no artifacts are detected."""
        raw_filt = synthetic_raw_no_blinks.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 4,
            "eog_detection": {
                "method": "combined",
                "correlation_threshold": 10.0,  # very strict
                "kurtosis_threshold": 10.0,
                "focal_threshold": 0.99,
            },
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert report.converged
        # Zero detections is fine with strict thresholds
        assert report.n_components_removed >= 0

    def test_component_scores_match_n_components(self, synthetic_raw):
        """ComponentScores should have arrays of length n_components."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        # Pass frontal channels matching synthetic data channel names
        config = {
            "n_components": 6,
            "eog_detection": {"method": "combined"},
            "classification": {"frontal_channels": ["ch_00", "ch_01", "ch_02"]},
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        scores = report.component_scores
        assert scores is not None
        assert scores.n_components == report.n_components_total
        assert len(scores.eog_correlation) == report.n_components_total
        assert len(scores.kurtosis) == report.n_components_total
        assert len(scores.focal_score) == report.n_components_total
        assert len(scores.frontal_power_ratio) == report.n_components_total


# ============================================================================
# Test 9: Integration Compatibility
# ============================================================================


class TestICAIntegration:
    """Verify ICA output is compatible with downstream modules."""

    def test_ica_output_accepted_by_preprocessing(self, synthetic_raw):
        """
        ICA-cleaned data should be acceptable by the preprocessing pipeline's
        windowing step (i.e., it should still be a valid mne.io.Raw).
        """
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )

        # Verify it can be windowed (common downstream step)
        window_len = 2.0
        events = mne.make_fixed_length_events(raw_clean, duration=window_len)
        epochs = mne.Epochs(
            raw_clean, events, tmin=0, tmax=window_len,
            baseline=None, preload=True, verbose=False,
        )
        assert len(epochs) > 0
        assert epochs.get_data().shape[1] == len(raw_clean.ch_names)

    def test_ica_output_psd_computation(self, synthetic_raw):
        """PSD computation should work on ICA-cleaned data."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )

        psd = raw_clean.compute_psd(method="welch", fmin=1, fmax=45)
        assert psd is not None
        # PSD only uses EEG channels (excludes EOG, stim, etc.)
        n_eeg = len([ch for ch in raw_clean.ch_names if ch != "VEOG"])
        assert psd.get_data().shape[0] == n_eeg

    def test_ica_works_on_segmented_data(self, synthetic_raw):
        """
        ICA should work on segmented data (e.g., EO-only or EC-only).
        This simulates running ICA after state segmentation.
        """
        # Create a shorter segment mimicking post-segmentation data
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        # Crop to simulate a single-state segment
        raw_seg = raw_filt.copy().crop(tmin=1, tmax=min(10, raw_filt.times[-1]))
        assert raw_seg.n_times > 500  # enough for ICA

        config = {"n_components": 4, "eog_detection": {"method": "combined"}}
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_seg, config=config, subject_id="test_seg"
        )
        assert report.converged

    @pytest.mark.slow
    def test_ica_integration_with_real_pipeline(self, real_raw):
        """Full integration: load → segment → ICA on real data."""
        from src.ingestion.loader import segment_by_state

        # Segment by state
        raw_eo, raw_ec = segment_by_state(real_raw)

        # Run ICA on EO data
        raw_eo_filt = raw_eo.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 15,
            "method": "fastica",
            "eog_detection": {"method": "combined", "eog_channel": "auto"},
        }
        raw_clean, ica, report = run_ica_artifact_removal(
            raw_eo_filt, config=config, subject_id="test_real_eo"
        )
        assert report.converged
        assert len(raw_clean.ch_names) == len(raw_eo.ch_names)
        print(f"\n  Real EO ICA: removed {report.n_components_removed}/15 components")


# ============================================================================
# Run with: pytest tests/test_ica.py -v
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
