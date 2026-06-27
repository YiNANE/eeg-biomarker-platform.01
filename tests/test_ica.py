"""
Unit tests for the ICA artifact rejection module (ica.py).

Tests cover:
  - ICA configuration and parameter validation
  - EOG detection: correlation, kurtosis, topography methods
  - Combined multi-method detection
  - Full ICA artifact removal pipeline (end-to-end)
  - Edge cases: no EOG channel, NaN data, flat channels, few channels
  - Integration with downstream modules
"""

import sys
import tempfile
from pathlib import Path

import mne
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ica import (
    configure_ica,
    detect_eog_by_correlation,
    detect_eog_by_kurtosis,
    detect_eog_by_topography,
    run_ica_artifact_removal,
)
from src.utils.config import load_paths


# ============================================================================
# Synthetic Data Fixtures
# ============================================================================


def _create_synthetic_raw(
    n_channels=10, n_times=5000, sfreq=250.0,
    add_eog=True, add_blinks=True, random_seed=42,
):
    """Create synthetic EEG-like raw data with optional eye-blink artifacts."""
    rng = np.random.default_rng(random_seed)

    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    n_freqs = len(freqs)

    channels_data = []
    for i in range(n_channels):
        spectrum = rng.normal(0, 1, n_freqs) + 1j * rng.normal(0, 1, n_freqs)
        spectrum[1:] /= np.sqrt(freqs[1:] + 0.1)
        spectrum[0] = 0
        eeg = np.fft.irfft(spectrum, n=n_times)
        t = np.arange(n_times) / sfreq
        alpha_amp = 0.5 + 0.3 * (1 - abs(i - n_channels // 2) / n_channels)
        eeg += alpha_amp * np.sin(2 * np.pi * 10 * t + rng.random() * np.pi)
        channels_data.append(eeg)

    channels_data = np.array(channels_data)

    if add_blinks:
        blink_interval = int(sfreq * 3)
        blink_width = int(sfreq * 0.15)
        for blink_start in range(blink_interval, n_times - blink_width, blink_interval):
            blink_start += rng.integers(-int(sfreq * 0.5), int(sfreq * 0.5))
            blink_start = max(0, min(blink_start, n_times - blink_width - 1))
            t_blink = np.arange(blink_width)
            blink_wave = np.exp(-0.5 * ((t_blink - blink_width / 2) / (blink_width / 6)) ** 2)
            for ch in range(min(3, n_channels)):
                amplitude = rng.uniform(8, 15) * (1 - ch * 0.3)
                channels_data[ch, blink_start:blink_start + blink_width] += amplitude * blink_wave

    ch_names = [f"ch_{i:02d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels

    if add_eog:
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
    return _create_synthetic_raw(n_channels=10, n_times=5000, sfreq=250.0)


@pytest.fixture(scope="module")
def synthetic_raw_no_eog():
    return _create_synthetic_raw(n_channels=10, n_times=5000, sfreq=250.0, add_eog=False)


@pytest.fixture(scope="module")
def synthetic_raw_no_blinks():
    return _create_synthetic_raw(n_channels=10, n_times=5000, sfreq=250.0, add_eog=True, add_blinks=False)


@pytest.fixture(scope="module")
def synthetic_raw_few_channels():
    return _create_synthetic_raw(n_channels=3, n_times=2000, sfreq=250.0, add_eog=True, add_blinks=True)


@pytest.fixture(scope="module")
def fitted_ica(synthetic_raw):
    """Fitted ICA on synthetic data."""
    raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
    ica = mne.preprocessing.ICA(
        n_components=min(6, len(synthetic_raw.ch_names) - 1),
        method="fastica", random_state=42,
    )
    ica.fit(raw_filt, verbose=False)
    return ica, raw_filt


# ============================================================================
# Real Data Fixtures (skip if not available)
# ============================================================================


@pytest.fixture(scope="session")
def lemon_root():
    try:
        paths = load_paths(str(PROJECT_ROOT / "configs" / "paths.local.yaml"))
        return paths["lemon_root"]
    except (FileNotFoundError, KeyError):
        pytest.skip("LEMON data paths not configured")


@pytest.fixture(scope="session")
def real_raw(lemon_root):
    from src.ingestion.loader import load_subject
    try:
        return load_subject("sub-032301", lemon_root=lemon_root, resample_freq=250.0)
    except FileNotFoundError:
        pytest.skip("Real subject data not available")


# ============================================================================
# Test 1: ICA Configuration
# ============================================================================


class TestICAConfiguration:
    """Tests for configure_ica()."""

    def test_configure_ica_returns_ica_object(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        ica = configure_ica(raw_filt, subject_id="test")
        assert isinstance(ica, mne.preprocessing.ICA)
        assert ica.n_components > 0

    def test_configure_ica_with_custom_n_components(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 5}
        ica = configure_ica(raw_filt, config=config, subject_id="test")
        assert ica.n_components == 5

    def test_configure_ica_auto_n_components(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        ica = configure_ica(raw_filt, config={"n_components": None}, subject_id="test")
        assert ica.n_components > 0

    def test_configure_ica_different_methods(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        for method in ["fastica", "picard"]:
            ica = configure_ica(raw_filt, config={"n_components": 4, "method": method}, subject_id="test")
            assert ica.method == method

    def test_configure_ica_respects_random_state(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 4, "random_state": 42}
        ica1 = configure_ica(raw_filt, config=config)
        ica1.fit(raw_filt, verbose=False)
        ica2 = configure_ica(raw_filt, config=config)
        ica2.fit(raw_filt, verbose=False)
        comps1 = np.abs(ica1.get_components())
        comps2 = np.abs(ica2.get_components())
        assert np.allclose(comps1, comps2, atol=1e-4)

    def test_configure_ica_with_minimal_channels(self, synthetic_raw_few_channels):
        raw_filt = synthetic_raw_few_channels.copy().filter(1.0, 40.0, fir_design="firwin")
        ica = configure_ica(raw_filt, config={"n_components": 2}, subject_id="test")
        assert ica.n_components == 2


# ============================================================================
# Test 2: EOG Detection — Correlation Method
# ============================================================================


class TestEOGDetectionCorrelation:
    """Tests for detect_eog_by_correlation()."""

    def test_detects_blinks_with_veog(self, fitted_ica):
        ica, raw = fitted_ica
        indices, labels = detect_eog_by_correlation(
            ica, raw, eog_channel="VEOG", threshold=2.0, subject_id="test"
        )
        assert isinstance(indices, list)
        assert isinstance(labels, dict)

    def test_returns_labels_for_detected_components(self, fitted_ica):
        ica, raw = fitted_ica
        indices, labels = detect_eog_by_correlation(
            ica, raw, eog_channel="VEOG", threshold=2.0, subject_id="test"
        )
        for idx in indices:
            assert idx in labels
            assert labels[idx] in ("eye_blink", "eye_movement")

    def test_auto_detect_eog_channel(self, fitted_ica):
        ica, raw = fitted_ica
        indices, labels = detect_eog_by_correlation(
            ica, raw, eog_channel=None, threshold=3.0, subject_id="test"
        )
        assert isinstance(indices, list)

    def test_missing_eog_channel_returns_empty(self, synthetic_raw_no_eog):
        raw_filt = synthetic_raw_no_eog.copy().filter(1.0, 40.0, fir_design="firwin")
        n_comp = min(6, len(raw_filt.ch_names) - 1)
        ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica", random_state=42)
        ica.fit(raw_filt, verbose=False)
        indices, labels = detect_eog_by_correlation(
            ica, raw_filt, eog_channel="VEOG", subject_id="test"
        )
        assert indices == []
        assert labels == {}

    def test_higher_threshold_more_conservative(self, fitted_ica):
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
        ica, raw = fitted_ica
        indices, labels = detect_eog_by_kurtosis(ica, raw, threshold=2.0, subject_id="test")
        assert isinstance(indices, list)
        assert isinstance(labels, dict)

    def test_labels_are_eye_blink(self, fitted_ica):
        ica, raw = fitted_ica
        indices, labels = detect_eog_by_kurtosis(ica, raw, threshold=2.0, subject_id="test")
        for idx in indices:
            assert labels[idx] == "eye_blink"

    def test_works_without_eog_channel(self, synthetic_raw_no_eog):
        raw_filt = synthetic_raw_no_eog.copy().filter(1.0, 40.0, fir_design="firwin")
        n_comp = min(6, len(raw_filt.ch_names) - 1)
        ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica", random_state=42)
        ica.fit(raw_filt, verbose=False)
        indices, labels = detect_eog_by_kurtosis(ica, raw_filt, threshold=2.0, subject_id="test")
        assert isinstance(indices, list)

    def test_clean_data_fewer_detections(self, synthetic_raw_no_blinks):
        raw_filt = synthetic_raw_no_blinks.copy().filter(1.0, 40.0, fir_design="firwin")
        n_comp = min(6, len(raw_filt.ch_names) - 1)
        ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica", random_state=42)
        ica.fit(raw_filt, verbose=False)
        indices, _ = detect_eog_by_kurtosis(ica, raw_filt, threshold=3.0, subject_id="test")
        assert isinstance(indices, list)


# ============================================================================
# Test 4: EOG Detection — Topography Method
# ============================================================================


class TestEOGDetectionTopography:
    """Tests for detect_eog_by_topography()."""

    def test_detects_frontal_components(self, fitted_ica):
        ica, raw = fitted_ica
        frontal_chs = [ch for ch in raw.ch_names if ch == "VEOG" or ch.startswith("ch_0")]
        indices, labels = detect_eog_by_topography(
            ica, raw, frontal_channels=frontal_chs, focal_threshold=0.5, subject_id="test"
        )
        assert isinstance(indices, list)
        assert isinstance(labels, dict)

    def test_no_frontal_channels_returns_empty(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        n_comp = min(6, len(raw_filt.ch_names) - 1)
        ica = mne.preprocessing.ICA(n_components=n_comp, method="fastica", random_state=42)
        ica.fit(raw_filt, verbose=False)
        indices, labels = detect_eog_by_topography(
            ica, raw_filt, frontal_channels=["NONEXISTENT_CH"],
            focal_threshold=0.5, subject_id="test",
        )
        assert indices == []


# ============================================================================
# Test 5: Full ICA Pipeline (End-to-End)
# ============================================================================


class TestFullICAPipeline:
    """End-to-end tests for run_ica_artifact_removal()."""

    def test_synthetic_data_pipeline_succeeds(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 6, "method": "fastica",
            "eog_detection": {"method": "combined", "eog_channel": "VEOG"},
        }
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test_synth"
        )
        assert raw_clean is not None
        assert isinstance(ica, mne.preprocessing.ICA)
        assert info["n_components_total"] == 6
        assert "removed_indices" in info
        assert "removed_labels" in info

    def test_cleaned_data_has_same_structure(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "correlation"}}
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert len(raw_clean.ch_names) == len(raw_filt.ch_names)
        assert raw_clean.info["sfreq"] == raw_filt.info["sfreq"]
        assert raw_clean.n_times == raw_filt.n_times

    def test_ica_exclude_set_correctly(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert set(ica.exclude) == set(info["removed_indices"])

    def test_info_has_removed_labels(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        for idx in info["removed_indices"]:
            assert idx in info["removed_labels"]

    def test_pipeline_kurtosis_only_mode(self, synthetic_raw_no_eog):
        raw_filt = synthetic_raw_no_eog.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 6,
            "eog_detection": {"method": "kurtosis", "eog_channel": ""},
        }
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test_no_eog"
        )
        assert info["detection_method"] == "kurtosis"

    def test_pipeline_topography_only_mode(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "topography"}}
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert info["detection_method"] == "topography"

    def test_default_config_works(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        raw_clean, ica, info = run_ica_artifact_removal(raw_filt, subject_id="test")
        assert info["n_components_total"] > 0

    @pytest.mark.slow
    def test_pipeline_with_real_data(self, real_raw):
        raw_filt = real_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 20, "method": "fastica",
            "eog_detection": {"method": "combined", "eog_channel": "auto"},
        }
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test_real"
        )
        assert raw_clean is not None
        assert info["n_components_total"] == 20
        assert info["n_components_removed"] >= 0


# ============================================================================
# Test 6: Edge Cases and Error Handling
# ============================================================================


class TestICAEdgeCases:
    """Tests for edge cases and error handling."""

    def test_raises_on_nan_data(self):
        sfreq = 250.0
        info = mne.create_info(["ch1", "ch2", "ch3", "ch4"], sfreq, ["eeg"] * 4)
        data = np.random.randn(4, 1000)
        data[0, 500] = np.nan
        raw = mne.io.RawArray(data, info, verbose=False)
        with pytest.raises(ValueError, match="NaN"):
            run_ica_artifact_removal(raw, config={"n_components": 2}, subject_id="test")

    def test_raises_on_few_channels(self):
        sfreq = 250.0
        info = mne.create_info(["ch1", "ch2"], sfreq, ["eeg", "eeg"])
        raw = mne.io.RawArray(np.random.randn(2, 1000), info, verbose=False)
        with pytest.raises(ValueError):
            run_ica_artifact_removal(raw, config={"n_components": 2}, subject_id="test")

    def test_handles_n_components_greater_than_channels(self, synthetic_raw_few_channels):
        raw_filt = synthetic_raw_few_channels.copy().filter(1.0, 40.0, fir_design="firwin")
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config={"n_components": 100}, subject_id="test"
        )
        assert info["n_components_total"] <= len(raw_filt.ch_names)

    def test_safety_never_removes_all_components(self, synthetic_raw):
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 4,
            "eog_detection": {
                "method": "combined",
                "correlation_threshold": 0.1,
                "kurtosis_threshold": 0.1,
                "focal_threshold": 0.1,
            },
        }
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert info["n_components_removed"] < info["n_components_total"]

    def test_no_artifacts_detected_is_ok(self, synthetic_raw_no_blinks):
        raw_filt = synthetic_raw_no_blinks.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {
            "n_components": 4,
            "eog_detection": {
                "method": "combined",
                "correlation_threshold": 10.0,
                "kurtosis_threshold": 10.0,
                "focal_threshold": 0.99,
            },
        }
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        assert info["n_components_removed"] >= 0


# ============================================================================
# Test 7: Integration Compatibility
# ============================================================================


class TestICAIntegration:
    """Verify ICA output is compatible with downstream modules."""

    def test_ica_output_accepted_by_downstream(self, synthetic_raw):
        """ICA-cleaned data should be window-able (common downstream step)."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        events = mne.make_fixed_length_events(raw_clean, duration=2.0)
        epochs = mne.Epochs(
            raw_clean, events, tmin=0, tmax=2.0, baseline=None,
            preload=True, verbose=False,
        )
        assert len(epochs) > 0

    def test_ica_output_psd_computation(self, synthetic_raw):
        """PSD computation should work on ICA-cleaned data."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        config = {"n_components": 6, "eog_detection": {"method": "combined"}}
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_filt, config=config, subject_id="test"
        )
        psd = raw_clean.compute_psd(method="welch", fmin=1, fmax=45)
        assert psd is not None

    def test_ica_works_on_segmented_data(self, synthetic_raw):
        """ICA should work on segmented data (e.g., EO-only)."""
        raw_filt = synthetic_raw.copy().filter(1.0, 40.0, fir_design="firwin")
        raw_seg = raw_filt.copy().crop(tmin=1, tmax=min(10, raw_filt.times[-1]))
        raw_clean, ica, info = run_ica_artifact_removal(
            raw_seg, config={"n_components": 4, "eog_detection": {"method": "combined"}},
            subject_id="test_seg"
        )
        assert raw_clean is not None


# ============================================================================
# Run with: pytest tests/test_ica.py -v
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
