"""Unit tests for the ingestion module (loader.py)."""
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

from src.ingestion.loader import (
    load_subject,
    segment_by_state,
    load_batch,
    validate_raw,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def lemon_root():
    """Return the LEMON data root from paths config."""
    from src.utils.config import load_paths
    paths = load_paths(str(PROJECT_ROOT / "configs" / "paths.local.yaml"))
    return paths["lemon_root"]


@pytest.fixture(scope="session")
def real_raw(lemon_root):
    """Load a real subject for integration-level tests."""
    return load_subject("sub-032301", lemon_root=lemon_root, resample_freq=250.0)


@pytest.fixture
def synthetic_raw():
    """Create a synthetic Raw object with EO/EC annotations for unit tests."""
    sfreq = 250.0
    n_channels = 62
    duration = 200.0  # 200 seconds
    n_times = int(sfreq * duration)

    # Create random data
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, (n_channels, n_times))

    # Create info
    ch_names = [f"ch_{i:02d}" for i in range(n_channels)]
    ch_types = ["eeg"] * n_channels
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

    raw = mne.io.RawArray(data, info)

    # Add annotations simulating LEMON structure:
    #   - S1 at 0s (start)
    #   - EO block: S210 at 2s, 4s, 6s, ... (every 2s)
    #   - S1 at 62s (switch to EC)
    #   - EC block: S200 at 64s, 66s, 68s, ...
    #   - S1 at 124s (switch to EO)
    #   - EO block: S210 at 126s, 128s, ...
    #   - S1 at 186s (switch to EC)
    #   - EC block: S200 at 188s, 190s, ...
    onsets = []
    durations = []
    descriptions = []

    # Block 1: EO (0-62s)
    onsets.append(0.0)
    durations.append(0.0)
    descriptions.append("Stimulus/S  1")
    for t in range(2, 60, 2):
        onsets.append(float(t))
        durations.append(0.0)
        descriptions.append("Stimulus/S210")

    # Block 2: EC (62-124s)
    onsets.append(62.0)
    durations.append(0.0)
    descriptions.append("Stimulus/S  1")
    for t in range(64, 124, 2):
        onsets.append(float(t))
        durations.append(0.0)
        descriptions.append("Stimulus/S200")

    # Block 3: EO (124-186s)
    onsets.append(124.0)
    durations.append(0.0)
    descriptions.append("Stimulus/S  1")
    for t in range(126, 186, 2):
        onsets.append(float(t))
        durations.append(0.0)
        descriptions.append("Stimulus/S210")

    # Block 4: EC (186-200s)
    onsets.append(186.0)
    durations.append(0.0)
    descriptions.append("Stimulus/S  1")
    for t in range(188, 200, 2):
        onsets.append(float(t))
        durations.append(0.0)
        descriptions.append("Stimulus/S200")

    combined = mne.Annotations(onset=onsets, duration=durations, description=descriptions)
    raw.set_annotations(combined)

    return raw


# ============================================================================
# Tests: load_subject
# ============================================================================

class TestLoadSubject:
    """Tests for load_subject()."""

    def test_load_real_subject(self, real_raw):
        """Verify a real subject loads successfully."""
        assert real_raw is not None
        assert real_raw.info["sfreq"] == 250.0
        assert len(real_raw.ch_names) == 62
        assert real_raw.times[-1] > 900  # ~1022s

    def test_load_real_subject_no_resample(self, lemon_root):
        """Verify loading without resampling preserves original frequency."""
        raw = load_subject("sub-032301", lemon_root=lemon_root, resample_freq=None)
        assert raw.info["sfreq"] == 2500.0  # Original LEMON sampling rate

    def test_load_nonexistent_subject(self, lemon_root):
        """Verify FileNotFoundError for missing subject."""
        with pytest.raises(FileNotFoundError):
            load_subject("sub-999999", lemon_root=lemon_root)

    def test_load_nonexistent_root(self):
        """Verify FileNotFoundError for missing lemon_root."""
        with pytest.raises(FileNotFoundError):
            load_subject("sub-0001", lemon_root="/nonexistent/path")

    def test_load_invalid_resample_freq(self, lemon_root):
        """Verify ValueError for invalid resample frequency."""
        with pytest.raises(ValueError, match="Invalid target sampling frequency"):
            load_subject("sub-032301", lemon_root=lemon_root, resample_freq=-100)

    def test_load_custom_resample(self, lemon_root):
        """Verify custom resample frequency works."""
        raw = load_subject("sub-032301", lemon_root=lemon_root, resample_freq=128.0)
        assert raw.info["sfreq"] == 128.0

    def test_load_nonexistent_session(self, lemon_root):
        """Verify FileNotFoundError for missing session directory."""
        with pytest.raises(FileNotFoundError):
            load_subject("sub-032301", lemon_root=lemon_root, session="nonexistent")

    def test_load_upsampling_warning(self, lemon_root, capsys):
        """Verify upsampling produces a warning message."""
        # Load at 2500 Hz (original), then request upsampling to 5000 Hz
        raw = load_subject("sub-032301", lemon_root=lemon_root, resample_freq=5000.0)
        captured = capsys.readouterr()
        assert "Upsampling" in captured.out


# ============================================================================
# Tests: segment_by_state
# ============================================================================


class TestSegmentByState:
    """Tests for segment_by_state()."""

    def test_segment_synthetic(self, synthetic_raw):
        """Verify segmentation on synthetic data produces correct blocks."""
        raw_eo, raw_ec = segment_by_state(synthetic_raw)

        # EO: block 1 (0-62s) + block 3 (124-186s) = 124s
        # EC: block 2 (62-124s) + block 4 (186-200s) = 76s
        assert raw_eo.times[-1] == pytest.approx(124.0, abs=0.1)
        assert raw_ec.times[-1] == pytest.approx(76.0, abs=0.1)

    def test_segment_real_subject(self, real_raw):
        """Verify segmentation on real data produces EO and EC segments."""
        raw_eo, raw_ec = segment_by_state(real_raw)

        # Both states should have data
        assert raw_eo.times[-1] > 0
        assert raw_ec.times[-1] > 0

        # Total should be less than original (excluding transitions)
        total = raw_eo.times[-1] + raw_ec.times[-1]
        assert total < real_raw.times[-1]

        # Both should have 62 channels at 250 Hz
        assert len(raw_eo.ch_names) == 62
        assert len(raw_ec.ch_names) == 62
        assert raw_eo.info["sfreq"] == 250.0
        assert raw_ec.info["sfreq"] == 250.0

    def test_segment_no_annotations(self):
        """Verify ValueError when raw has no annotations."""
        sfreq = 250.0
        info = mne.create_info(["ch1"], sfreq, ["eeg"])
        raw = mne.io.RawArray(np.random.randn(1, 1000), info)
        with pytest.raises(ValueError, match="No annotations found"):
            segment_by_state(raw)

    def test_segment_missing_eo(self, synthetic_raw):
        """Verify ValueError when EO annotations are missing."""
        # Remove EO annotations
        bad_annot = mne.Annotations(
            onset=[0, 62],
            duration=[0, 0],
            description=["Stimulus/S  1", "Stimulus/S200"],
        )
        synthetic_raw.set_annotations(bad_annot)
        with pytest.raises(ValueError, match="Eyes Open annotation"):
            segment_by_state(synthetic_raw)

    def test_segment_missing_ec(self, synthetic_raw):
        """Verify ValueError when EC annotations are missing."""
        bad_annot = mne.Annotations(
            onset=[0, 62],
            duration=[0, 0],
            description=["Stimulus/S  1", "Stimulus/S210"],
        )
        synthetic_raw.set_annotations(bad_annot)
        with pytest.raises(ValueError, match="Eyes Closed annotation"):
            segment_by_state(synthetic_raw)

    def test_segment_missing_switch(self, synthetic_raw):
        """Verify ValueError when switch markers are missing."""
        bad_annot = mne.Annotations(
            onset=[0, 2, 4],
            duration=[0, 0, 0],
            description=["Stimulus/S210", "Stimulus/S210", "Stimulus/S200"],
        )
        synthetic_raw.set_annotations(bad_annot)
        with pytest.raises(ValueError, match="Switch marker"):
            segment_by_state(synthetic_raw)


# ============================================================================
# Tests: load_batch
# ============================================================================

class TestLoadBatch:
    """Tests for load_batch()."""

    def test_load_batch_single(self, lemon_root):
        """Verify batch loading a single subject."""
        result = load_batch(["sub-032301"], lemon_root=lemon_root, verbose=False)
        assert "sub-032301" in result
        assert result["sub-032301"].info["sfreq"] == 250.0

    def test_load_batch_multiple(self, lemon_root):
        """Verify batch loading multiple subjects."""
        result = load_batch(
            ["sub-032301", "sub-032302"],
            lemon_root=lemon_root,
            verbose=False,
        )
        assert len(result) == 2
        for sid in ["sub-032301", "sub-032302"]:
            assert sid in result

    def test_load_batch_with_failures(self, lemon_root):
        """Verify batch loading handles failures gracefully."""
        result = load_batch(
            ["sub-032301", "sub-999999"],
            lemon_root=lemon_root,
            verbose=False,
        )
        assert "sub-032301" in result
        assert "sub-999999" not in result

    def test_load_batch_empty_list(self, lemon_root):
        """Verify batch loading with empty list returns empty dict."""
        result = load_batch([], lemon_root=lemon_root, verbose=False)
        assert result == {}


# ============================================================================
# Tests: validate_raw
# ============================================================================

class TestValidateRaw:
    """Tests for validate_raw()."""

    def test_validate_passes(self, real_raw):
        """Verify validation passes for a real subject."""
        report = validate_raw(real_raw, subject_id="sub-032301")
        assert report["passed"] is True
        assert len(report["issues"]) == 0

    def test_validate_wrong_sfreq(self, real_raw):
        """Verify validation catches wrong sampling frequency."""
        report = validate_raw(real_raw, expected_sfreq=500.0)
        assert report["passed"] is False
        assert any("Sampling frequency mismatch" in i for i in report["issues"])

    def test_validate_wrong_channels(self, real_raw):
        """Verify validation catches wrong channel count."""
        report = validate_raw(real_raw, expected_n_channels=32)
        assert report["passed"] is False
        assert any("Channel count mismatch" in i for i in report["issues"])

    def test_validate_duration_too_short(self, real_raw):
        """Verify validation catches too-short duration."""
        report = validate_raw(real_raw, min_duration=99999)
        assert report["passed"] is False
        assert any("Duration out of range" in i for i in report["issues"])

    def test_validate_raise_on_fail(self, real_raw):
        """Verify raise_on_fail=True raises ValueError."""
        with pytest.raises(ValueError, match="Sampling frequency mismatch"):
            validate_raw(real_raw, expected_sfreq=500.0, raise_on_fail=True)

    def test_validate_nan_detection(self):
        """Verify validation detects NaN values."""
        sfreq = 250.0
        info = mne.create_info(["ch1"], sfreq, ["eeg"])
        data = np.random.randn(1, 1000)
        data[0, 500] = np.nan
        raw = mne.io.RawArray(data, info)
        report = validate_raw(raw)
        assert report["passed"] is False
        assert any("NaN" in i for i in report["issues"])

    def test_validate_flat_channel_detection(self):
        """Verify validation detects flat (zero-variance) channels."""
        sfreq = 250.0
        info = mne.create_info(["ch1", "ch2"], sfreq, ["eeg", "eeg"])
        data = np.zeros((2, 1000))
        data[1, :] = np.random.randn(1000)  # ch2 is normal
        raw = mne.io.RawArray(data, info)
        report = validate_raw(raw)
        assert report["passed"] is False
        assert any("flat channel" in i for i in report["issues"])

    def test_validate_report_structure(self, real_raw):
        """Verify validation report has correct structure."""
        report = validate_raw(real_raw)
        assert "passed" in report
        assert "subject" in report
        assert "checks" in report
        assert "issues" in report
        assert "sfreq" in report["checks"]
        assert "n_channels" in report["checks"]
        assert "duration" in report["checks"]
        assert "data_clean" in report["checks"]
        assert "variance" in report["checks"]


# ============================================================================
# Tests: Integration — Full pipeline compatibility
# ============================================================================

class TestPipelineIntegration:
    """Verify ingestion output is compatible with preprocessing pipeline."""

    def test_segment_output_is_raw(self, real_raw):
        """Verify segment_by_state returns mne.io.Raw objects."""
        raw_eo, raw_ec = segment_by_state(real_raw)
        # Check by class name to avoid import path issues
        assert type(raw_eo).__name__ == "RawArray", f"Expected RawArray, got {type(raw_eo).__name__}"
        assert type(raw_ec).__name__ == "RawArray", f"Expected RawArray, got {type(raw_ec).__name__}"
        # Verify they have Raw interface
        assert hasattr(raw_eo, "get_data")
        assert hasattr(raw_ec, "get_data")
        assert hasattr(raw_eo, "info")
        assert hasattr(raw_ec, "info")

    def test_segment_output_has_correct_sfreq(self, real_raw):
        """Verify segmented data has correct sampling rate."""
        raw_eo, raw_ec = segment_by_state(real_raw)
        assert raw_eo.info["sfreq"] == 250.0
        assert raw_ec.info["sfreq"] == 250.0

    def test_segment_output_has_annotations(self, real_raw):
        """Verify segmented data has state annotations."""
        raw_eo, raw_ec = segment_by_state(real_raw)
        assert len(raw_eo.annotations) > 0
        assert len(raw_ec.annotations) > 0
        assert raw_eo.annotations.description[0] == "EO"
        assert raw_ec.annotations.description[0] == "EC"

    def test_run_preprocessing_accepts_segmented_data(self, real_raw):
        """Verify run_preprocessing accepts segment_by_state output."""
        from src.preprocessing.pipeline import run_preprocessing
        from src.utils.config import load_config

        config = load_config(str(PROJECT_ROOT / "configs" / "preprocessing.yaml"))
        config["run_ica"] = False  # Skip ICA for speed
        config["overwrite"] = True

        raw_eo, raw_ec = segment_by_state(real_raw)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_preprocessing(
                raw_eo, config,
                output_dir=tmpdir,
                subject_id="test_EO",
            )
            assert "epochs" in result
            assert "qc" in result
            assert result["qc"]["n_windows"] > 0

    @pytest.mark.xfail(
        strict=False,
        reason="scipy eigh_tridiagonal crash on Windows (environment issue)",
    )
    def test_psd_computation_on_segmented_data(self, real_raw):
        """Verify PSD computation works on segmented + preprocessed data."""
        from src.preprocessing.pipeline import run_preprocessing
        from src.utils.config import load_config

        config = load_config(str(PROJECT_ROOT / "configs" / "preprocessing.yaml"))
        config["run_ica"] = False
        config["overwrite"] = True

        raw_eo, raw_ec = segment_by_state(real_raw)

        with tempfile.TemporaryDirectory() as tmpdir:
            result_eo = run_preprocessing(raw_eo, config, output_dir=tmpdir, subject_id="test_EO")
            result_ec = run_preprocessing(raw_ec, config, output_dir=tmpdir, subject_id="test_EC")

            psd_eo = result_eo["epochs"].compute_psd(fmin=1, fmax=45)
            psd_ec = result_ec["epochs"].compute_psd(fmin=1, fmax=45)

            avg_eo = psd_eo.average()
            avg_ec = psd_ec.average()

            # Verify PSD has expected shape
            assert avg_eo.data.shape[0] == 62  # channels
            assert avg_ec.data.shape[0] == 62
            assert len(avg_eo.freqs) > 0
            assert len(avg_ec.freqs) > 0


# ============================================================================
# Run with: pytest tests/test_ingestion.py -v
# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
