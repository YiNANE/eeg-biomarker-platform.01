"""Load raw EEG data for a subject from the LEMON dataset."""
import mne
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def load_subject(
    subject_id: str,
    lemon_root: str,
    session: str = "rest",
    resample_freq: float = 250.0,
) -> mne.io.Raw:
    """
    Load raw EEG for a given subject, with optional resampling.

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. "sub-0001".
    lemon_root : str
        Path to the LEMON dataset root.
    session : str
        Session label (default "rest").
    resample_freq : float, optional
        Target sampling frequency in Hz. If None, no resampling is applied.
        Default is 250.0 Hz (standardized frequency for the pipeline).

    Returns
    -------
    mne.io.Raw
        Raw EEG object (resampled if requested).

    Raises
    ------
    FileNotFoundError
        If the subject directory or EEG file does not exist.
    ValueError
        If the file format is unsupported or data is corrupted.
    RuntimeError
        If resampling fails.
    """
    root = Path(lemon_root)

    # --- Error 1: lemon_root does not exist ---
    if not root.exists():
        raise FileNotFoundError(
            f"LEMON root directory not found: {root}"
        )

    # LEMON BIDS structure: sub-<id>/ses-<session>/eeg/
    eeg_dir = root / subject_id / f"ses-{session}" / "eeg"

    # --- Error 2: subject directory does not exist ---
    if not eeg_dir.parent.exists():
        raise FileNotFoundError(
            f"Subject directory not found: {eeg_dir.parent}. "
            f"Available subjects: "
            f"{[d.name for d in root.iterdir() if d.is_dir() and d.name.startswith('sub-')]}"
        )

    # --- Error 3: session directory does not exist ---
    if not eeg_dir.exists():
        raise FileNotFoundError(
            f"Session directory not found: {eeg_dir}. "
            f"Available sessions: "
            f"{[d.name for d in (root / subject_id).iterdir() if d.is_dir()]}"
        )

    # Search for supported file formats
    candidates = (
        list(eeg_dir.glob("*.vhdr"))
        + list(eeg_dir.glob("*.edf"))
        + list(eeg_dir.glob("*.bdf"))
        + list(eeg_dir.glob("*.fif"))
    )

    # --- Error 4: no EEG file found ---
    if not candidates:
        existing_files = [f.name for f in eeg_dir.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"No supported EEG file found for {subject_id} in {eeg_dir}. "
            f"Supported formats: .vhdr, .edf, .bdf, .fif. "
            f"Existing files: {existing_files}"
        )

    # --- Error 5: check for companion file integrity (BrainVision) ---
    selected = candidates[0]
    if selected.suffix == ".vhdr":
        vmrk = selected.with_suffix(".vmrk")
        eeg = selected.with_suffix(".eeg")
        missing_companions = []
        if not vmrk.exists():
            missing_companions.append(str(vmrk.name))
        if not eeg.exists():
            missing_companions.append(str(eeg.name))
        if missing_companions:
            raise FileNotFoundError(
                f"BrainVision companion file(s) missing for {subject_id}: "
                f"{missing_companions}. "
                f"Expected: {selected.name}, {vmrk.name}, {eeg.name}"
            )

    # --- Attempt to load the file ---
    try:
        raw = mne.io.read_raw(str(selected), preload=True)
    except Exception as e:
        raise ValueError(
            f"Failed to read EEG file for {subject_id}: {selected.name}. "
            f"Error: {e}. "
            f"The file may be corrupted or in an unsupported format."
        ) from e

    # --- Error 6: validate basic data integrity after loading ---
    if raw.info["sfreq"] <= 0:
        raise ValueError(
            f"Invalid sampling frequency ({raw.info['sfreq']} Hz) for {subject_id}"
        )
    if len(raw.ch_names) == 0:
        raise ValueError(f"No channels found in data for {subject_id}")
    if raw.times[-1] <= 0:
        raise ValueError(f"Non-positive data duration ({raw.times[-1]:.1f}s) for {subject_id}")

    # --- Resample if requested ---
    if resample_freq is not None and raw.info["sfreq"] != resample_freq:
        if resample_freq <= 0:
            raise ValueError(
                f"Invalid target sampling frequency: {resample_freq} Hz"
            )
        if raw.info["sfreq"] < resample_freq:
            print(
                f"  [WARN] Upsampling {subject_id}: "
                f"{raw.info['sfreq']:.0f} Hz -> {resample_freq:.0f} Hz "
                f"(no new information created)"
            )
        print(
            f"  Resampling {subject_id}: "
            f"{raw.info['sfreq']:.0f} Hz -> {resample_freq:.0f} Hz"
        )
        try:
            raw.resample(resample_freq)
        except Exception as e:
            raise RuntimeError(
                f"Resampling failed for {subject_id}: {e}"
            ) from e

    return raw


def segment_by_state(
    raw: mne.io.Raw,
    eo_description: str = "Stimulus/S210",
    ec_description: str = "Stimulus/S200",
    switch_description: str = "Stimulus/S  1",
) -> Tuple[mne.io.Raw, mne.io.Raw]:
    """
    Segment raw EEG data into Eyes Open (EO) and Eyes Closed (EC) states
    based on S1 (switch) markers.

    LEMON dataset structure:
      - S1 markers define block boundaries (alternating EO/EC, ~62s each)
      - Within each block, S210 (EO) or S200 (EC) visual stimuli occur every 2s
      - S1 markers often coincide with the first S210/S200 of a block
      - Total: 8 EO blocks + 8 EC blocks, alternating

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data with annotations (must be preloaded).
    eo_description : str
        Annotation description for Eyes Open (default "Stimulus/S210").
    ec_description : str
        Annotation description for Eyes Closed (default "Stimulus/S200").
    switch_description : str
        Annotation description for state transition (default "Stimulus/S  1").

    Returns
    -------
    Tuple[mne.io.Raw, mne.io.Raw]
        (raw_eo, raw_ec) — segmented raw objects for each state.
    """
    if len(raw.annotations) == 0:
        raise ValueError("No annotations found in raw data.")

    # Collect all annotation types
    switch_onsets = []
    eo_onsets = []
    ec_onsets = []

    for ann in raw.annotations:
        desc = ann["description"]
        onset = ann["onset"]
        if desc == switch_description:
            switch_onsets.append(onset)
        elif desc == eo_description:
            eo_onsets.append(onset)
        elif desc == ec_description:
            ec_onsets.append(onset)

    if not eo_onsets:
        raise ValueError(
            f"Eyes Open annotation '{eo_description}' not found. "
            f"Available: {set(a['description'] for a in raw.annotations)}"
        )
    if not ec_onsets:
        raise ValueError(
            f"Eyes Closed annotation '{ec_description}' not found. "
            f"Available: {set(a['description'] for a in raw.annotations)}"
        )
    if not switch_onsets:
        raise ValueError(
            f"Switch marker '{switch_description}' not found. "
            f"Cannot determine block boundaries."
        )

    # Build block boundaries from S1 markers
    # S1 markers define the transitions between EO and EC blocks
    # The first S1 at ~4s marks the start of the experiment
    # Subsequent S1 markers define block boundaries
    boundaries = sorted(set(switch_onsets + [0.0, raw.times[-1]]))

    # Determine state for each block by counting S210 vs S200 stimuli
    eo_blocks = []
    ec_blocks = []

    for i in range(len(boundaries) - 1):
        t_start = boundaries[i]
        t_end = boundaries[i + 1]

        # Count stimuli in this block (excluding the boundary S1 itself)
        block_eo = sum(
            1 for o in eo_onsets if t_start < o < t_end
        )
        block_ec = sum(
            1 for o in ec_onsets if t_start < o < t_end
        )

        if block_eo > block_ec:
            eo_blocks.append((t_start, t_end))
        elif block_ec > block_eo:
            ec_blocks.append((t_start, t_end))
        # If equal (both 0), skip — this is the initial setup period

    if not eo_blocks:
        raise ValueError(
            "No EO blocks could be identified. "
            f"Boundaries: {boundaries}"
        )
    if not ec_blocks:
        raise ValueError(
            "No EC blocks could be identified. "
            f"Boundaries: {boundaries}"
        )

    def _extract_blocks(raw, blocks):
        """Extract and concatenate data for a list of (start, end) blocks."""
        sfreq = raw.info["sfreq"]
        data_parts = []
        for t_start, t_end in blocks:
            s = int(round(t_start * sfreq))
            e = int(round(t_end * sfreq))
            data_parts.append(raw.get_data()[:, s:e])
        return np.concatenate(data_parts, axis=1)

    # Extract EO and EC data
    eo_data = _extract_blocks(raw, eo_blocks)
    ec_data = _extract_blocks(raw, ec_blocks)

    # Create new Raw objects
    info = raw.info.copy()
    raw_eo = mne.io.RawArray(eo_data, info)
    raw_ec = mne.io.RawArray(ec_data, info)

    # Add state annotations
    raw_eo.set_annotations(
        mne.Annotations(onset=[0], duration=[raw_eo.times[-1]], description=["EO"])
    )
    raw_ec.set_annotations(
        mne.Annotations(onset=[0], duration=[raw_ec.times[-1]], description=["EC"])
    )

    print(
        f"  EO blocks: {len(eo_blocks)}, "
        f"total {raw_eo.times[-1]:.1f}s"
    )
    print(
        f"  EC blocks: {len(ec_blocks)}, "
        f"total {raw_ec.times[-1]:.1f}s"
    )

    return raw_eo, raw_ec


def load_batch(
    subject_ids,
    lemon_root: str,
    session: str = "rest",
    resample_freq: float = 250.0,
    verbose: bool = True,
):
    """
    Batch load multiple subjects from the LEMON dataset.

    Parameters
    ----------
    subject_ids : list of str
        List of subject identifiers, e.g. ["sub-032301", "sub-032302"].
    lemon_root : str
        Path to the LEMON dataset root.
    session : str
        Session label (default "rest").
    resample_freq : float, optional
        Target sampling frequency in Hz. If None, no resampling.
        Default is 250.0 Hz.
    verbose : bool
        Whether to print progress messages (default True).

    Returns
    -------
    dict
        {subject_id: mne.io.Raw} mapping. Failed subjects are omitted.
    """
    result = {}
    errors = {}
    for sid in subject_ids:
        try:
            raw = load_subject(
                sid,
                lemon_root=lemon_root,
                session=session,
                resample_freq=resample_freq,
            )
            result[sid] = raw
            if verbose:
                print(
                    f"  [OK] {sid}: {raw.info['sfreq']:.0f} Hz, "
                    f"{raw.times[-1]:.1f}s, {len(raw.ch_names)} ch"
                )
        except Exception as e:
            errors[sid] = str(e)
            if verbose:
                print(f"  [FAIL] {sid}: {e}")

    if verbose:
        print(f"\nBatch load summary: {len(result)}/{len(subject_ids)} succeeded")
        if errors:
            print(f"  Failed: {list(errors.keys())}")

    return result


def validate_raw(
    raw: mne.io.Raw,
    subject_id: str = "",
    expected_sfreq: float = 250.0,
    expected_n_channels: int = 62,
    min_duration: float = 60.0,
    max_duration: float = 2000.0,
    raise_on_fail: bool = False,
) -> dict:
    """
    Validate a loaded Raw object for data quality and consistency.

    Checks performed:
      - Sampling frequency matches expected value (within 1% tolerance)
      - Number of channels matches expected count
      - Data duration is within acceptable range
      - Data contains no NaN or Inf values
      - Data has non-zero variance (not a flat signal)

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG object to validate.
    subject_id : str
        Subject identifier for error messages (optional).
    expected_sfreq : float
        Expected sampling frequency in Hz (default 250.0).
    expected_n_channels : int
        Expected number of EEG channels (default 62).
    min_duration : float
        Minimum acceptable duration in seconds (default 60.0).
    max_duration : float
        Maximum acceptable duration in seconds (default 2000.0).
    raise_on_fail : bool
        If True, raise ValueError on first failure. If False, collect
        all issues and return them (default False).

    Returns
    -------
    dict
        Validation report with keys:
          - "passed": bool — True if all checks passed
          - "subject": str — subject identifier
          - "checks": dict — individual check results
          - "issues": list of str — descriptions of any failures
    """
    label = subject_id or "unknown"
    issues = []
    checks = {}

    # --- Check 1: Sampling frequency ---
    actual_sfreq = raw.info["sfreq"]
    sfreq_ok = abs(actual_sfreq - expected_sfreq) / expected_sfreq < 0.01
    checks["sfreq"] = {
        "passed": sfreq_ok,
        "expected": expected_sfreq,
        "actual": actual_sfreq,
    }
    if not sfreq_ok:
        msg = (
            f"[{label}] Sampling frequency mismatch: "
            f"expected {expected_sfreq:.0f} Hz, got {actual_sfreq:.0f} Hz"
        )
        issues.append(msg)
        if raise_on_fail:
            raise ValueError(msg)

    # --- Check 2: Number of channels ---
    actual_n_ch = len(raw.ch_names)
    ch_ok = actual_n_ch == expected_n_channels
    checks["n_channels"] = {
        "passed": ch_ok,
        "expected": expected_n_channels,
        "actual": actual_n_ch,
    }
    if not ch_ok:
        msg = (
            f"[{label}] Channel count mismatch: "
            f"expected {expected_n_channels}, got {actual_n_ch}"
        )
        issues.append(msg)
        if raise_on_fail:
            raise ValueError(msg)

    # --- Check 3: Data duration ---
    duration = raw.times[-1]
    duration_ok = min_duration <= duration <= max_duration
    checks["duration"] = {
        "passed": duration_ok,
        "expected_range": f"[{min_duration}, {max_duration}] s",
        "actual": f"{duration:.1f} s",
    }
    if not duration_ok:
        msg = (
            f"[{label}] Duration out of range: "
            f"{duration:.1f}s (expected {min_duration}-{max_duration}s)"
        )
        issues.append(msg)
        if raise_on_fail:
            raise ValueError(msg)

    # --- Check 4: NaN / Inf values ---
    data = raw.get_data()
    has_nan = np.any(np.isnan(data))
    has_inf = np.any(np.isinf(data))
    data_clean = not (has_nan or has_inf)
    checks["data_clean"] = {
        "passed": data_clean,
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
    }
    if not data_clean:
        problems = []
        if has_nan:
            problems.append("NaN")
        if has_inf:
            problems.append("Inf")
        msg = f"[{label}] Data contains {' and '.join(problems)} values"
        issues.append(msg)
        if raise_on_fail:
            raise ValueError(msg)

    # --- Check 5: Non-zero variance (flat signal detection) ---
    channel_var = np.var(data, axis=1)
    flat_channels = np.where(channel_var < 1e-12)[0]
    var_ok = len(flat_channels) == 0
    checks["variance"] = {
        "passed": var_ok,
        "flat_channels": flat_channels.tolist(),
        "n_flat": len(flat_channels),
    }
    if not var_ok:
        msg = (
            f"[{label}] {len(flat_channels)} flat channel(s) detected: "
            f"{flat_channels.tolist()}"
        )
        issues.append(msg)
        if raise_on_fail:
            raise ValueError(msg)

    passed = len(issues) == 0
    report = {
        "passed": passed,
        "subject": label,
        "checks": checks,
        "issues": issues,
    }

    if not passed:
        print(f"  [WARN] {label}: {len(issues)} validation issue(s)")
        for issue in issues:
            print(f"         {issue}")

    return report
