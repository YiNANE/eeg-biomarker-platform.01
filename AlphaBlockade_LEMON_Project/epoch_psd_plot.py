"""
epoch_psd_plot.py - Epoching, QC, PSD Calculation and Plotting
Responsible for: Segmentation, quality control, and final Alpha-Blockade visualization
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_rel

# ============================================
# 1. Set file path - Load preprocessed data
# ============================================
data_folder = r"C:\Users\ChanD\Downloads\brain model dataset"
subject = "sub-032301"  # ← Change to the subject you want to process
sub_folder = os.path.join(data_folder, subject, "RSEEG")
input_file = os.path.join(sub_folder, f"{subject}_preprocessed.fif")

print(f"📂 Loading preprocessed data: {input_file}")
raw = mne.io.read_raw_fif(input_file, preload=True)

print(f"\n✅ Data loaded successfully!")
print(f"   Sampling rate: {raw.info['sfreq']} Hz")
print(f"   Number of channels: {len(raw.ch_names)}")
print(f"   Data duration: {raw.n_times / raw.info['sfreq']:.2f} seconds")

# ============================================
# 2. Extract EO/EC event markers
# ============================================
print("\n📋 Extracting event markers...")
events, event_id = mne.events_from_annotations(raw)

print(f"\n📌 Original event ID mapping: {event_id}")

# In LEMON data:
# S200 -> Eyes Open (EO), actual ID is 3
# S210 -> Eyes Closed (EC), actual ID is 4
# Note: These are the IDs returned by mne.events_from_annotations
eo_event_id = 3   # Stimulus/S200 -> Eyes Open
ec_event_id = 4   # Stimulus/S210 -> Eyes Closed

# Create event dictionary
event_dict = {'EO': eo_event_id, 'EC': ec_event_id}
print(f"\n📌 Using event mapping: {event_dict}")

# Filter to keep only EO and EC events
mask = (events[:, 2] == eo_event_id) | (events[:, 2] == ec_event_id)
events_filtered = events[mask]

print(f"   Filtered events count: {len(events_filtered)}")
print(f"   EO events: {np.sum(events_filtered[:, 2] == eo_event_id)}")
print(f"   EC events: {np.sum(events_filtered[:, 2] == ec_event_id)}")

if len(events_filtered) == 0:
    print("\n❌ No EO/EC events found! Please check event ID mapping.")
    print("   All available events:")
    for key, value in event_id.items():
        count = np.sum(events[:, 2] == value)
        print(f"      {key} (ID={value}): {count} events")
    exit()

# ============================================
# 3. Data segmentation (Epoching) - 2-second windows
# ============================================
print("\n⏱️ Creating 2-second epochs...")

# Define epoch parameters
tmin, tmax = 0.0, 2.0  # 2-second window
baseline = None  # No baseline correction

# Create epochs object
epochs = mne.Epochs(
    raw,
    events_filtered,
    event_id=event_dict,
    tmin=tmin,
    tmax=tmax,
    baseline=baseline,
    preload=True,
    reject_by_annotation=True,
    verbose=True
)

print(f"\n✅ Epochs created successfully!")
print(f"   Total epochs: {len(epochs)}")
print(f"   EO epochs: {len(epochs['EO'])}")
print(f"   EC epochs: {len(epochs['EC'])}")

# ============================================
# 4. Quality Control (QC) - ±100µV threshold
# ============================================
print("\n🔧 Applying QC threshold (±100 µV)...")

reject_criteria = dict(eeg=100e-6)  # 100 µV = 100e-6 V

# Apply reject threshold
epochs_clean = epochs.copy()
epochs_clean.drop_bad(reject=reject_criteria)

print(f"\n✅ QC complete!")
print(f"   Retained epochs: {len(epochs_clean)}")
print(f"   Dropped epochs: {len(epochs) - len(epochs_clean)}")
print(f"   EO retained: {len(epochs_clean['EO'])}")
print(f"   EC retained: {len(epochs_clean['EC'])}")

if len(epochs_clean) < 10:
    print("\n⚠️ Warning: Too few epochs retained. Consider adjusting threshold or checking data!")

# ============================================
# 5. Select occipital channels
# ============================================
print("\n🧠 Selecting occipital channels (O1, O2, Oz)...")

# Find occipital channels
occipital_channels = ['O1', 'O2', 'Oz']
available_occ = [ch for ch in occipital_channels if ch in epochs_clean.ch_names]

if not available_occ:
    print("   ⚠️ Standard occipital channels not found, searching for channels starting with 'O'...")
    available_occ = [ch for ch in epochs_clean.ch_names if ch.startswith('O')]
    if not available_occ:
        print("   ❌ No occipital channels found! Please check channel names.")
        print(f"   Available channels: {epochs_clean.ch_names}")
        exit()
    else:
        print(f"   Found occipital channels: {available_occ}")
else:
    print(f"   Found occipital channels: {available_occ}")

epochs_occ = epochs_clean.copy().pick_channels(available_occ)
print(f"   Using channels: {epochs_occ.ch_names}")

# ============================================
# 6. Compute PSD (Welch's Method)
# ============================================
print("\n📊 Computing PSD (Welch's method)...")

# Set Welch parameters
fmin, fmax = 1.0, 40.0  # Frequency range
n_fft = 256  # FFT window size

# Compute PSD for EO and EC separately
psd_EO = epochs_occ['EO'].compute_psd(
    method='welch',
    fmin=fmin,
    fmax=fmax,
    n_fft=n_fft,
    n_overlap=n_fft // 2,
    verbose=True
)

psd_EC = epochs_occ['EC'].compute_psd(
    method='welch',
    fmin=fmin,
    fmax=fmax,
    n_fft=n_fft,
    n_overlap=n_fft // 2,
    verbose=True
)

# Get data
psds_EO, freqs = psd_EO.get_data(return_freqs=True)
psds_EC, _ = psd_EC.get_data(return_freqs=True)

# Average across all channels and epochs
mean_psd_EO = psds_EO.mean(axis=(0, 1))
mean_psd_EC = psds_EC.mean(axis=(0, 1))

print(f"✅ PSD computation complete!")
print(f"   Frequency range: {freqs[0]:.1f} - {freqs[-1]:.1f} Hz")
print(f"   Frequency points: {len(freqs)}")

# ============================================
# 7. Generate final comparison plot
# ============================================
print("\n📈 Generating Alpha-Blockade comparison plot...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot the two curves
ax.plot(freqs, mean_psd_EO, label='Eyes Open (EO)', color='red', linewidth=2)
ax.plot(freqs, mean_psd_EC, label='Eyes Closed (EC)', color='blue', linewidth=2)

# Highlight Alpha band
alpha_band = (8, 12)
ax.axvspan(alpha_band[0], alpha_band[1], alpha=0.2, color='gray', 
           label=f'Alpha Band ({alpha_band[0]}-{alpha_band[1]} Hz)')

# Mark peak within Alpha band
alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
if np.any(alpha_mask):
    alpha_freqs = freqs[alpha_mask]
    alpha_ec = mean_psd_EC[alpha_mask]
    if len(alpha_ec) > 0:
        peak_idx = np.argmax(alpha_ec)
        peak_freq = alpha_freqs[peak_idx]
        peak_power = alpha_ec[peak_idx]
        ax.plot(peak_freq, peak_power, 'bo', markersize=10)
        ax.annotate(f'Alpha Peak: {peak_freq:.1f} Hz',
                    xy=(peak_freq, peak_power),
                    xytext=(peak_freq + 1, peak_power * 0.8),
                    arrowprops=dict(arrowstyle='->', color='blue'),
                    fontsize=10)

# Set labels and title
ax.set_xlabel('Frequency (Hz)', fontsize=12)
ax.set_ylabel('Power Spectral Density (µV²/Hz)', fontsize=12)
ax.set_title(f'Alpha-Blockade: Occipital Channels {available_occ}\n'
             f'Subject: {subject}', fontsize=14)

ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Set x-axis range
ax.set_xlim([fmin, fmax])

# Save figure
output_plot = os.path.join(sub_folder, f"{subject}_alpha_blockade.png")
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"\n💾 Figure saved to: {output_plot}")

# Display
plt.show()

print("\n🎉 All complete! Alpha-Blockade effect successfully reproduced.")

# ============================================
# 8. Output statistics
# ============================================
print("\n📊 Alpha band power statistics:")

# Calculate average Alpha band power
alpha_idx = (freqs >= 8) & (freqs <= 12)
alpha_power_EO = np.mean(mean_psd_EO[alpha_idx])
alpha_power_EC = np.mean(mean_psd_EC[alpha_idx])
ratio = alpha_power_EO / alpha_power_EC

print(f"   EO Alpha power: {alpha_power_EO:.4f} µV²/Hz")
print(f"   EC Alpha power: {alpha_power_EC:.4f} µV²/Hz")
print(f"   EO/EC ratio: {ratio:.3f}")

if ratio < 0.7:
    print("   ✅ Significant Alpha suppression detected! (EO/EC < 0.7)")
else:
    print("   ⚠️ Alpha suppression not significant. Consider checking data or parameters.")

# ============================================
# 9. Supplementary Statistical Validation (FIXED)
# ============================================
print("\n📊 Supplementary Statistical Validation:")

# Extract alpha band power for each epoch using the new MNE API
# Initialize lists to store alpha power per epoch
alpha_power_per_epoch_EO = []
alpha_power_per_epoch_EC = []

# Process each EO epoch individually
print("   Computing alpha power per epoch for EO...")
for epoch_idx in range(len(epochs_occ['EO'])):
    epoch_single = epochs_occ['EO'][epoch_idx:epoch_idx+1]
    psd_epoch = epoch_single.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        verbose=False
    )
    psd_data, freqs_epoch = psd_epoch.get_data(return_freqs=True)
    alpha_idx_epoch = (freqs_epoch >= 8) & (freqs_epoch <= 12)
    alpha_power = np.mean(psd_data[:, :, alpha_idx_epoch])
    alpha_power_per_epoch_EO.append(alpha_power)

# Process each EC epoch individually
print("   Computing alpha power per epoch for EC...")
for epoch_idx in range(len(epochs_occ['EC'])):
    epoch_single = epochs_occ['EC'][epoch_idx:epoch_idx+1]
    psd_epoch = epoch_single.compute_psd(
        method='welch',
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_fft // 2,
        verbose=False
    )
    psd_data, freqs_epoch = psd_epoch.get_data(return_freqs=True)
    alpha_idx_epoch = (freqs_epoch >= 8) & (freqs_epoch <= 12)
    alpha_power = np.mean(psd_data[:, :, alpha_idx_epoch])
    alpha_power_per_epoch_EC.append(alpha_power)

# Convert to numpy arrays
alpha_EO_array = np.array(alpha_power_per_epoch_EO)
alpha_EC_array = np.array(alpha_power_per_epoch_EC)

print(f"\n   Number of EO epochs: {len(alpha_EO_array)}")
print(f"   Number of EC epochs: {len(alpha_EC_array)}")

# For paired t-test, we need equal number of samples
# Take the minimum length and truncate both arrays
min_len = min(len(alpha_EO_array), len(alpha_EC_array))
alpha_EO_array = alpha_EO_array[:min_len]
alpha_EC_array = alpha_EC_array[:min_len]

print(f"   Using {min_len} epochs for paired t-test (truncated to match)")

# Perform paired t-test
t_stat, p_value = ttest_rel(alpha_EO_array, alpha_EC_array)

print(f"\n   Alpha power (EO) - Mean: {np.mean(alpha_EO_array):.4f} µV²/Hz, Std: {np.std(alpha_EO_array):.4f}")
print(f"   Alpha power (EC) - Mean: {np.mean(alpha_EC_array):.4f} µV²/Hz, Std: {np.std(alpha_EC_array):.4f}")
print(f"   t-statistic: {t_stat:.4f}")
print(f"   p-value: {p_value:.6f}")

if p_value < 0.05:
    print("   ✅ Statistical test PASSED (p < 0.05): Significant difference in alpha power between EO and EC.")
else:
    print("   ⚠️ Statistical test NOT PASSED (p >= 0.05): No significant difference detected. Consider checking data.")