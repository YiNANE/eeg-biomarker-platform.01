"""
preprocess_lemon.py - Complete Preprocessing Pipeline for LEMON Dataset
Compatible with BrainVision format (.vhdr, .eeg, .vmrk)
Fixed internal reference errors in .vhdr files
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================
# 1. Set file paths - Specify three files separately
# ============================================
data_folder = r"C:\Users\ChanD\Downloads\brain model dataset"
subject = "sub-032301"  # ← Change to the subject you want to process

sub_folder = os.path.join(data_folder, subject, "RSEEG")

# Specify the three files separately
vhdr_file = os.path.join(sub_folder, f"{subject}.vhdr")
eeg_file = os.path.join(sub_folder, f"{subject}.eeg")
vmrk_file = os.path.join(sub_folder, f"{subject}.vmrk")

print(f"📂 Loading data: {vhdr_file}")

# Check if all three files exist
print("\n📋 Checking file existence:")
print(f"   .vhdr: {os.path.exists(vhdr_file)}")
print(f"   .eeg:  {os.path.exists(eeg_file)}")
print(f"   .vmrk: {os.path.exists(vmrk_file)}")

if not all(os.path.exists(f) for f in [vhdr_file, eeg_file, vmrk_file]):
    print("\n❌ Files missing! Please check:")
    if not os.path.exists(vhdr_file):
        print(f"   Missing .vhdr file: {vhdr_file}")
    if not os.path.exists(eeg_file):
        print(f"   Missing .eeg file: {eeg_file}")
    if not os.path.exists(vmrk_file):
        print(f"   Missing .vmrk file: {vmrk_file}")
    exit()

# ============================================
# 2. Load raw data - Use full parameters to avoid internal reference errors
# ============================================
print("\nLoading data...")

try:
    # Method 1: Direct read, ignoring internal reference errors
    raw = mne.io.read_raw_brainvision(
        vhdr_fname=vhdr_file,
        eog=(),
        misc=(),
        scale=1.0,
        preload=True,
        verbose=True
    )
    print("✅ Data loaded successfully (Method 1)")
except Exception as e:
    print(f"⚠️ Method 1 failed: {e}")
    print("Attempting Method 2...")
    
    # Method 2: Temporarily modify .vhdr file content before reading
    import tempfile
    import shutil
    
    # Read the original .vhdr file content
    with open(vhdr_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace incorrect filename references
    content = content.replace('sub-010002', subject)
    
    # Create a temporary .vhdr file
    temp_vhdr = tempfile.NamedTemporaryFile(mode='w', suffix='.vhdr', delete=False, encoding='utf-8')
    temp_vhdr.write(content)
    temp_vhdr.close()
    
    # Copy .eeg and .vmrk files to the temporary directory
    temp_dir = os.path.dirname(temp_vhdr.name)
    temp_eeg = os.path.join(temp_dir, f"{subject}.eeg")
    temp_vmrk = os.path.join(temp_dir, f"{subject}.vmrk")
    shutil.copy2(eeg_file, temp_eeg)
    shutil.copy2(vmrk_file, temp_vmrk)
    
    try:
        raw = mne.io.read_raw_brainvision(
            vhdr_fname=temp_vhdr.name,
            eog=(),
            misc=(),
            scale=1.0,
            preload=True,
            verbose=True
        )
        print("✅ Data loaded successfully (Method 2)")
        
        # Clean up temporary files (keep data)
        # os.unlink(temp_vhdr.name)
        # os.unlink(temp_eeg)
        # os.unlink(temp_vmrk)
        
    except Exception as e2:
        print(f"❌ Method 2 also failed: {e2}")
        print("Please manually open the .vhdr file in Notepad and replace 'sub-010002' with 'sub-032301'")
        exit()

# ============================================
# 3. Display data information
# ============================================
print(f"\n✅ Data loaded successfully!")
print(f"   Sampling rate: {raw.info['sfreq']} Hz")
print(f"   Number of channels: {len(raw.ch_names)}")
print(f"   Data duration: {raw.n_times / raw.info['sfreq']:.2f} seconds")
print(f"   First 10 channels: {raw.ch_names[:10]}")

# ============================================
# 4. View event annotations (EO/EC markers)
# ============================================
print("\n📋 Event annotations in data:")
print(raw.annotations)

try:
    events, event_id = mne.events_from_annotations(raw)
    print(f"\n📌 Event ID mapping: {event_id}")
    print(f"   Total events: {len(events)}")
except Exception as e:
    print(f"⚠️ Unable to extract events: {e}")

# ============================================
# 5. Filtering
# ============================================
print("\n🔧 Applying bandpass filter (1-45 Hz)...")
raw.filter(l_freq=1, h_freq=45, method='iir', iir_params={'order': 4, 'ftype': 'butter'})

print("🔧 Applying notch filter (50 Hz)...")
raw.notch_filter(freqs=[50], method='iir', iir_params={'order': 4, 'ftype': 'butter'})

# ============================================
# 6. Downsample to 250 Hz
# ============================================
print("🔧 Downsampling to 250 Hz...")
raw.resample(sfreq=250)

# ============================================
# 7. Set electrode positions
# ============================================
print("🔧 Setting standard electrode positions (10-20 system)...")
try:
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, match_case=False, on_missing='warn')
except Exception as e:
    print(f"   ⚠️ Failed to set electrode positions: {e}")

# ============================================
# 8. Re-reference to average reference
# ============================================
print("🔧 Setting to average reference...")
raw.set_eeg_reference('average', projection=False)

# ============================================
# 9. ICA for ocular artifact removal
# ============================================
print("\n🧠 Running ICA (this may take several minutes)...")

# Detect EOG channels
eog_channels = [ch for ch in raw.ch_names if 'EOG' in ch.upper() or 'HEOG' in ch.upper() or 'VEOG' in ch.upper()]
if not eog_channels:
    eog_channels = [ch for ch in raw.ch_names if 'FP' in ch.upper() or 'Fp' in ch]
    if not eog_channels:
        eog_channels = raw.ch_names[:2] if len(raw.ch_names) >= 2 else [raw.ch_names[0]]
    print(f"   Using these channels as EOG reference: {eog_channels}")
else:
    print(f"   Detected EOG channels: {eog_channels}")

try:
    n_comp = min(20, len(raw.ch_names) - 1)
    ica = mne.preprocessing.ICA(n_components=n_comp, method='picard', random_state=42, max_iter='auto')
    
    print(f"   Fitting ICA ({n_comp} components)...")
    ica.fit(raw)
    
    print("🔍 Automatically marking ocular components...")
    try:
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels, threshold=2.5)
        if len(eog_indices) > 0:
            print(f"   Detected ocular components: {eog_indices}")
            ica.exclude = eog_indices
            raw = ica.apply(raw)
            print("   ✅ Ocular components removed")
        else:
            eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels, threshold=1.5)
            if len(eog_indices) > 0:
                print(f"   Detected ocular components with looser threshold: {eog_indices}")
                ica.exclude = eog_indices
                raw = ica.apply(raw)
                print("   ✅ Ocular components removed")
            else:
                print("   ⚠️ No ocular components detected")
    except Exception as e:
        print(f"   ⚠️ Automatic marking failed: {e}")
except Exception as e:
    print(f"   ⚠️ ICA processing failed: {e}")

# ============================================
# 10. Save preprocessed data
# ============================================
output_path = os.path.join(sub_folder, f"{subject}_preprocessed.fif")
try:
    raw.save(output_path, overwrite=True)
    print(f"\n💾 Preprocessed data saved to: {output_path}")
except Exception as e:
    print(f"⚠️ Save failed: {e}")

# ============================================
# 11. Plot overview of processed data
# ============================================
print("\n📊 Plotting overview of preprocessed data...")
try:
    raw.plot(duration=10, n_channels=20, title="Preprocessed EEG Signal (first 10 seconds)")
except Exception as e:
    print(f"⚠️ Plotting failed: {e}")

print("\n🎉 Preprocessing complete!")

# Print final data information
print("\n📊 Final data information:")
print(raw)