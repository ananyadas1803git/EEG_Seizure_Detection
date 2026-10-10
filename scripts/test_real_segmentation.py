
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


import mne
import numpy as np

from src.eeg_seizure.segmentation import segment_eeg, label_windows

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit" / "chb01"

# Test the actual seizure-containing recording.
recording_path = DATA_DIR / "chb01_03.edf"

raw = mne.io.read_raw_edf(
    recording_path,
    preload=True,
    verbose="ERROR",
)

sampling_rate = int(raw.info["sfreq"])
window_seconds = 10

# Get the EEG array: channels x samples.
eeg = raw.get_data()

# Split the continuous EEG into windows.
windows = segment_eeg(
    eeg,
    sampling_rate=sampling_rate,
    window_seconds=window_seconds,
)

# Find the start time of every window in the recording.
window_starts = np.arange(len(windows)) * window_seconds

# Seizure interval from chb01-summary.txt.
seizure_intervals = [(2996, 3036)]

labels = label_windows(
    window_starts=window_starts,
    window_seconds=window_seconds,
    seizure_intervals=seizure_intervals,
)

print("REAL EEG SEGMENTATION RESULTS")
print("-" * 40)
print(f"Recording: {recording_path.name}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Original EEG shape: {eeg.shape}")
print(f"Segmented EEG shape: {windows.shape}")
print(f"Number of windows: {len(windows)}")
print(f"Seizure windows: {int(labels.sum())}")
print(f"Non-seizure windows: {int((labels == 0).sum())}")

positive_indices = np.flatnonzero(labels == 1)
print(f"\nSeizure-labeled windows (total {len(positive_indices)}):")
for i in positive_indices:
    start = window_starts[i]
    end = start + window_seconds
    print(f"  Window {i}: {start:.0f}-{end:.0f} seconds")

# A 40s seizure (2996s - 3036s) spans across 5 discrete 10s windows
# (2990-3000, 3000-3010, 3010-3020, 3020-3030, 3030-3040)
assert len(positive_indices) == 5, f"Expected 5 seizure windows, got {len(positive_indices)}"
print("\nValidation check passed: exactly 5 windows overlap the annotated seizure interval.")

