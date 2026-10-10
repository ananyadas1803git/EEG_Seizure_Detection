
from pathlib import Path
import mne
import matplotlib.pyplot as plt

# Locate the project and EEG files
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit" / "chb01"

recording_path = DATA_DIR / "chb01_01.edf"
summary_path = DATA_DIR / "chb01-summary.txt"

# Check that the required files exist
if not recording_path.exists():
    raise FileNotFoundError(f"EEG recording not found: {recording_path}")

if not summary_path.exists():
    raise FileNotFoundError(f"Summary file not found: {summary_path}")

# Load EEG recording without loading all signal samples into RAM
raw = mne.io.read_raw_edf(recording_path, preload=False, verbose="ERROR")

print("EEG RECORDING INFORMATION")
print("-" * 40)
print(f"File: {recording_path.name}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Number of channels: {len(raw.ch_names)}")
print(f"Recording duration: {raw.times[-1] / 60:.2f} minutes")
print(f"Channel names: {raw.ch_names}")

# Display the first 20 seconds of EEG
raw.plot(
    duration=20,
    n_channels=min(10, len(raw.ch_names)),
    scalings="auto",
    title="CHB-MIT EEG: chb01_01.edf",
    show=True,
)

# Display the recording summary for later annotation analysis
print("\nRECORDING SUMMARY")
print("-" * 40)
print(summary_path.read_text(errors="replace"))
