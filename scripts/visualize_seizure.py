
from pathlib import Path

import mne
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chbmit" / "chb01"

# Load the recording that contains an annotated seizure
recording_path = DATA_DIR / "chb01_03.edf"

raw = mne.io.read_raw_edf(
    recording_path,
    preload=False,
    verbose="ERROR",
)

# From chb01-summary.txt
seizure_start = 2996
seizure_end = 3036

# Select a 10-second window during the annotated seizure
plot_start = seizure_start + 10
plot_end = plot_start + 10

# Load only the selected time window into memory
eeg, times = raw.get_data(
    start=int(plot_start * raw.info["sfreq"]),
    stop=int(plot_end * raw.info["sfreq"]),
    return_times=True,
)

# Plot the first four channels
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(times, eeg[i] * 1e6, linewidth=0.6)
    ax.set_ylabel("µV")
    ax.set_title(raw.ch_names[i])
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")
fig.suptitle("CHB-MIT EEG During Annotated Seizure")
fig.tight_layout()

output_path = PROJECT_ROOT / "reports" / "figures" / "chb01_03_seizure_window.png"
fig.savefig(output_path, dpi=300)
print(f"Seizure plot saved to: {output_path}")

plt.show()
