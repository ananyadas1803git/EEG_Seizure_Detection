"""
EEG Seizure Detection: Leakage-Aware and Imbalance-Robust Deep Learning Framework.
"""

from src.eeg_seizure.segmentation import segment_eeg, label_windows

__all__ = [
    "segment_eeg",
    "label_windows",
]
