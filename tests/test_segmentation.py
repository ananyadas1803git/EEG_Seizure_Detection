
import numpy as np

from src.eeg_seizure.segmentation import segment_eeg


def test_segment_eeg_shape():#checking the number and dimension of window
    # Simulate one minute of 23-channel EEG data.
    eeg = np.zeros((23, 60 * 256))

    windows = segment_eeg(
        eeg,
        sampling_rate=256,
        window_seconds=10,
    )

    assert windows.shape == (6, 23, 2560)

#checks what happens when the recording duration isn't divisible by the window length.
def test_incomplete_window_is_discarded():
    # 25 seconds should produce two complete windows.
    eeg = np.zeros((2, 25 * 256))

    windows = segment_eeg(eeg)

    assert windows.shape == (2, 2, 2560)

#checks that the function rejects input with the wrong dimensions
def test_invalid_input_is_rejected():
    eeg = np.zeros(1000)

    try:
        segment_eeg(eeg)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")

from src.eeg_seizure.segmentation import label_windows

#checks whether labels match our expected results.
def test_label_windows_using_seizure_intervals():
    # Annotated seizure: 2996 to 3036 seconds.
    starts = np.array([2986, 2996, 3006, 3016, 3026, 3036])

    labels = label_windows(
        window_starts=starts,
        window_seconds=10,
        seizure_intervals=[(2996, 3036)],
    )

    expected = np.array([0, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(labels, expected)


def test_segment_eeg_with_stride():
    # 30 seconds of data, window 10s, stride 5s -> (30 - 10) / 5 + 1 = 5 windows
    eeg = np.zeros((19, 30 * 256))
    windows, starts = segment_eeg(
        eeg,
        sampling_rate=256,
        window_seconds=10,
        stride_seconds=5,
        return_starts=True,
    )
    assert windows.shape == (5, 19, 2560)
    np.testing.assert_array_equal(starts, np.array([0.0, 5.0, 10.0, 15.0, 20.0]))


def test_label_windows_with_min_overlap_ratio():
    # Seizure is 100 to 102 seconds (only 2 seconds)
    # Window 1: 95 to 105 (2 sec overlap / 10 sec = 0.2 ratio)
    starts = np.array([95.0])
    
    # With min_overlap_ratio=0.0 (any overlap), marked as seizure
    labels_any = label_windows(starts, window_seconds=10, seizure_intervals=[(100.0, 102.0)], min_overlap_ratio=0.0)
    assert labels_any[0] == 1

    # With min_overlap_ratio=0.5 (requires >= 50% overlap), marked as non-seizure
    labels_strict = label_windows(starts, window_seconds=10, seizure_intervals=[(100.0, 102.0)], min_overlap_ratio=0.5)
    assert labels_strict[0] == 0
