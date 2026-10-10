import numpy as np


def segment_eeg(
    eeg: np.ndarray,
    sampling_rate: int = 256,
    window_seconds: int = 10,
    stride_seconds: int | float | None = None,
    return_starts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Divide continuous EEG data into fixed-duration windows (non-overlapping or sliding).

    Parameters
    ----------
    eeg : np.ndarray
        EEG data with shape (channels, samples).
    sampling_rate : int
        Number of samples per second.
    window_seconds : int
        Duration of each window in seconds.
    stride_seconds : int | float | None, optional
        Step size between consecutive windows in seconds.
        If None, defaults to window_seconds (non-overlapping).
    return_starts : bool, optional
        If True, also returns an array of window start times in seconds.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        Windows array with shape (windows, channels, samples_per_window).
        If return_starts is True, returns (windows, start_times).
    """
    if eeg.ndim != 2:
        raise ValueError("EEG must have shape (channels, samples).")

    if sampling_rate <= 0 or window_seconds <= 0:
        raise ValueError("Sampling rate and window duration must be positive.")

    if stride_seconds is not None and stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive.")

    samples_per_window = int(sampling_rate * window_seconds)
    step_samples = int(
        sampling_rate * (stride_seconds if stride_seconds is not None else window_seconds)
    )

    total_samples = eeg.shape[1]
    if total_samples < samples_per_window:
        empty_windows = np.empty((0, eeg.shape[0], samples_per_window), dtype=eeg.dtype)
        empty_starts = np.empty((0,), dtype=np.float64)
        return (empty_windows, empty_starts) if return_starts else empty_windows

    # If non-overlapping, use fast block reshaping
    if step_samples == samples_per_window:
        n_windows = total_samples // samples_per_window
        usable_samples = n_windows * samples_per_window
        windows = eeg[:, :usable_samples].reshape(
            eeg.shape[0],
            n_windows,
            samples_per_window,
        ).transpose(1, 0, 2)
        start_times = np.arange(n_windows, dtype=np.float64) * (step_samples / sampling_rate)
    else:
        # Sliding window with step_samples
        start_indices = np.arange(0, total_samples - samples_per_window + 1, step_samples)
        windows = np.lib.stride_tricks.sliding_window_view(
            eeg, samples_per_window, axis=1
        )[:, start_indices, :].transpose(1, 0, 2)
        start_times = start_indices.astype(np.float64) / sampling_rate

    if return_starts:
        return windows, start_times
    return windows


def label_windows(
    window_starts: np.ndarray,
    window_seconds: int | float,
    seizure_intervals: list[tuple[float, float]],
    min_overlap_ratio: float = 0.0,
) -> np.ndarray:
    """
    Label windows using annotated seizure intervals.

    Parameters
    ----------
    window_starts : np.ndarray
        Array of start times (in seconds) for each window.
    window_seconds : int or float
        Duration of each window in seconds.
    seizure_intervals : list[tuple[float, float]]
        List of (start, end) tuples indicating seizure periods in seconds.
    min_overlap_ratio : float, default=0.0
        Minimum fraction of the window that must overlap a seizure
        for it to be labeled as positive (1). If 0.0, any overlap > 0
        is marked as positive.

    Returns
    -------
    np.ndarray
        Binary labels (1 for seizure, 0 for non-seizure).
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    if not (0.0 <= min_overlap_ratio <= 1.0):
        raise ValueError("min_overlap_ratio must be between 0.0 and 1.0.")

    labels = np.zeros(len(window_starts), dtype=np.int64)

    for i, start in enumerate(window_starts):
        end = start + window_seconds

        for seizure_start, seizure_end in seizure_intervals:
            if seizure_end <= seizure_start:
                raise ValueError("Seizure intervals must have positive duration.")

            overlap = max(0.0, min(end, seizure_end) - max(start, seizure_start))
            threshold = min_overlap_ratio * window_seconds

            if min_overlap_ratio == 0.0:
                is_overlap = overlap > 0.0
            else:
                is_overlap = overlap >= threshold

            if is_overlap:
                labels[i] = 1
                break

    return labels
