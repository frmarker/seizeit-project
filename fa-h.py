import numpy as np

def compute_eventwise_fa_per_hour(yt, p_pre, th = 0.2, fs = 1.0, min_dur_sec = 5.0, refractory_sec = 60.0):

    # Convert inputs to numpy arrays
    yt = np.asarray(yt).astype(int)
    p_pre = np.asarray(p_pre).astype(float)

    # Binary prediction
    yp = (p_pre >= th).astype(int)

    n = len(yp)
    if n == 0:
        return np.nan

    # Find start/end indices of clusters of 1's
    # pad at ends to detect transitions
    padded = np.pad(yp, (1, 1), mode = 'constant', constant_values = 0)
    diff = np.diff(padded)

    starts = np.where(diff == 1)[0]       
    ends   = np.where(diff == -1)[0] - 1  

    # Filter events by duration and overlap with true pre-ictal
    assert len(starts) == len(ends)
    events = []
    for s, e in zip(starts, ends):
        dur_sec = (e - s + 1) / fs
        if dur_sec < min_dur_sec:
            # Too short, ignore
            continue  
        # Discard events that overlap true pre-ictal
        if np.any(yt[s:e+1] == 1):
            continue
        events.append([s, e])

    if not events:
        # No false alarms at all
        return 0.0  

    # Merge events closer than refractory_sec
    merged = []
    cur_s, cur_e = events[0]
    min_gap_samples = int(refractory_sec * fs)

    for s, e in events[1:]:
        gap = s - cur_e - 1
        if gap <= min_gap_samples:
            # merge into current cluster
            cur_e = e
        else:
            merged.append([cur_s, cur_e])
            cur_s, cur_e = s, e
    merged.append([cur_s, cur_e])

    n_fa_events = len(merged)

    # Interictal time in hours
    interictal_seconds = np.sum(yt == 0) / fs
    if interictal_seconds <= 0:
        return np.nan

    fa_per_hour = n_fa_events / (interictal_seconds / 3600.0)
    return fa_per_hour