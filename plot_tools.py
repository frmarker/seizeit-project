import numpy as np
import matplotlib.pyplot as plt
from classes.data import Data
from net.utils import apply_preprocess_eeg

def plot_recording_with_events(config, rec, segments, t_start=0, t_end=60):
 
    print(f'Loading and preprocessing {rec} ...')
    raw = Data.loadData(config.data_path, rec, modalities=['eeg', 'ecg'])
    # Gives 1, 2, or 3 channels
    chans = apply_preprocess_eeg(config, raw) 

    fs = config.fs
    s0 = int(t_start * fs)
    s1 = int(t_end * fs)

    plt.figure(figsize = (15, 6))

    # Offset channels for readability
    offset = 0
    for i, ch in enumerate(chans):
        plt.plot(np.arange(s0, s1) / fs, ch[s0:s1] + offset, label=f'Channel {i}')
        offset += np.max(np.abs(ch[s0:s1])) * 1.5

    # Draw event bars
    for seg in segments:
        _, seg_start, seg_end, lbl = seg
        if seg_start > t_end or seg_end < t_start:
            continue 

        color = {0: 'gray', 1: 'orange', 2: 'red'}[lbl]
        plt.axvspan(seg_start, seg_end, color=color, alpha=0.25)

    plt.title(f'{rec[0]} {rec[1]}  —  preprocessed EEG/HRV with events')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV or a.u.)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()