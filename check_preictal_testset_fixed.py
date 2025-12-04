import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

from net import DL_config as dlc
from net.key_generator import generate_data_keys_sequential
from classes.data import Data
from net.utils import apply_preprocess_eeg
from scipy import signal as scipy_signal



# Robustly get Config class
Config = getattr(dlc, 'Config', getattr(dlc, 'DL_Config', None))
if Config is None:
    raise ImportError('Could not find Config or DL_Config in net/DL_config.py')


# Plotting EEG + ECG + events + pre-ictal window
def plot_eeg_with_events_and_preictal(config, rec, events_root, out_dir, preictal_sec = 30.0,):

    # Unpack recording
    subj, run = rec
    print(f'Plotting EEG + ECG + events + pre-ictal for {subj} {run}')

    # Load EDF via your Data class, both EEG and ECG
    raw = Data.loadData(config.data_path, rec, modalities=['eeg', 'ecg'])
    if raw is None or len(getattr(raw, 'data', [])) == 0:
        # No data = cannot plot anything
        print(f'No data for {subj} {run}, skipping plot.')
        return

    # Preprocess EEG into focal + cross channels, but don't require EEG to plot
    ch_focal = None
    ch_cross = None
    has_eeg = False
    try:
        ch_focal, ch_cross = apply_preprocess_eeg(config, raw)
        has_eeg = True
    except Exception as e:
        print(f'Could not preprocess EEG for {subj} {run}: {e}')
        print('Continuing with ECG/HRV-only plot if available')
        has_eeg = False

    # Extract ECG if available
    has_ecg = False
    ch_ecg = None
    channels = list(raw.channels)
    print(f'    Available channels: {channels}')
    for i, ch_name in enumerate(channels):
        if 'ecg' in ch_name.lower() or 'ekg' in ch_name.lower():
            ch_ecg = raw.data[i]
            has_ecg = True
            print(f'Found ECG channel: {ch_name}')
            break

    # If no ECG found by name, check if we have more channels beyond EEG
    if not has_ecg and len(raw.data) > 2:
        # Assume third channel might be ECG
        ch_ecg = raw.data[2]
        has_ecg = True
        print(f'Assuming channel {channels[2] if len(channels) > 2 else 'index 2'} is ECG')

    fs = config.fs

    # Determine a base time axis from whatever signal we have
    base_len = None
    if has_eeg and ch_focal is not None:
        base_len = len(ch_focal)
    elif len(getattr(raw, 'data', [])) > 0:
        # Fall back to the first available channel length
        base_len = len(raw.data[0])

    t = None
    if base_len is not None:
        t = np.arange(base_len) / fs

    # Prepare ECG / HRV if available
    hr_times = None
    hr_bpm = None
    t_ecg = None
    if has_ecg and ch_ecg is not None:
        # Resample ECG to match base_len if needed
        if t is not None and len(ch_ecg) != len(t):
            ch_ecg = scipy_signal.resample(ch_ecg, len(t))
        if t is not None:
            t_ecg = t
        else:
            t_ecg = np.arange(len(ch_ecg)) / fs

        # Compute simple HRV from ECG
        # Detect R-peaks with improved parameters
        try:
            # Normalize ECG for better peak detection
            ecg_normalized = (ch_ecg - np.mean(ch_ecg)) / (np.std(ch_ecg) + 1e-8)
            # Use height threshold and minimum distance between peaks
            peaks, properties = scipy_signal.find_peaks(
                ecg_normalized,
                # min 300ms between peaks (max 200 bpm)
                distance = int(0.3 * fs), 
                # peaks above 1 std 
                height = 1.0,  
                # require clear peaks
                prominence = 0.5)
            
            print(f'Detected {len(peaks)} R-peaks in ECG')
        except Exception as e:
            print(f'Peak detection failed: {e}')
            peaks = np.array([])

        # Compute HRV from R-R intervals
        if peaks is not None and len(peaks) > 1:
            peak_times = peaks / fs  # seconds
            rr_intervals = np.diff(peak_times)  # seconds
            # Filter out unrealistic RR intervals (HR between 30-200 bpm)
            valid_rr = (rr_intervals >= 0.3) & (rr_intervals <= 2.0)
            if np.sum(valid_rr) > 0:
                # Instantaneous heart rate (bpm)
                hr_bpm = 60.0 / rr_intervals[valid_rr]
                # Align HR values to the time of the second beat in each pair
                hr_times = peak_times[1:][valid_rr]
                print(f'Computed HRV: {len(hr_times)} valid intervals, HR range {hr_bpm.min():.1f}-{hr_bpm.max():.1f} bpm')
            else:
                print(f'No valid RR intervals found (all outside 30-200 bpm range)')
                hr_times = None
                hr_bpm = None
        else:
            # If we cannot get a reasonable HRV signal, we keep ECG but skip HRV
            print(f'Insufficient peaks for HRV computation')
            hr_times = None
            hr_bpm = None

    # Try to load events TSV
    events_tsv = os.path.join(events_root, subj, 'ses-01', 'eeg', f'{subj}_ses-01_task-szMonitoring_{run}_events.tsv',)

    events = []
    if os.path.exists(events_tsv):
        try:
            df_ev = pd.read_csv(events_tsv, sep = '\t')
            # Expecting columns 'onset' (s) and 'duration' (s)
            if 'onset' in df_ev.columns:
                for _, row in df_ev.iterrows():
                    onset = float(row['onset'])
                    dur = float(row['duration']) if 'duration' in df_ev.columns else 0.0
                    events.append((onset, onset + dur))
        except Exception as e:
            print(f'Could not read events for {subj} {run}: {e}')
    else:
        print(f'No events file found for {subj} {run}: {events_tsv}')

    # Build pre-ictal windows (30 s before event onset)
    preictal_windows = []
    for (st, en) in events:
        pre_st = max(0.0, st - preictal_sec)
        pre_en = st
        preictal_windows.append((pre_st, pre_en))

    # Start plotting
    os.makedirs(out_dir, exist_ok = True)

    # Decide what we can actually plot
    has_hrv = has_ecg and hr_times is not None and hr_bpm is not None

    if has_eeg and has_hrv:
        n_plots = 3
    elif has_eeg and not has_hrv and has_ecg:
        # EEG + raw ECG
        n_plots = 3
    elif has_eeg:
        n_plots = 2
    elif has_hrv or has_ecg:
        # Only ECG/HRV available
        n_plots = 1
    else:
        print(f'Neither EEG nor ECG/HRV usable for {subj} {run}, skipping plot.')
        return

    fig, axs = plt.subplots(n_plots, 1, figsize=(14, 8), sharex = True)
    if n_plots == 1:
        axs = [axs]

    fig.suptitle(f'{subj} {run} - EEG/HRV with events & {preictal_sec}s pre-ictal', fontsize = 28)

    # Focal and cross EEG channels (if available)
    ax_idx = 0
    if has_eeg and n_plots >= 2:
        axs[0].plot(t, ch_focal, linewidth = 0.4, label = 'EEG Signal')
        axs[0].set_ylabel('Focal EEG (µV)', fontsize = 20)
        axs[0].grid(True, alpha = 0.3)

        # Cross channel
        axs[1].plot(t, ch_cross, linewidth = 0.4, label = 'EEG Signal')
        axs[1].set_ylabel('Cross EEG (µV)', fontsize = 20)
        axs[1].grid(True, alpha = 0.3)
        ax_idx = 2
    elif has_eeg and n_plots == 2:
        # Just in case: treat it as one-EEG-channel layout
        axs[0].plot(t, ch_focal, linewidth = 0.4, label = 'EEG Signal')
        axs[0].set_ylabel('EEG (µV)', fontsize = 20)
        axs[0].grid(True, alpha = 0.3)
        ax_idx = 1
    else:
        ax_idx = 0

    # HRV/heart-rate or raw ECG channel when we have ECG but no EEG
    if has_ecg:
        if has_hrv and n_plots - 1 >= ax_idx:
            # Plot HRV when available
            axs[ax_idx].plot(hr_times, hr_bpm, linewidth = 0.6, label = 'HRV (bpm)')
            axs[ax_idx].set_ylabel('HR (bpm)', fontsize = 20)
            axs[ax_idx].set_ylim(80, 230)
        elif t_ecg is not None and ch_ecg is not None and n_plots - 1 >= ax_idx:
            # Fallback: plot raw ECG if HRV not usable
            axs[ax_idx].plot(t_ecg, ch_ecg, linewidth = 0.4, label = 'ECG Signal')
            axs[ax_idx].set_ylabel('ECG (mV)', fontsize = 20)
            axs[ax_idx].set_ylim(80, 230)

        axs[ax_idx].set_xlabel('Time (s)', fontsize = 22)
        axs[ax_idx].grid(True, alpha = 0.3)

    # Event boundaries
    for i, (st, en) in enumerate(events):
        for ax in axs:
            ax.axvline(st, color = 'r', linestyle = '--', linewidth = 1.0, alpha = 0.8,
                      label = 'Event Start' if i == 0 else '')
            ax.axvline(en, color = 'r', linestyle = ':', linewidth = 1.0, alpha = 0.6,
                      label = 'Event End' if i == 0 else '')

    # Pre-ictal windows (shaded)
    for i, (pre_st, pre_en) in enumerate(preictal_windows):
        for ax in axs:
            ax.axvspan(pre_st, pre_en, color = 'orange', alpha = 0.2, label = f'{preictal_sec}s Pre-ictal' if i == 0 else '')

    # Add legends to all subplots
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc = 'upper right', fontsize = 14)

    out_file = os.path.join(out_dir, f'{subj}_{run}_preictal.png')

    # Set x-limits based on whichever time axis we have
    x_end = None
    if t is not None and len(t) > 0:
        x_end = t[-1]
    elif hr_times is not None and len(hr_times) > 0:
        x_end = hr_times[-1]

    if x_end is not None:
        plt.xlim(0, x_end)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)

    print(f'Saved plot to {out_file}')


# Check testset + plot pre-ictal recordings
def main():
    config = Config()
    # Make sure this matches your project
    config.data_path = '/Users/rosalouisemarker/Desktop/Digital Media Project/dataset'
    config.fs = 250
    config.frame = 2
    config.stride = 1
    config.stride_s = 0.5
    config.boundary = 0.5
    config.sample_type = 'sequential'

    events_root = config.data_path
    out_dir = 'preictal_plots'

    print('=== Checking pre-ictal content in SZ2 TEST set ===')
    print(f'Data path: {config.data_path}')
    print(f'Events root: {events_root}')
    print(f'Output dir: {out_dir}\n')

    # Load test subject list
    test_tsv = os.path.join('net', 'datasets', 'SZ2_test.tsv')
    if not os.path.exists(test_tsv):
        raise FileNotFoundError(f'Test list not found: {test_tsv}')

    test_pats_list = (
        pd.read_csv(test_tsv, sep = '\t', header = None, skiprows = [0, 1, 2],)[0].to_list())

    # Build test recordings (subject + run)
    test_recs_list = []
    for s in test_pats_list:
        eeg_dir = os.path.join(config.data_path, s, 'ses-01', 'eeg')
        if not os.path.isdir(eeg_dir):
            continue
        for r in os.listdir(eeg_dir):
            if r.endswith('.edf'):
                run = r.split('_')[-2]
                test_recs_list.append([s, run])

    print(f'Found {len(test_recs_list)} test recordings.\n')

    # Generate test segments
    print('Generating test segments (this can take a while)')
    segments = generate_data_keys_sequential(config, test_recs_list, verbose = True)
    labels = np.array([seg[3] for seg in segments])

    print('=== LABEL DISTRIBUTION IN TEST SET (from key generator) ===')
    print(f'Total segments: {len(labels)}')
    print(f'Interictal (0): {np.sum(labels == 0)}')
    print(f'Pre-ictal (1): {np.sum(labels == 1)}')
    print(f'Ictal (2): {np.sum(labels == 2)}')
    print(f'\nAny pre-ictal? : {bool(np.any(labels == 1))}\n')

    # Identify recordings with pre-ictal segments
    print('=== RECORDINGS CONTAINING PRE-ICTAL SEGMENTS ===')
    recs_with_pre = set()
    for seg in segments:
        rec_idx, start, end, label = seg
        if label == 1:
            recs_with_pre.add(int(rec_idx))

    if not recs_with_pre:
        print('No pre-ictal recordings found in test set.')
        return

    for rec_idx in sorted(recs_with_pre):
        subj, run = test_recs_list[rec_idx]
        print(f' - {subj} {run}')

    print('\n=== Generating EEG + ECG + event + pre-ictal plots ===')
    for rec_idx in sorted(recs_with_pre):
        subj, run = test_recs_list[rec_idx]
        plot_eeg_with_events_and_preictal(config, [subj, run], events_root = events_root, out_dir=out_dir, preictal_sec=30.0,)

    print('\nDone.')


if __name__ == '__main__':
    main()
