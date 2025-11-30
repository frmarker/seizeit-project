import os
import glob
import re

import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import pandas as pd

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Subject that actually exists and has EEG + ECG
SUBJECT_ID = "sub-054"   # change to sub-007, sub-053, sub-091, … if you like

# Compute DATA_PATH = "<project root>/dataset"
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(THIS_DIR)  # "Digital Media Project"
DATA_PATH  = os.path.join(ROOT_DIR, "dataset")

print(f"Using DATA_PATH = {DATA_PATH}")
print(f"Subject: {SUBJECT_ID}")


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def list_modality_files(subject: str, modality: str):
    """
    Return sorted list of EDF files for given subject + modality.
    modality in {'eeg', 'ecg'}
    """
    pattern = os.path.join(
        DATA_PATH,
        subject,
        "ses-01",
        modality,
        f"{subject}_ses-01_task-szMonitoring_run-*_{modality}.edf",
    )
    files = sorted(glob.glob(pattern))
    return files


def get_run_id(path: str):
    """
    Extract 'run-XX' string from filename.
    """
    base = os.path.basename(path)
    m = re.search(r"run-(\d+)_", base)
    if m:
        return f"run-{m.group(1)}"
    return None


def read_first_channel(edf_path: str):
    """
    Read first channel from an EDF file and its sampling frequency.
    """
    if not os.path.exists(edf_path):
        raise FileNotFoundError(edf_path)

    f = pyedflib.EdfReader(edf_path)
    try:
        print("\nCHANNELS IN:", edf_path)
        print(f.getSignalLabels())

        sig = f.readSignal(0)          # channel 0
        fs  = f.getSampleFrequency(0)  # sampling frequency of channel 0
    finally:
        f._close()
        del f
    return sig, fs


# -------------------------------------------------------------------
# Find EEG + ECG runs
# -------------------------------------------------------------------

eeg_files = list_modality_files(SUBJECT_ID, "eeg")
ecg_files = list_modality_files(SUBJECT_ID, "ecg")

print("EEG files found:")
for fpath in eeg_files:
    print("  ", fpath)

print("ECG files found:")
for fpath in ecg_files:
    print("  ", fpath)

# Map run-id -> file
eeg_by_run = {get_run_id(p): p for p in eeg_files}
ecg_by_run = {get_run_id(p): p for p in ecg_files}

runs_eeg = set(k for k in eeg_by_run.keys() if k is not None)
runs_ecg = set(k for k in ecg_by_run.keys() if k is not None)
runs_both = sorted(runs_eeg & runs_ecg)

print("\nRuns with BOTH EEG and ECG:")
print(" ", runs_both)

if not runs_both:
    raise RuntimeError(
        f"No runs with both EEG and ECG for {SUBJECT_ID}. "
        "Pick another SUBJECT_ID that you know has both modalities."
    )


# -------------------------------------------------------------------
# Load and concatenate signals
# -------------------------------------------------------------------

eeg_segments = []
ecg_segments = []
run_labels   = []
fs_eeg = None
fs_ecg = None

seizure_times = []  # NEW: global seizure onset times (seconds)
total_samples_so_far = 0  # NEW: to track where each run starts in the concatenated signal

for run in runs_both:
    eeg_path = eeg_by_run[run]
    ecg_path = ecg_by_run[run]

    print(f"\nLoading {run} EEG: {eeg_path}")
    try:
        sig_eeg, fs_e = read_first_channel(eeg_path)
    except FileNotFoundError:
        print(f"  [WARNING] Missing EEG file for {run}, skipping this run.")
        continue

    print(f"Loading {run} ECG: {ecg_path}")
    try:
        sig_ecg, fs_c = read_first_channel(ecg_path)
    except FileNotFoundError:
        print(f"  [WARNING] Missing ECG file for {run}, skipping this run.")
        continue

    # Sampling rate consistency checks (same as before)
    if fs_eeg is None:
        fs_eeg = fs_e
    elif fs_eeg != fs_e:
        print(f"Warning: EEG fs changed from {fs_eeg} to {fs_e} for {run}")

    if fs_ecg is None:
        fs_ecg = fs_c
    elif fs_ecg != fs_c:
        print(f"Warning: ECG fs changed from {fs_ecg} to {fs_c} for {run}")

    # --- NEW: figure out where this run starts on the global time axis ---
    # use EEG fs as the reference
    run_start_sec = total_samples_so_far / fs_eeg

    # --- NEW: read events.tsv and collect seizure onsets ---
    events_path = eeg_path.replace("_eeg.edf", "_events.tsv")
    if os.path.exists(events_path):
        try:
            events = pd.read_csv(events_path, sep="\t")
            # BIDS convention: 'onset' in seconds; seizure often in 'trial_type' or 'event'
            if "trial_type" in events.columns:
                mask = events["trial_type"].str.contains("seiz", case=False, na=False)
            elif "event" in events.columns:
                mask = events["event"].str.contains("seiz", case=False, na=False)
            else:
                mask = np.zeros(len(events), dtype=bool)

            for _, row in events[mask].iterrows():
                if "onset" in row:
                    onset_local = float(row["onset"])  # seconds from start of this run
                    seizure_times.append(run_start_sec + onset_local)
        except Exception as e:
            print(f"  [WARNING] Could not parse events for {run}: {e}")
    else:
        print(f"  [INFO] No events file for {run}: {events_path}")

    # --- existing code: store signals and move on ---
    eeg_segments.append(sig_eeg)
    ecg_segments.append(sig_ecg)
    run_labels.append(run)

    # NEW: update running total of samples (for the *next* run)
    total_samples_so_far += len(sig_eeg)

# For simplicity, assume same sampling rate across EEG runs & across ECG runs
fs = fs_eeg  # use EEG fs for time axis; typically same as ECG in this dataset

# Concatenate across runs
eeg_concat = np.concatenate(eeg_segments)
ecg_concat = np.concatenate(ecg_segments)

n_samples = len(eeg_concat)
time = np.arange(n_samples) / fs

# Compute run boundaries (in seconds)
lengths = [len(s) for s in eeg_segments]
boundaries_samples = np.cumsum(lengths)[:-1]
boundaries_sec = boundaries_samples / fs


# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------

fig, axes = plt.subplots(2, 1, sharex=True, figsize=(16, 8))

axes[0].plot(time, eeg_concat)
axes[0].set_ylabel("EEG (a.u.)")
axes[0].set_title(f"{SUBJECT_ID} – EEG across runs ({', '.join(run_labels)})")

axes[1].plot(time, ecg_concat)
axes[1].set_ylabel("ECG (a.u.)")
axes[1].set_xlabel("Time [s]")
axes[1].set_title(f"{SUBJECT_ID} – ECG across runs ({', '.join(run_labels)})")

# Vertical lines at run boundaries
# Vertical lines at run boundaries
for b in boundaries_sec:
    axes[0].axvline(b, linestyle="--", alpha=0.3)
    axes[1].axvline(b, linestyle="--", alpha=0.3)

# --- NEW: seizure annotations ---
if seizure_times:
    ylim_eeg = axes[0].get_ylim()
    ylim_ecg = axes[1].get_ylim()

    for t in seizure_times:
        axes[0].axvline(t, color="red", linewidth=1.2, alpha=0.8)
        axes[1].axvline(t, color="red", linewidth=1.2, alpha=0.8)

        # Small "SZ" label near the top of each subplot
        axes[0].text(t, 0.9 * ylim_eeg[1], "SZ", color="red",
                     rotation=90, ha="center", va="top", fontsize=8)
        axes[1].text(t, 0.9 * ylim_ecg[1], "SZ", color="red",
                     rotation=90, ha="center", va="top", fontsize=8)

plt.tight_layout()
plt.show()