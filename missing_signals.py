import os
from pathlib import Path

data_path = Path("/Users/rosalouisemarker/Desktop/Digital Media Project/dataset")

total = 0
missing_eeg = 0
missing_ecg = 0
missing_both = 0

for subj in sorted(data_path.glob("sub-*")):
    eeg_files = list((subj / "ses-01" / "eeg").glob("*_eeg.edf"))
    ecg_files = list((subj / "ses-01" / "ecg").glob("*_ecg.edf"))

    # Count recordings (runs) by run number
    runs = set()
    for f in eeg_files + ecg_files:
        run = f.name.split("_")[3]   # "run-01"
        runs.add(run)

    for run in runs:
        total += 1

        has_eeg = any(run in f.name for f in eeg_files)
        has_ecg = any(run in f.name for f in ecg_files)

        if not has_eeg and not has_ecg:
            missing_both += 1
        elif not has_eeg:
            missing_eeg += 1
        elif not has_ecg:
            missing_ecg += 1

print("=== RESULTS ===")
print(f"Total recordings: {total}")
print("")
print(f"Missing EEG only: {missing_eeg}  ({100*missing_eeg/total:.2f}%)")
print(f"Missing ECG only: {missing_ecg}  ({100*missing_ecg/total:.2f}%)")
print(f"Missing BOTH:     {missing_both} ({100*missing_both/total:.2f}%)")
print("")
print(f"Valid EEG+ECG:    {total - (missing_eeg + missing_ecg + missing_both)} "
      f"({100*(total - (missing_eeg + missing_ecg + missing_both))/total:.2f}%)")