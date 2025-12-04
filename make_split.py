import os
from glob import glob
import random

# Configuration
DATA_ROOT = '/Users/rosalouisemarker/Desktop/Digital Media Project/dataset'

# Fractions for the split 
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
RANDOM_SEED = 409


def subjects_with_eeg_ecg(root):
    # Find subjects with at least one run containing both EEG and ECG files
    subs = sorted([d for d in glob(os.path.join(root, 'sub-*')) if os.path.isdir(d)])
    good_subs = []

    # Check each subject
    for sub in subs:
        sessions = sorted(
            [s for s in glob(os.path.join(sub, 'ses-*')) if os.path.isdir(s)]
        )
        has_ok = False
        # Check each session
        for ses in sessions:
            eeg_dir = os.path.join(ses, 'eeg')
            ecg_dir = os.path.join(ses, 'ecg')
            # Find EEG files
            eeg_files = glob(os.path.join(eeg_dir, '*_eeg.edf'))
            if not eeg_files:
                continue
            # Check for corresponding ECG files
            for eeg_file in eeg_files:
                base = os.path.basename(eeg_file)
                ecg_file = os.path.join(ecg_dir, base.replace('_eeg.edf', '_ecg.edf'))
                if os.path.isfile(eeg_file) and os.path.isfile(ecg_file):
                    has_ok = True
                    break
            # If found in this session, no need to check further
            if has_ok:
                break
        # If at least one run with both EEG and ECG exists, add subject
        if has_ok:
            good_subs.append(os.path.basename(sub))

    return good_subs


def main():
    if not os.path.isdir(DATA_ROOT):
        print('DATA_ROOT does not exist:', DATA_ROOT)
        return

    # Get subjects with at least one EEG+ECG run
    subs = subjects_with_eeg_ecg(DATA_ROOT)
    print(f'Subjects with at least one EEG+ECG run: {len(subs)}')
    print(subs)

    random.seed(RANDOM_SEED)
    random.shuffle(subs)

    # Create splits
    n = len(subs)
    n_train = int(n * TRAIN_FRACTION)
    n_val = int(n * VAL_FRACTION)
    n_test = n - n_train - n_val

    train_subs = subs[:n_train]
    val_subs = subs[n_train:n_train + n_val]
    test_subs = subs[n_train + n_val:]

    print('\n=== Suggested split ===')
    print(f'Train: {len(train_subs)} subjects')
    print(f'Val: {len(val_subs)} subjects')
    print(f'Test: {len(test_subs)} subjects')

    print('\nPaste these into DL_config.py:\n')
    print('TRAIN_SUBJECTS =', train_subs)
    print('VAL_SUBJECTS =', val_subs)
    print('TEST_SUBJECTS =', test_subs)


if __name__ == '__main__':
    main()