import os
from glob import glob

# Path to your dataset
DATA_ROOT = '/Users/rosalouisemarker/Desktop/Digital Media Project/dataset'


def list_subjects(root):
    return sorted([d for d in glob(os.path.join(root, 'sub-*')) if os.path.isdir(d)])


def check_files(eeg_file, ecg_file):
    info = {
        'eeg_ok': False,
        'eeg_size': 0,
        'ecg_ok': False,
        'ecg_size': 0,}

    # Check EEG file
    if eeg_file and os.path.isfile(eeg_file):
        info['eeg_ok'] = True
        info['eeg_size'] = os.path.getsize(eeg_file)

    # Check ECG file
    if ecg_file and os.path.isfile(ecg_file):
        info['ecg_ok'] = True
        info['ecg_size'] = os.path.getsize(ecg_file)

    return info


def main(max_subjects=None):
    print(f'Scanning dataset: {DATA_ROOT}')

    if not os.path.isdir(DATA_ROOT):
        print('Dataset folder not found.')
        return

    subjects = list_subjects(DATA_ROOT)
    if max_subjects:
        subjects = subjects[:max_subjects]

    # Counters 
    total = 0
    eeg_missing = 0
    ecg_missing = 0
    ok = 0

    # Iterate over subjects and sessions
    for idx, sub in enumerate(subjects, start=1):
        sub_name = os.path.basename(sub)
        sessions = sorted([s for s in glob(os.path.join(sub, 'ses-*')) if os.path.isdir(s)])

        print(f'[{idx}/{len(subjects)}] {sub_name} — {len(sessions)} sessions')

        for ses in sessions:
            eeg_dir = os.path.join(ses, 'eeg')
            ecg_dir = os.path.join(ses, 'ecg')

            eeg_files = sorted(glob(os.path.join(eeg_dir, '*_eeg.edf')))

            for eeg_file in eeg_files:
                total += 1
                base = os.path.basename(eeg_file)

                if os.path.isdir(ecg_dir):
                    ecg_file = os.path.join(ecg_dir, base.replace('_eeg.edf', '_ecg.edf'))
                else:
                    ecg_file = None

                info = check_files(eeg_file, ecg_file)

                # Problems
                if not info['eeg_ok']:
                    eeg_missing += 1
                    print(f'Missing EEG: {eeg_file}')
                elif not info['ecg_ok']:
                    ecg_missing += 1
                    print(f'Missing ECG: {ecg_file}')
                else:
                    ok += 1

    print('\n=== SUMMARY ===')
    print(f'Total runs detected: {total}')
    print(f'OK (EEG+ECG present): {ok}')
    print(f'Missing EEG: {eeg_missing}')
    print(f'Missing ECG: {ecg_missing}')


if __name__ == '__main__':
    main(max_subjects = None)