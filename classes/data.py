import os
import numpy as np
import warnings
import mne


class Data:
    """
    Simple container class for EEG/ECG data loaded from EDF.
    """

    def __init__(self, data=None, channels=None, fs=None):
        self.data = data if data is not None else []         # list[np.ndarray]
        self.channels = channels if channels is not None else ()  # tuple[str]
        self.fs = fs if fs is not None else ()               # tuple[float]

    @classmethod
    def empty(cls):
        """Returns a clean EMPTY Data object."""
        return cls(data=[], channels=(), fs=())

    @classmethod
    def loadData(cls, data_path, recording, modalities=("eeg",)):
        """
        Loads EEG/ECG data from EDF files with robust error checking.
        This function now:
            • enforces BIDS run + modality matching
            • detects broken symlinks
            • checks EDF readability using MNE
            • treats EEG as required and ECG as optional

        Args
        ----
        data_path : str
            Path to BIDS dataset root
        recording : tuple[str]
            (subject, run), e.g. ('sub-001', 'run-01')
        modalities : list[str] or tuple[str]
            Which modalities to load, e.g. ("eeg",) or ("eeg","ecg")

        Returns
        -------
        Data instance (possibly EMPTY)
        """
        subj, run = recording

        # Path where EDFs usually sit: <root>/sub-xxx/ses-01/eeg/
        eeg_dir = os.path.join(data_path, subj, "ses-01", "eeg")
        if not os.path.isdir(eeg_dir):
            warnings.warn(
                f"[Warning] {subj} {run}: EEG directory missing — creating EMPTY Data."
            )
            return cls.empty()

        # List all EDF candidates (EEG+ECG may be mixed here depending on dataset)
        try:
            candidates = [f for f in os.listdir(eeg_dir) if f.lower().endswith(".edf")]
        except FileNotFoundError:
            warnings.warn(
                f"[Warning] {subj} {run}: EEG directory missing — creating EMPTY Data."
            )
            return cls.empty()

        def match_edf(mod):
            """
            Find EDF file matching BOTH:
                - the run ID (e.g. 'run-01')
                - the modality token (e.g. '_eeg.' / '_ecg.')

            Returns:
                full_path or None, and optionally an error message
            """

            # Filenames must contain both the run number AND the modality
            edf_list = [
                f for f in candidates
                if (run in f) and (f"_{mod}." in f)
            ]

            if len(edf_list) == 0:
                return None, f"{mod.upper()} file missing."

            if len(edf_list) > 1:
                warnings.warn(
                    f"[Note] {subj} {run}: multiple {mod.upper()} EDFs found, "
                    f"using {edf_list[0]}"
                )

            path = os.path.join(eeg_dir, edf_list[0])

            # Symlink integrity check
            if os.path.islink(path):
                real = os.path.realpath(path)
                if not os.path.exists(real):
                    return None, (
                        f"{mod.upper()} EDF symlink broken: {path} → {real}"
                    )

            # Check readability with MNE (before loading big data)
            try:
                _ = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
            except Exception as e:
                return None, (
                    f"{mod.upper()} EDF unreadable: {path} (error: {e})"
                )

            return path, None

        # ==== REQUIRED: EEG ====
        eeg_file, eeg_err = match_edf("eeg")
        if eeg_file is None:
            warnings.warn(
                f"[Warning] {subj} {run}: {eeg_err} — creating EMPTY Data."
            )
            return cls.empty()

        # ==== OPTIONAL: ECG ====
        ecg_file, ecg_err = match_edf("ecg")
        if ecg_file is None:
            warnings.warn(
                f"[Note] {subj} {run}: {ecg_err} — continuing with EEG only.",
                stacklevel=2
            )

        # Begin loading data
        loaded_data = []
        loaded_channels = []
        loaded_fs = []

        def load_mod(path, mod_name):
            """Load EDF signals robustly using MNE."""
            try:
                raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
            except Exception as e:
                warnings.warn(
                    f"[Warning] {subj} {run}: failed reading {mod_name.upper()} "
                    f"('{path}') — {e}. Returning EMPTY Data."
                )
                return None

            # Extract signals, channels, and sampling rates
            ds = raw.get_data()  # shape (n_channels, n_samples)
            ch = raw.info["ch_names"]
            sf = [raw.info["sfreq"]] * len(ch)

            return ds, ch, sf

        # Load EEG
        res = load_mod(eeg_file, "eeg")
        if res is None:
            return cls.empty()
        data_eeg, channels_eeg, fs_eeg = res

        loaded_data.append(data_eeg)
        loaded_channels.extend(channels_eeg)
        loaded_fs.extend(fs_eeg)

        # Load ECG if available
        if ("ecg" in modalities) and (ecg_file is not None):
            res = load_mod(ecg_file, "ecg")
            if res is not None:
                data_ecg, channels_ecg, fs_ecg = res
                loaded_data.append(data_ecg)
                loaded_channels.extend(channels_ecg)
                loaded_fs.extend(fs_ecg)

        # Concatenate data arrays along channel axis
        if len(loaded_data) == 0:
            warnings.warn(
                f"[Warning] {subj} {run}: No data loaded — returning EMPTY Data."
            )
            return cls.empty()

        # Stack modalities along the channel dimension
        full_data = np.concatenate(loaded_data, axis=0)

        return cls(
            data=[d.astype(np.float32) for d in full_data],
            channels=tuple(loaded_channels),
            fs=tuple(loaded_fs),
        )