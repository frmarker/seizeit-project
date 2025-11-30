import numpy as np


class Data:
    def __init__(self, data, channels, fs):
        self.data = data            # list[np.ndarray]
        self.channels = channels    # tuple[str]
        self.fs = fs                # tuple[float]

    @classmethod
    def loadData(
        cls,
        data_path,
        recording,
        modalities,
    ):
        """
        Instantiate a Data instance from EDF files.

        Args:
            data_path (str): base path to the BIDS dataset
            recording (tuple[str]): (subject, run), e.g. ('sub-001', 'run-01')
            modalities (tuple[str] | list[str]): which modalities to load
                (e.g. ('eeg',) or ('eeg', 'ecg'))

        Returns:
            Data:
                - Data object if EEG exists (ECG is optional).
                - Empty Data (data=[], channels=(), fs=()) if EEG missing or unreadable.
        """
        import os
        import pyedflib
        import warnings

        subj, run = recording

        def find_edf(mod):
            """Return EDF filepath if it exists, else None."""
            mod_dir = os.path.join(data_path, subj, "ses-01", mod)
            if not os.path.isdir(mod_dir):
                return None

            edf = os.path.join(
                mod_dir,
                "_".join(
                    [subj, "ses-01", "task-szMonitoring", run, mod + ".edf"]
                ),
            )
            return edf if os.path.exists(edf) else None

        # === Locate EEG (required) ===
        eeg_file = find_edf("eeg")
        if eeg_file is None:
            warnings.warn(
                f"[Warning] {subj} {run}: EEG file missing — creating EMPTY Data object."
            )
            return cls(data=[], channels=(), fs=())

        # === ECG is OPTIONAL ===
        ecg_file = find_edf("ecg")
        has_ecg = ecg_file is not None

        if not has_ecg:
            # Single, clear note per recording when ECG is missing
            warnings.warn(
                f"[Note] {subj} {run}: ECG missing — continuing with EEG only.",
                stacklevel=2,
            )

        data = []
        channels = []
        sampling_frequencies = []

        for mod in modalities:
            # Map modality to actual file path
            if mod == "eeg":
                mod_file = eeg_file
            elif mod == "ecg" and has_ecg:
                mod_file = ecg_file
            else:
                # Either unsupported modality or ECG requested but not available
                continue

            try:
                import pyedflib  # in case file is imported without it at top
                with pyedflib.EdfReader(mod_file) as edf:
                    sampling_frequencies.extend(edf.getSampleFrequencies())
                    channels.extend(edf.getSignalLabels())
                    n = edf.signals_in_file
                    for i in range(n):
                        data.append(edf.readSignal(i))
            except OSError as e:
                warnings.warn(
                    f"{mod_file}: read error ({e}). Returning EMPTY Data object."
                )
                return cls(data=[], channels=(), fs=())

        if len(data) == 0:
            warnings.warn(
                f"No data loaded for {subj} {run}. Returning EMPTY Data object."
            )
            return cls(data=[], channels=(), fs=())

        data = [np.asarray(d, dtype=np.float32) for d in data]

        return cls(
            data=data,
            channels=tuple(channels),
            fs=tuple(sampling_frequencies),
        )