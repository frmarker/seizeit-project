import math
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from net.utils import apply_preprocess_eeg
from classes.data import Data

# Cache for preprocessed recordings to avoid reloading EDF
_REC_CACHE = {}


class SequentialGenerator(keras.utils.Sequence):

    def __init__(self, config, recs, segments, batch_size=32,
                 shuffle=False, verbose=True):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

        self.recs = recs
        self.segments = segments

        n_segs = len(segments)
        frame_len = int(config.frame * config.fs)

        self.data_segs = np.empty(
            shape=(n_segs, frame_len, config.CH),
            dtype=np.float32,
        )
        # labels: [non-pre-ictal, pre-ictal]
        self.labels = np.empty(shape = (n_segs, 2), dtype=np.float32)

        pbar = tqdm(total=n_segs, disable=not self.verbose)

        count = 0
        prev_rec = None
        rec_data = None

        for s in segments:
            curr_rec = int(s[0])

            # load new recording if needed
            if curr_rec != prev_rec:
                cache_key = (recs[curr_rec][0], recs[curr_rec][1], config.fs)
                if cache_key in _REC_CACHE:
                    rec_data = _REC_CACHE[cache_key]
                else:
                    # Determine which modalities to load (default: EEG+ECG)
                    modalities = getattr(self.config, 'modalities', ['eeg', 'ecg'])
                    raw = Data.loadData(
                        self.config.data_path,
                        self.recs[curr_rec],
                        modalities = modalities,
                    )
                    # If nothing loaded at all, or some requested modalities are missing,
                    # skip this recording entirely.
                    if len(raw.data) < len(modalities):
                        print(
                            f'Skipping {self.recs[curr_rec][0]} {self.recs[curr_rec][1]} '
                            f'(missing modalities: got {len(raw.data)}, expected {len(modalities)})')
                        prev_rec = None
                        pbar.update(1)
                        continue
                    rec_data = apply_preprocess_eeg(self.config, raw)
                    _REC_CACHE[cache_key] = rec_data
                prev_rec = curr_rec

            if rec_data is None:
                pbar.update(1)
                continue

            start_seg = int(s[1] * config.fs)
            stop_seg = int(s[2] * config.fs)

            # Pad with zeros if beyond recording length
            if stop_seg > len(rec_data[0]):
                self.data_segs[count, :, 0] = np.zeros(frame_len, dtype=np.float32)
                self.data_segs[count, :, 1] = np.zeros(frame_len, dtype=np.float32)
            else:
                self.data_segs[count, :, 0] = rec_data[0][start_seg:stop_seg]
                self.data_segs[count, :, 1] = rec_data[1][start_seg:stop_seg]

            # 2-class labels: 0 = non pre-ictal (interictal + ictal), 1 = pre-ictal
            lbl = int(s[3])
            if lbl == 1:
                # pre-ictal
                self.labels[count, :] = [0, 1]
            else:
                # non-pre-ictal (interictal + ictal collapsed)
                self.labels[count, :] = [1, 0]

            count += 1
            pbar.update(1)

        pbar.close()

        # in case some segments were skipped
        self.data_segs = self.data_segs[:count]
        self.labels = self.labels[:count]

        self.key_array = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.key_array) / self.batch_size)

    def __getitem__(self, index):
        # Compute start/stop and clip stop to the number of available samples
        start = index * self.batch_size
        stop = min((index + 1) * self.batch_size, len(self.key_array))

        # If for some reason start >= stop, return an empty batch
        if start >= stop:
            raise IndexError(f'Batch index {index} out of range in SequentialGenerator')

        keys = np.arange(start=start, stop=stop)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        if self.config.model in ('DeepConvNet', 'EEGnet'):
            x = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(
                0, 2, 1, 3
            )
        else:
            x = self.data_segs[self.key_array[keys], :, :]
        y = self.labels[self.key_array[keys]]
        return x, y


class SegmentedGenerator(keras.utils.Sequence):

    def __init__(self, config, recs, segments, batch_size = 32,
                 shuffle = True, verbose = True):
        super().__init__()

        self.config = config
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.verbose = verbose

        self.recs = recs
        self.segments = segments

        n_segs = len(segments)
        frame_len = int(config.frame * config.fs)

        self.data_segs = np.empty(
            shape=(n_segs, frame_len, config.CH),
            dtype=np.float32,
        )
        self.labels = np.empty(shape=(n_segs, 2), dtype=np.float32)

        segs_to_load = list(segments)  # copy
        pbar = tqdm(total=len(segs_to_load), disable=not self.verbose)

        count = 0

        while segs_to_load:
            curr_rec = int(segs_to_load[0][0])
            comm_recs = [i for i, x in enumerate(segs_to_load) if x[0] == curr_rec]

            cache_key = (self.recs[curr_rec][0], self.recs[curr_rec][1], self.config.fs)
            if cache_key in _REC_CACHE:
                rec_data = _REC_CACHE[cache_key]
            else:
                # Determine which modalities to load (default: EEG+ECG)
                modalities = getattr(self.config, 'modalities', ['eeg', 'ecg'])
                raw = Data.loadData(
                    self.config.data_path,
                    self.recs[curr_rec],
                    modalities=modalities,
                )
                # If nothing loaded at all, or some requested modalities are missing,
                # skip all segments for this recording.
                if len(raw.data) < len(modalities):
                    print(
                        f'Skipping {self.recs[curr_rec][0]} {self.recs[curr_rec][1]} '
                        f'(missing modalities: got {len(raw.data)}, expected {len(modalities)})')
                    segs_to_load = [s for i, s in enumerate(segs_to_load) if i not in comm_recs]
                    pbar.update(len(comm_recs))
                    continue
                rec_data = apply_preprocess_eeg(self.config, raw)
                _REC_CACHE[cache_key] = rec_data

            for r in comm_recs:
                seg = segs_to_load[r]
                start_seg = int(seg[1] * config.fs)
                stop_seg = int(seg[2] * config.fs)

                if stop_seg > len(rec_data[0]):
                    self.data_segs[count, :, 0] = np.zeros(frame_len, dtype=np.float32)
                    self.data_segs[count, :, 1] = np.zeros(frame_len, dtype=np.float32)
                else:
                    self.data_segs[count, :, 0] = rec_data[0][start_seg:stop_seg]
                    self.data_segs[count, :, 1] = rec_data[1][start_seg:stop_seg]

                # 2-class labels: 0 = non pre-ictal (interictal + ictal), 1=pre-ictal
                lbl = int(seg[3])
                if lbl == 1:
                    # pre-ictal
                    self.labels[count, :] = [0, 1]
                else:
                    # non pre-ictal (interictal + ictal collapsed)
                    self.labels[count, :] = [1, 0]

                count += 1
                pbar.update(1)

            segs_to_load = [
                s for i, s in enumerate(segs_to_load) if i not in comm_recs
            ]

        pbar.close()

        self.data_segs = self.data_segs[:count]
        self.labels = self.labels[:count]

        self.key_array = np.arange(len(self.labels))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.key_array) / self.batch_size)

    def __getitem__(self, index):
        # Compute start/stop and clip stop to the number of available samples
        start = index * self.batch_size
        stop = min((index + 1) * self.batch_size, len(self.key_array))

        # If for some reason start >= stop, return an empty batch
        if start >= stop:
            raise IndexError(f'Batch index {index} out of range in SequentialGenerator')

        keys = np.arange(start=start, stop=stop)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        if self.config.model in ('DeepConvNet', 'EEGnet'):
            x = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(
                0, 2, 1, 3
            )
        else:
            x = self.data_segs[self.key_array[keys], :, :]
        y = self.labels[self.key_array[keys]]
        return x, y