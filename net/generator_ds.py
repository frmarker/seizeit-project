import math
import numpy as np
from tensorflow import keras
from tqdm import tqdm

from net.utils import apply_preprocess_eeg
from classes.data import Data

# -------------------------------------------------------
# Cache for preprocessed recordings to avoid reloading EDF
# -------------------------------------------------------
_REC_CACHE = {}


class SequentialGenerator(keras.utils.Sequence):
    """
    Keras sequential data generator (continuous segments in time).

    Args:
        config: experiment config
        recs: list of recordings [[sub-xxx, run-xx], ...]
        segments: list of [rec_idx, start_s, stop_s, label]
        batch_size: batch size
        shuffle: if True, shuffle segment order each epoch
    """

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
        # labels: (n_segments, 3) → [interictal, preictal, ictal]
        self.labels = np.empty(shape=(n_segs, 3), dtype=np.float32)

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
                    raw = Data.loadData(
                        config.data_path,
                        recs[curr_rec],
                        modalities=["eeg", "ecg"],
                    )
                    # raw is never None now; check if empty
                    if len(raw.data) == 0:
                        print(
                            f"Skipping {recs[curr_rec][0]} {recs[curr_rec][1]} "
                            f"(empty Data: missing/broken EEG)"
                        )
                        prev_rec = None
                        pbar.update(1)
                        continue
                    rec_data = apply_preprocess_eeg(config, raw)
                    _REC_CACHE[cache_key] = rec_data
                prev_rec = curr_rec

            if rec_data is None:
                pbar.update(1)
                continue

            start_seg = int(s[1] * config.fs)
            stop_seg = int(s[2] * config.fs)

            # safety: pad with zeros if beyond recording length
            if stop_seg > len(rec_data[0]):
                self.data_segs[count, :, 0] = np.zeros(frame_len, dtype=np.float32)
                self.data_segs[count, :, 1] = np.zeros(frame_len, dtype=np.float32)
            else:
                self.data_segs[count, :, 0] = rec_data[0][start_seg:stop_seg]
                self.data_segs[count, :, 1] = rec_data[1][start_seg:stop_seg]

            # 3-class labels: 0=interictal, 1=pre-ictal, 2=ictal
            lbl = int(s[3])
            if lbl == 0:
                self.labels[count, :] = [1, 0, 0]
            elif lbl == 1:
                self.labels[count, :] = [0, 1, 0]
            elif lbl == 2:
                self.labels[count, :] = [0, 0, 1]
            else:
                self.labels[count, :] = [0, 0, 0]

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
            raise IndexError(f"Batch index {index} out of range in SequentialGenerator")

        keys = np.arange(start=start, stop=stop)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        if self.config.model in ("DeepConvNet", "EEGnet"):
            x = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(
                0, 2, 1, 3
            )
        else:
            x = self.data_segs[self.key_array[keys], :, :]
        y = self.labels[self.key_array[keys]]
        return x, y


class SegmentedGenerator(keras.utils.Sequence):
    """
    Segmented generator optimized for many segments from many recordings.

    segments: [rec_idx, start_s, stop_s, label]
    labels:   0 = interictal, 1 = pre-ictal, 2 = ictal
    """

    def __init__(self, config, recs, segments, batch_size=32,
                 shuffle=True, verbose=True):
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
        self.labels = np.empty(shape=(n_segs, 3), dtype=np.float32)

        segs_to_load = list(segments)  # copy
        pbar = tqdm(total=len(segs_to_load), disable=not self.verbose)

        count = 0

        while segs_to_load:
            curr_rec = int(segs_to_load[0][0])
            comm_recs = [i for i, x in enumerate(segs_to_load) if x[0] == curr_rec]

            cache_key = (recs[curr_rec][0], recs[curr_rec][1], config.fs)
            if cache_key in _REC_CACHE:
                rec_data = _REC_CACHE[cache_key]
            else:
                raw = Data.loadData(
                    config.data_path,
                    recs[curr_rec],
                    modalities=["eeg", "ecg"],
                )
                if len(raw.data) == 0:
                    print(
                        f"Skipping {recs[curr_rec][0]} {recs[curr_rec][1]} "
                        f"(empty Data: missing/broken EEG)"
                    )
                    segs_to_load = [
                        s for i, s in enumerate(segs_to_load) if i not in comm_recs
                    ]
                    pbar.update(len(comm_recs))
                    continue
                rec_data = apply_preprocess_eeg(config, raw)
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

                lbl = int(seg[3])
                if lbl == 0:
                    self.labels[count, :] = [1, 0, 0]
                elif lbl == 1:
                    self.labels[count, :] = [0, 1, 0]
                elif lbl == 2:
                    self.labels[count, :] = [0, 0, 1]
                else:
                    self.labels[count, :] = [0, 0, 0]

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
            raise IndexError(f"Batch index {index} out of range in SequentialGenerator")

        keys = np.arange(start=start, stop=stop)
        x, y = self.__data_generation__(keys)
        return x, y

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)

    def __data_generation__(self, keys):
        if self.config.model in ("DeepConvNet", "EEGnet"):
            x = self.data_segs[self.key_array[keys], :, :, np.newaxis].transpose(
                0, 2, 1, 3
            )
        else:
            x = self.data_segs[self.key_array[keys], :, :]
        y = self.labels[self.key_array[keys]]
        return x, y