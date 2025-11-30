import numpy as np
from scipy import signal
import tensorflow as tf
from sklearn.metrics import roc_auc_score

K = tf.keras.backend

# ------------------------------------------------------------------
# Simple in-memory cache for preprocessed EEG per recording
# Keyed by (id(rec), target_fs) so repeated calls on the same
# Data object and sampling rate do not redo expensive filtering
# and resampling.
# ------------------------------------------------------------------
_PREPROC_CACHE = {}


def set_gpu():
    """
    Detects GPUs and (currently) sets automatic memory growth
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')

            print(len(gpus), 'Physical GPUs, ', len(logical_gpus), 'Logical GPUs')
        except RuntimeError as e:
            print(e)


# ------------------------------------------------------------------
# LOSSES
# ------------------------------------------------------------------

def focal_loss(y_true, y_pred):
    """
    Binary focal loss (legacy).
    """
    gamma = 2.0
    alpha = 0.25

    y_true = tf.cast(y_true, tf.float32)
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1 - p_t), gamma)
    loss = weight * cross_entropy
    loss = K.mean(K.sum(loss, axis=1))
    return loss


def weighted_focal_loss(y_true, y_pred, gamma=2.0):
    """
    Multi-class focal-style loss for 3-class one-hot labels:
    0 = interictal, 1 = pre-ictal, 2 = ictal.

    We give extra weight to the pre-ictal class (index 1), so the
    optimization is biased toward correctly modeling pre-ictal segments.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    eps = K.epsilon()
    y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

    # Class weights: emphasize pre-ictal (class 1)
    class_weights = tf.constant([1.0, 2.0, 1.0], dtype=tf.float32)

    # Standard categorical cross-entropy per class
    ce = -y_true * tf.math.log(y_pred)  # (batch, 3)

    # Apply class weights only on the true class positions
    ce = ce * class_weights

    # Focal modulation based on p_t (probability of the true class)
    p_t = tf.reduce_sum(y_true * y_pred, axis=-1)  # (batch,)
    focal_factor = tf.pow(1.0 - p_t, gamma)

    loss = tf.reduce_sum(ce, axis=-1) * focal_factor  # (batch,)
    return tf.reduce_mean(loss)


def weighted_binary_crossentropy(zero_weight, one_weight):
    def _wbce(y_true, y_pred):
        b_ce = tf.keras.metrics.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true[:, 1] * one_weight + (1 - y_true[:, 1]) * zero_weight
        weighted_b_ce = weight_vector * b_ce
        return K.mean(weighted_b_ce)

    return _wbce


def weighted_binary_crossentropy_adapt(y_true, y_pred):
    b_ce = tf.keras.metrics.binary_crossentropy(y_true, y_pred)

    one_wt = tf.cast(
        tf.reduce_sum(tf.cast(y_true[:, 1] == 0, tf.float32)) /
        (tf.reduce_sum(tf.cast(y_true[:, 1], tf.float32)) + K.epsilon()),
        'float32',
    )
    zero_wt = tf.constant(1, 'float32')

    weight_vector = y_true[:, 1] * one_wt + (1 - y_true[:, 1]) * zero_wt
    weighted_b_ce = weight_vector * b_ce
    return K.mean(weighted_b_ce)


def decay_schedule(epoch, lr):
    if lr > 1e-5:
        if (epoch + 1) % 10 == 0:
            lr = lr / 2
    return lr


# ------------------------------------------------------------------
# METRICS
# ------------------------------------------------------------------

def aucc(y_true, y_pred):
    return tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)


def sens(y_true, y_pred):
    """
    Sensitivity (recall) for the pre-ictal class (class index 1).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    true_pre = y_true[:, 1]
    pred_pre = tf.round(y_pred[:, 1])

    true_positives = tf.reduce_sum(true_pre * pred_pre)
    possible_positives = tf.reduce_sum(true_pre)
    return tf.math.divide_no_nan(true_positives, possible_positives)


def spec(y_true, y_pred):
    """
    Specificity on the interictal class (class 0) w.r.t. pre-ictal predictions.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    inter = y_true[:, 0]
    pred_pre = tf.round(y_pred[:, 1])

    true_negatives = tf.reduce_sum(inter * (1.0 - pred_pre))
    possible_negatives = tf.reduce_sum(inter)
    return tf.math.divide_no_nan(true_negatives, possible_negatives)


def sens_ovlp(y_true, y_pred):
    """
    Any-overlap sensitivity for pre-ictal events (class 1).
    Uses `perf_measure_ovlp_tensor` on the pre-ictal column only.
    """
    TP, FN, _ = perf_measure_ovlp_tensor(y_true, y_pred)
    TP = tf.cast(TP, tf.float64)
    FN = tf.cast(FN, tf.float64)
    return tf.math.divide_no_nan(TP, TP + FN)


def fah_ovlp(y_true, y_pred):
    """
    False alarms per hour (any-overlap) for pre-ictal events.
    """
    _, _, FP = perf_measure_ovlp_tensor(y_true, y_pred)
    FP = tf.cast(FP, tf.float64)
    length = tf.cast(tf.shape(y_true)[0], tf.float64)
    return FP * tf.constant(3600.0, dtype=tf.float64) / length


def fah_epoch(y_true, y_pred):
    """
    False alarms per hour for pre-ictal predictions on truly interictal segments.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    inter = y_true[:, 0]
    pred_pre = tf.round(y_pred[:, 1])

    fa_epoch = tf.reduce_sum(inter * pred_pre)
    fa_epoch = tf.cast(fa_epoch, tf.float64)
    length = tf.cast(tf.shape(y_true)[0], tf.float64)

    return fa_epoch * tf.constant(3600.0, dtype=tf.float64) / length


def faRate_epoch(y_true, y_pred):
    """
    False alarm rate per segment (pre-ictal predictions on interictal segments).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    inter = y_true[:, 0]
    pred_pre = tf.round(y_pred[:, 1])

    fa_epoch = tf.reduce_sum(inter * pred_pre)
    fa_epoch = tf.cast(fa_epoch, tf.float64)
    length = tf.cast(tf.shape(y_true)[0], tf.float64)
    return tf.math.divide_no_nan(fa_epoch, length)


def score(y_true, y_pred):
    s = sens_ovlp(y_true, y_pred)
    f = fah_epoch(y_true, y_pred)
    return s * tf.constant(100.0, dtype=tf.float64) - tf.constant(0.4, dtype=tf.float64) * f


def perf_measure_ovlp_tensor(y_true, y_pred):
    """
    Any-overlap performance metric in pure TF for pre-ictal class.
    Returns integer TP, FN, FP.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # TRUE events (pre-ictal column)
    true_evs = tf.concat([y_true[:, 1], tf.constant([0.0], dtype=tf.float32)], axis=0)
    true_evs = tf.concat([tf.constant([0.0], dtype=tf.float32), true_evs], axis=0)

    mask = tf.equal(true_evs, 1.0)
    start_positions = tf.where(tf.logical_and(~mask[:-1], mask[1:])) + 1
    end_positions = tf.where(tf.logical_and(mask[:-1], ~mask[1:])) + 1
    true_ranges = tf.concat([start_positions, end_positions], axis=1)

    # PREDICTED events (pre-ictal column)
    pred_evs = tf.concat(
        [tf.round(y_pred[:, 1]), tf.constant([0.0], dtype=tf.float32)], axis=0
    )
    pred_evs = tf.concat([tf.constant([0.0], dtype=tf.float32), pred_evs], axis=0)

    mask = tf.equal(pred_evs, 1.0)
    start_positions = tf.where(tf.logical_and(~mask[:-1], mask[1:])) + 1
    end_positions = tf.where(tf.logical_and(mask[:-1], ~mask[1:])) + 1
    pred_ranges = tf.concat([start_positions, end_positions], axis=1)

    def empty_int_triplet():
        return (tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64),
                tf.constant(0, dtype=tf.int64))

    cond_no_true = tf.equal(tf.shape(true_ranges)[0], 0)
    cond_no_pred = tf.equal(tf.shape(pred_ranges)[0], 0)
    if_true_empty = tf.logical_or(cond_no_true, cond_no_pred)

    def compute_counts():
        true_expanded = tf.cast(tf.expand_dims(true_ranges, axis=0), tf.int64)
        pred_expanded = tf.cast(tf.expand_dims(pred_ranges, axis=1), tf.int64)

        overlap_start = tf.maximum(true_expanded[:, :, 0], pred_expanded[:, :, 0])
        overlap_end = tf.minimum(true_expanded[:, :, 1], pred_expanded[:, :, 1])

        overlaps = tf.maximum(
            tf.constant(0, dtype=tf.int64),
            overlap_end - overlap_start,
        )
        overlaps_sum_true = tf.reduce_sum(overlaps, axis=0)
        overlaps_sum_pred = tf.reduce_sum(overlaps, axis=1)

        TP = tf.cast(tf.math.count_nonzero(overlaps_sum_true), tf.int64)
        FN = tf.cast(tf.shape(overlaps_sum_true)[0], tf.int64) - TP
        FP = tf.cast(tf.shape(overlaps_sum_pred)[0], tf.int64) - tf.cast(
            tf.math.count_nonzero(overlaps_sum_pred), tf.int64
        )
        return TP, FN, FP

    TP, FN, FP = tf.cond(if_true_empty, empty_int_triplet, compute_counts)
    return TP, FN, FP


# ------------------------------------------------------------------
# EEG PREPROCESSING WITH CACHE
# ------------------------------------------------------------------

def apply_preprocess_eeg(config, rec):
    """
    Take a Data object and return preprocessed focal and cross channels.

    Robust to:
      - empty Data (no channels)  → raises ValueError
      - missing expected channel names → falls back to first/second channel.
    """
    # Basic sanity
    if rec is None or not hasattr(rec, "data") or len(rec.data) == 0:
        raise ValueError("Empty Data object passed to apply_preprocess_eeg.")

    cache_key = (id(rec), config.fs)
    if cache_key in _PREPROC_CACHE:
        return _PREPROC_CACHE[cache_key]

    channels = list(rec.channels)

    # Find focal channel
    idx_focal = [i for i, c in enumerate(channels) if c == 'BTEleft SD']
    if not idx_focal:
        idx_focal = [i for i, c in enumerate(channels) if c == 'BTEright SD']
    if not idx_focal:
        # fallback: first channel
        idx_focal = [0]

    # Find cross channel
    idx_cross = [i for i, c in enumerate(channels) if c == 'CROSStop SD']
    if not idx_cross:
        # fallback: a different channel if possible, else re-use focal
        if len(channels) > 1:
            idx_cross = [1]
        else:
            idx_cross = idx_focal

    ch_focal, _ = pre_process_ch(rec.data[idx_focal[0]], rec.fs[idx_focal[0]], config.fs)
    ch_cross, _ = pre_process_ch(rec.data[idx_cross[0]], rec.fs[idx_cross[0]], config.fs)

    result = [ch_focal.astype(np.float32), ch_cross.astype(np.float32)]
    _PREPROC_CACHE[cache_key] = result
    return result


def pre_process_ch(ch_data, fs_data, fs_resamp):
    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp * len(ch_data) / fs_data))

    b, a = signal.butter(4, 0.5 / (fs_resamp / 2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60 / (fs_resamp / 2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, [49.5 / (fs_resamp / 2), 50.5 / (fs_resamp / 2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)

    return ch_data, fs_resamp


def pre_process_ch(ch_data, fs_data, fs_resamp):

    if fs_resamp != fs_data:
        ch_data = signal.resample(ch_data, int(fs_resamp*len(ch_data)/fs_data))
    
    b, a = signal.butter(4, 0.5/(fs_resamp/2), 'high')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, 60/(fs_resamp/2), 'low')
    ch_data = signal.filtfilt(b, a, ch_data)

    b, a = signal.butter(4, [49.5/(fs_resamp/2), 50.5/(fs_resamp/2)], 'bandstop')
    ch_data = signal.filtfilt(b, a, ch_data)

    return ch_data, fs_resamp


#### EVENT & MASK MANIPULATION ###

def eventList2Mask(events, totalLen, fs):
    """Convert list of events to mask.
    
    Returns a logical array of length totalLen.
    All event epochs are set to True
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        totalLen: length of array to return in samples
        fs: sampling frequency of the data in Hertz
    Return:
        mask: logical array set to True during event epochs and False the rest
              if the time.
    """
    mask = np.zeros((totalLen,))
    for event in events:
        for i in range(min(int(event[0]*fs), totalLen), min(int(event[1]*fs), totalLen)):
            mask[i] = 1
    return mask


def mask2eventList(mask, fs):
    """Convert mask to list of events.
        
    Args:
        mask: logical array set to True during event epochs and False the rest
          if the time.
        fs: sampling frequency of the data in Hertz
    Return:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
    """
    events = list()
    tmp = []
    start_i = np.where(np.diff(np.array(mask, dtype=int)) == 1)[0]
    end_i = np.where(np.diff(np.array(mask, dtype=int)) == -1)[0]
    
    if len(start_i) == 0 and mask[0]:
        events.append([0, (len(mask)-1)/fs])
    else:
        # Edge effect
        if mask[0]:
            events.append([0, (end_i[0]+1)/fs])
            end_i = np.delete(end_i, 0)
        # Edge effect
        if mask[-1]:
            if len(start_i):
                tmp = [[(start_i[-1]+1)/fs, (len(mask))/fs]]
                start_i = np.delete(start_i, len(start_i)-1)
        for i in range(len(start_i)):
            events.append([(start_i[i]+1)/fs, (end_i[i]+1)/fs])
        events += tmp
    return events


def merge_events(events, distance):
    """ Merge events.
    
    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        distance: maximum distance (in seconds) between events to be merged
    Return:
        events: list of events (after merging) times in seconds.
    """
    i = 1
    tot_len = len(events)
    while i < tot_len:
        if events[i][0] - events[i-1][1] < distance:
            events[i-1][1] = events[i][1]
            events.pop(i)
            tot_len -= 1
        else:
            i += 1
    return events


def get_events(events, margin):
    ''' Converts the unprocessed events to the post-processed events based on physiological constrains:
    - seizure alarm events distanced by 0.2*margin (in seconds) are merged together
    - only events with a duration longer than margin*0.8 are kept
    (for more info, check: K. Vandecasteele et al., “Visual seizure annotation and automated seizure detection using
    behind-the-ear elec- troencephalographic channels,” Epilepsia, vol. 61, no. 4, pp. 766–775, 2020.)

    Args:
        events: list of events times in seconds. Each row contains two
                columns: [start time, end time]
        margin: float, the desired margin in seconds

    Returns:
        ev_list: list of events times in seconds after merging and discarding short events.
    '''
    events_merge = merge_events(events, 0.2*margin)
    ev_list = []
    for i in range(len(events_merge)):
        if events_merge[i][1] - events_merge[i][0] >= margin*0.8:
            ev_list.append(events_merge[i])

    return ev_list



def post_processing(y_pred, fs, th, margin):
    ''' Post process the predictions given by the model based on physiological constraints: a seizure is
    not shorter than 10 seconds and events separated by 2 seconds are merged together.

    Args:
        y_pred: array with the seizure classification probabilties (of each segment)
        fs: sampling frequency of the y_pred array (1/window length - in this challenge fs = 1/2)
        th: threshold value for seizure probability (float between 0 and 1)
        margin: float, the desired margin in seconds (check get_events)
    
    Returns:
        pred: array with the processed classified labels by the model
    '''
    pred = (y_pred > th)
    events = mask2eventList(pred, fs)
    events = get_events(events, margin)
    pred = eventList2Mask(events, len(y_pred), fs)

    return pred


def getOverlap(a, b):
    ''' If > 0, the two intervals overlap.
    a = [start_a, end_a]; b = [start_b, end_b]
    '''
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def perf_measure_epoch(y_true, y_pred):
    ''' Calculate the performance metrics based on the EPOCH method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments

    Returns:
        TP: true positives
        FP: false positives
        TN: true negatives
        FN: false negatives
    '''

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_true[i] == y_pred[i] == 1:
           TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
           FP += 1
        if y_true[i] == y_pred[i] == 0:
           TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
           FN += 1

    return TP, FP, TN, FN


def perf_measure_ovlp(y_true, y_pred, fs):
    ''' Calculate the performance metrics based on the any-overlap method.
    
    Args:
        y_true: array with the ground-truth labels of the segments
        y_pred: array with the predicted labels of the segments
        fs: sampling frequency of the predicted and ground-truth label arrays
            (in this challenge, fs = 1/2)

    Returns:
        TP: true positives
        FP: false positives
        FN: false negatives
    '''
    true_events = mask2eventList(y_true, fs)
    pred_events = mask2eventList(y_pred, fs)

    TP = 0
    FP = 0
    FN = 0

    for pr in pred_events:
        found = False
        for tr in true_events:
            if getOverlap(pr, tr) > 0:
                TP += 1
                found = True
        if not found:
            FP += 1
    for tr in true_events:
        found = False
        for pr in pred_events:
            if getOverlap(tr, pr) > 0:
                found = True
        if not found:
            FN += 1

    return TP, FP, FN


def get_metrics_scoring(y_pred, y_true, fs, th):
    ''' Get the score for the challenge.

    Args:
        pred_file: path to the prediction file containing the objects 'filenames',
                   'predictions' and 'labels' (as returned by 'predict_net' function)
    
    Returns:
        score: the score of the challenge
        sens_ovlp: sensitivity calculated with the any-overlap method
        FA_epoch: false alarm rate (false alarms per hour) calculated with the EPOCH method
    '''

    total_N = len(y_pred)*(1/fs)
    total_seiz = np.sum(y_true)

    # Post process predictions (merge predicted events separated by 2 second and discard events smaller than 8 seconds)
    y_pred = post_processing(y_pred, fs=fs, th=th, margin=10)

    TP_epoch, FP_epoch, TN_epoch, FN_epoch = perf_measure_epoch(y_true, y_pred)

    TP_ovlp, FP_ovlp, FN_ovlp = perf_measure_ovlp(y_true, y_pred, fs=1/2)

    if total_seiz == 0:
        sens_ovlp = float("nan")
        prec_ovlp = float("nan")
        f1_ovlp = float("nan")
    else:
        sens_ovlp = TP_ovlp/(TP_ovlp + FN_ovlp)
        if TP_ovlp == 0 and FP_ovlp == 0:
            prec_ovlp = float("nan")
            f1_ovlp = float("nan")
        else:
            prec_ovlp = TP_ovlp/(TP_ovlp + FP_ovlp)
            if prec_ovlp+sens_ovlp == 0:
                f1_ovlp = float("nan")
            else:
                f1_ovlp = (2*prec_ovlp*sens_ovlp)/(prec_ovlp+sens_ovlp)
    
    FA_ovlp = FP_ovlp*3600/total_N
    FA_epoch = FP_epoch*3600/total_N

    if total_seiz == 0:
        sens_epoch = float("nan")
        prec_epoch = float("nan")
        f1_epoch = float("nan")
    else:
        sens_epoch = TP_epoch/(TP_epoch + FN_epoch)
        if TP_ovlp == 0 and FP_ovlp == 0:
            prec_epoch = float("nan")
            f1_epoch = float("nan")
        else:
            prec_epoch = TP_epoch/(TP_epoch + FP_epoch)
            if prec_epoch+sens_epoch == 0:
                f1_epoch = float("nan")
            else:
                f1_epoch = (2*prec_epoch*sens_epoch)/(prec_epoch+sens_epoch)

    spec_epoch = TN_epoch/(TN_epoch + FP_epoch)

    return sens_ovlp, prec_ovlp, FA_ovlp, f1_ovlp, sens_epoch, spec_epoch, prec_epoch, FA_epoch, f1_epoch
