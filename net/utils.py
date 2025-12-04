import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


# EEG + ECG preprocessing (normalizing each channel)
def apply_preprocess_eeg(config, raw):

    eeg = raw.data[0].astype(np.float32)
    ecg = raw.data[1].astype(np.float32)

    # Remove extreme spikes
    eeg = np.clip(eeg, -500, 500)
    ecg = np.clip(ecg, -5000, 5000)

    # Z-score each channel independently
    eeg = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-6)
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)

    return np.stack([eeg, ecg], axis=0)


# Focal loss for binary classification
def weighted_focal_loss(gamma=2., alpha=1., class_weights=None):

    if class_weights is None:
        class_weights = K.constant([1., 1.])

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1 - 1e-7)
        ce = -(y_true * K.log(y_pred))
        # weight positive/negative classes
        ce = ce * class_weights
        weight = K.pow((1 - y_pred), gamma)
        fl = weight * ce
        return K.sum(fl, axis=1)

    return loss


# Sensitivity (recall) for pre-ictal
def sens(y_true, y_pred):

    y_true = K.argmax(y_true, axis=1)
    y_pred = K.argmax(y_pred, axis=1)

    tp = K.sum(K.cast(K.equal(y_true, 1) & K.equal(y_pred, 1), 'float32'))
    p = K.sum(K.cast(K.equal(y_true, 1), 'float32')) + 1e-6
    return tp / p


# Specificity for class 0 (non-pre-ictal)
def spec(y_true, y_pred):
    y_true = K.argmax(y_true, axis=1)
    y_pred = K.argmax(y_pred, axis=1)

    tn = K.sum(K.cast(K.equal(y_true, 0) & K.equal(y_pred, 0), 'float32'))
    n = K.sum(K.cast(K.equal(y_true, 0), 'float32')) + 1e-6
    return tn / n


# False-alarm rate per hour 
def fa_rate_epoch(y_true, y_pred):
    y_true = K.argmax(y_true, axis=1)
    y_pred = K.argmax(y_pred, axis=1)

    fp = K.sum(K.cast((K.equal(y_pred, 1) & K.equal(y_true, 0)), 'float32'))
    # Your config uses 2s windows → 1800 windows per hour
    rate = fp / 1800.0
    return rate