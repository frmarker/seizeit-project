from tensorflow import keras
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    Concatenate,
    BatchNormalization,
    SpatialDropout1D,
    GRU,
    Dense,
    Softmax,
)
from tensorflow.keras.models import Model


def net(config):
    """
    ChronoNet-style architecture for binary pre-ictal detection.

    Labels (after remapping):
    0 = non-pre-ictal (interictal + ictal)
    1 = pre-ictal
    """

    n_samples = int(config.frame * config.fs)
    n_ch = config.CH  # should be 2 (focal + cross)

    inp = Input(shape=(n_samples, n_ch), name="input_layer")

    # ---- Block 1: multi-scale temporal conv ----
    c1 = Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu")(inp)
    c2 = Conv1D(32, kernel_size=9, strides=2, padding="same", activation="relu")(inp)
    c3 = Conv1D(32, kernel_size=17, strides=2, padding="same", activation="relu")(inp)

    x = Concatenate(name="concat_1")([c1, c2, c3])
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.2)(x)

    # ---- Block 2 ----
    c4 = Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    c5 = Conv1D(32, kernel_size=9, strides=2, padding="same", activation="relu")(x)
    c6 = Conv1D(32, kernel_size=17, strides=2, padding="same", activation="relu")(x)

    x = Concatenate(name="concat_2")([c4, c5, c6])
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.2)(x)

    # ---- Block 3 ----
    c7 = Conv1D(32, kernel_size=5, strides=2, padding="same", activation="relu")(x)
    c8 = Conv1D(32, kernel_size=9, strides=2, padding="same", activation="relu")(x)
    c9 = Conv1D(32, kernel_size=17, strides=2, padding="same", activation="relu")(x)

    x = Concatenate(name="concat_3")([c7, c8, c9])
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.2)(x)

    # ---- Recurrent stack ----
    g1 = GRU(32, return_sequences=True)(x)
    g2 = GRU(32, return_sequences=True)(g1)
    g_cat = Concatenate(name="gru_concat")([g1, g2])
    g3 = GRU(32, return_sequences=True)(g_cat)
    g_cat2 = Concatenate(name="gru_concat2")([g1, g2, g3])

    g_final = GRU(32, return_sequences=False)(g_cat2)

    # ---- Output: 2-class softmax (0=non-pre-ictal, 1=pre-ictal) ----
    logits = Dense(2, name="dense")(g_final)
    out = Softmax(name="softmax")(logits)

    model = Model(inputs=inp, outputs=out, name="ChronoNet_binary_preictal")

    return model