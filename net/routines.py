import os
import numpy as np

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    LearningRateScheduler,
)
from tensorflow.keras.metrics import AUC

from net.utils import (
    weighted_focal_loss,
    sens,
    spec,
    sens_ovlp,
    fah_ovlp,
    fah_epoch,
    faRate_epoch,
    score,
    decay_schedule,
)


def train_net(config, model, gen_train, gen_val, model_save_path):
    """
    Routine to train the model with the desired configurations.
    """

    K.set_image_data_format("channels_last")
    model.summary()

    name = config.get_name()

    optimizer = Adam(
        learning_rate=config.lr, beta_1=0.9, beta_2=0.999, amsgrad=False
    )

    # 3-class focal loss
    loss = weighted_focal_loss
    auc = AUC(name="auc")

    metrics = [
        "accuracy",
        auc,
        sens,
        spec,
        sens_ovlp,
        fah_ovlp,
        fah_epoch,
        faRate_epoch,
        score,
    ]

    monitor = "val_score"
    monitor_mode = "max"

    early_stopping = True
    patience = 10

    callbacks_dir = os.path.join(model_save_path, "Callbacks")
    history_dir = os.path.join(model_save_path, "History")
    weights_dir = os.path.join(model_save_path, "Weights")

    os.makedirs(callbacks_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Keras 3: save_weights_only requires .weights.h5
    cb_model = os.path.join(callbacks_dir, name + "_{epoch:02d}.weights.h5")
    csv_logger = CSVLogger(os.path.join(history_dir, name + ".csv"), append=True)

    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )

    mc = ModelCheckpoint(
        cb_model,
        monitor=monitor,
        verbose=1,
        save_weights_only=True,
        save_best_only=False,
        mode=monitor_mode,
    )

    if early_stopping:
        es = EarlyStopping(
            monitor = monitor,
            patience=10,
            verbose=1,
            mode = monitor_mode,
        )

    lr_sched = LearningRateScheduler(decay_schedule)

    callbacks_list = [mc, csv_logger, lr_sched]
    if early_stopping:
        callbacks_list.insert(1, es)

    hist = model.fit(
        gen_train,
        validation_data=gen_val,
        epochs=config.nb_epochs,
        callbacks=callbacks_list,
        shuffle=False,
        verbose=1,
        class_weight=getattr(config, "class_weights", None),
    )

    # Pick best epoch by highest val_score
    best_epoch = int(np.argmax(hist.history["val_score"])) + 1
    best_weights_path = cb_model.format(epoch=best_epoch)

    best_model = model
    best_model.load_weights(best_weights_path)

    # Save final best weights
    final_weights_path = os.path.join(weights_dir, name + ".weights.h5")
    best_model.save_weights(final_weights_path)

    print(f"Saved best model weights to {final_weights_path}")


def predict_net(generator, model_weights_path, model):
    """
    Routine to obtain predictions from the trained model.

    Returns:
        y_pred: 1D array of pre-ictal probabilities (class 1)
        y_true: 1D array of binary labels (1 = pre-ictal, 0 = other)
    """

    K.set_image_data_format("channels_last")
    model.load_weights(model_weights_path)

    # collect true labels from generator (one-hot [N,3])
    y_aux = []
    for j in range(len(generator)):
        _, y = generator[j]
        if y is not None and len(y) > 0:
            y_aux.append(y)

    if len(y_aux) == 0:
        # no data in generator – return empty arrays
        return np.array([]), np.array([])

    true_labels = np.vstack(y_aux)  # [N,3]

    prediction = model.predict(generator, verbose=0)  # [N,3]

    # Pre-ictal probability = class 1
    y_pred = prediction[:, 1].astype("float32")

    # Convert one-hot to binary mask for class 1
    y_true = true_labels[:, 1].astype("float32")

    return y_pred, y_true