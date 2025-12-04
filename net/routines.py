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


    K.set_image_data_format('channels_last')
    model.summary()

    name = config.get_name()

    optimizer = Adam(
        learning_rate=config.lr, beta_1=0.9, beta_2=0.999, amsgrad=False
    )

    # 3-class focal loss
    loss = weighted_focal_loss
    auc = AUC(name='auc')

    metrics = [
        'accuracy',
        auc,
        sens,
        spec,
        sens_ovlp,
        fah_ovlp,
        fah_epoch,
        faRate_epoch,
        score,
    ]

    monitor = 'val_score'
    monitor_mode = 'max'

    early_stopping = True
    patience = 10

    callbacks_dir = os.path.join(model_save_path, 'Callbacks')
    history_dir = os.path.join(model_save_path, 'History')
    weights_dir = os.path.join(model_save_path, 'Weights')

    os.makedirs(callbacks_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)

    # Save_weights_only requires .weights.h5
    cb_model = os.path.join(callbacks_dir, name + '_{epoch:02d}.weights.h5')
    csv_logger = CSVLogger(os.path.join(history_dir, name + '.csv'), append=True)

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
        class_weight=getattr(config, 'class_weights', None),
    )

    # Pick best epoch by highest val_score
    best_epoch = int(np.argmax(hist.history['val_score'])) + 1
    best_weights_path = cb_model.format(epoch=best_epoch)

    best_model = model
    best_model.load_weights(best_weights_path)

    # Save final best weights
    final_weights_path = os.path.join(weights_dir, name + '.weights.h5')
    best_model.save_weights(final_weights_path)

    print(f'Saved best model weights to {final_weights_path}')


def predict_net(generator, model_weights_path, model):

    # Load trained weights
    model.load_weights(model_weights_path)

    # Collect true labels from generator (typically one-hot [N, 3])
    # We iterate over the Sequence so the order matches model.predict().
    y_batches = []
    for i in range(len(generator)):
        batch = generator[i]

        # Expect (x, y) or (inputs, targets)
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            x, y = batch[0], batch[1]
        else:
            # Fallback: treat as (x, y)
            x, y = batch

        if y is None:
            continue

        # In case of multi-output generators, take the first output
        if isinstance(y, (list, tuple)):
            if len(y) == 0:
                continue
            y = y[0]

        y = np.asarray(y)
        if y.size == 0:
            continue

        y_batches.append(y)

    if len(y_batches) == 0:
        # No data in generator – return empty arrays
        return np.array([]), np.array([])

    true_labels = np.concatenate(y_batches, axis=0)  
    
    # Get predictions from the model
    prediction = model.predict(generator, verbose=1) 
    # Ensure same length on both arrays (guard against any off-by-one)
    n = min(len(true_labels), len(prediction))
    true_labels = true_labels[:n]
    prediction = prediction[:n]


    # Reduce to pre-ictal probability (class index 1) and binary label  
    if prediction.ndim == 2 and prediction.shape[1] >= 2:
        # class 1 = pre-ictal
        y_pred = prediction[:, 1].astype('float32')
    else:
        # Fallback: treat scalar output as "pre-ictal probability"
        y_pred = prediction.astype('float32').flatten()

    if true_labels.ndim == 2 and true_labels.shape[1] >= 2:
        # Convert one-hot labels to binary mask for pre-ictal
        y_true = true_labels[:, 1].astype('float32')
    else:
        # Assume labels are already binary 0/1
        y_true = true_labels.astype('float32').flatten()

    return y_pred, y_true