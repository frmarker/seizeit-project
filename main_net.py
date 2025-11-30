import os
import gc
import time
import pickle

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf

from net.key_generator import (
    generate_data_keys_sequential,
    generate_data_keys_subsample,
    generate_data_keys_sequential_window,
)
from net.generator_ds import SegmentedGenerator, SequentialGenerator
from net.routines import train_net, predict_net
from net.utils import apply_preprocess_eeg
from classes.data import Data


def train(config, load_generators, save_generators):
    """ Routine to run the model's training routine.

        Args:
            config (cls): a config object with the data input type and model parameters
            load_generators (bool): boolean to load the training and validation generators from file
            save_generators (bool): boolean to save the training and validation generators
    """

    name = config.get_name()

    # Select model
    if config.model == 'ChronoNet':
        from net.ChronoNet import net
    elif config.model == 'EEGnet':
        from net.EEGnet import net
    elif config.model == 'DeepConvNet':
        from net.DeepConv_Net import net
    else:
        raise ValueError(f"Unknown model type: {config.model}")

    # Ensure save dirs
    if not os.path.exists(os.path.join(config.save_dir, 'models')):
        os.mkdir(os.path.join(config.save_dir, 'models'))

    model_save_path = os.path.join(config.save_dir, 'models', name)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)

    config_path = os.path.join(config.save_dir, 'models', name, 'configs')
    if not os.path.exists(config_path):
        os.mkdir(config_path)

    # Save config used for this run
    config.save_config(save_path=config_path)

    #######################################################################################################################
    ### Fixed train/val/test ###
    #######################################################################################################################
    if config.cross_validation == 'fixed':

        if config.dataset == 'SZ2':
            # ------------------------------------------------------------------
            # Build list of training recordings
            # ------------------------------------------------------------------
            train_pats_list = pd.read_csv(
                os.path.join('net', 'datasets', 'SZ2_training.tsv'),
                sep='\t',
                header=None,
                skiprows=[0, 1, 2]
            )
            train_pats_list = train_pats_list[0].to_list()

            train_recs_list = [
                [s, r.split('_')[-2]]
                for s in train_pats_list
                for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg'))
                if 'edf' in r
            ]

            # Names/paths for generator pickles
            gen_name = f"{config.dataset}_frame-{config.frame}_sampletype-{config.sample_type}"

            meta_train_path = os.path.join('net', 'generators', f"meta_train_{gen_name}.pkl")
            meta_val_path   = os.path.join('net', 'generators', f"meta_val_{gen_name}.pkl")

            gen_train_path  = os.path.join('net', 'generators', f"gen_train_{gen_name}.pkl")
            gen_val_path    = os.path.join('net', 'generators', f"gen_val_{gen_name}.pkl")

            gen_train = None
            gen_val = None

            # ------------------------------------------------------------------
            # OPTION 1: Load existing generator metadata / generator objects
            # ------------------------------------------------------------------
            if load_generators:
                print('Loading generator metadata...')

                used_source = None

                # Prefer new-style meta_* pickles
                if os.path.exists(meta_train_path) and os.path.exists(meta_val_path):
                    try:
                        with open(meta_train_path, 'rb') as f:
                            meta_train = pickle.load(f)
                        with open(meta_val_path, 'rb') as f:
                            meta_val = pickle.load(f)

                        # Case A: proper metadata dicts
                        if (
                            isinstance(meta_train, dict)
                            and isinstance(meta_val, dict)
                            and "recs_list" in meta_train
                            and "segments" in meta_train
                            and "recs_list" in meta_val
                            and "segments" in meta_val
                        ):
                            train_recs_list = meta_train["recs_list"]
                            train_segments  = meta_train["segments"]

                            val_recs_list = meta_val["recs_list"]
                            val_segments  = meta_val["segments"]

                            gen_train = SegmentedGenerator(
                                config,
                                train_recs_list,
                                train_segments,
                                batch_size=config.batch_size,
                                shuffle=True
                            )
                            gen_val = SequentialGenerator(
                                config,
                                val_recs_list,
                                val_segments,
                                batch_size=600,
                                shuffle=False
                            )
                            used_source = "meta"
                            print(f"Loaded generator metadata from meta_* pickles for {gen_name}.")

                        # Case B: meta_* actually contain full generator objects
                        elif (
                            isinstance(meta_train, SegmentedGenerator)
                            and isinstance(meta_val, SequentialGenerator)
                        ):
                            gen_train = meta_train
                            gen_val = meta_val
                            used_source = "meta_generators"
                            print(f"Loaded generator objects stored inside meta_* pickles for {gen_name}.")

                        else:
                            print(
                                "meta_train/meta_val are not in the expected format. "
                                f"Types are: train={type(meta_train)}, val={type(meta_val)}. "
                                "Will try gen_* pickles."
                            )

                    except Exception as e:
                        print(f"Error loading meta_* pickles: {e}")
                        print("Will try gen_* pickles next.")

                # Fallback: old-style gen_* pickles (full generator objects)
                if used_source is None and os.path.exists(gen_train_path) and os.path.exists(gen_val_path):
                    try:
                        with open(gen_train_path, 'rb') as f:
                            gen_train = pickle.load(f)
                        with open(gen_val_path, 'rb') as f:
                            gen_val = pickle.load(f)
                        used_source = "gen"
                        print(f"Loaded generator objects from gen_* pickles for {gen_name}.")
                    except Exception as e:
                        print(f"Error loading gen_* pickles: {e}")

                # If neither worked, fall back to regenerating
                if used_source is None or gen_train is None or gen_val is None:
                    print("No valid generator pickles found or they could not be loaded. Regenerating from raw data.")
                    load_generators = False

            # ------------------------------------------------------------------
            # OPTION 2: Generate generators from raw data (and optionally save)
            # ------------------------------------------------------------------
            if not load_generators:
                # Training segments
                if config.sample_type == 'subsample':
                    train_segments = generate_data_keys_subsample(config, train_recs_list)
                else:
                    train_segments = generate_data_keys_sequential(config, train_recs_list)

                print('Generating training segments...')
                gen_train = SegmentedGenerator(
                    config,
                    train_recs_list,
                    train_segments,
                    batch_size=config.batch_size,
                    shuffle=True
                )

                # Validation recordings
                val_pats_list = pd.read_csv(
                    os.path.join('net', 'datasets', 'SZ2_validation.tsv'),
                    sep='\t',
                    header=None,
                    skiprows=[0, 1, 2]
                )
                val_pats_list = val_pats_list[0].to_list()
                val_recs_list = [
                    [s, r.split('_')[-2]]
                    for s in val_pats_list
                    for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg'))
                    if 'edf' in r
                ]

                # 5-minute sequential windows
                val_segments = generate_data_keys_sequential_window(config, val_recs_list, 5 * 60)

                print('Generating validation segments...')
                gen_val = SequentialGenerator(
                    config,
                    val_recs_list,
                    val_segments,
                    batch_size=600,
                    shuffle=False
                )

                if save_generators:
                    print("Saving generator metadata and generator objects...")

                    if not os.path.exists('net/generators'):
                        os.mkdir('net/generators')

                    # Save lightweight metadata
                    meta_train = {"recs_list": train_recs_list, "segments": train_segments}
                    meta_val   = {"recs_list": val_recs_list, "segments": val_segments}

                    with open(meta_train_path, "wb") as f:
                        pickle.dump(meta_train, f, pickle.HIGHEST_PROTOCOL)
                    with open(meta_val_path, "wb") as f:
                        pickle.dump(meta_val, f, pickle.HIGHEST_PROTOCOL)

                    # Also save full generator objects (backward compatibility)
                    with open(gen_train_path, "wb") as f:
                        pickle.dump(gen_train, f, pickle.HIGHEST_PROTOCOL)
                    with open(gen_val_path, "wb") as f:
                        pickle.dump(gen_val, f, pickle.HIGHEST_PROTOCOL)

            # ------------------------------------------------------------------
            # Train the model using the prepared generators
            # ------------------------------------------------------------------
            print('### Training model....')
            model = net(config)

            start_train = time.time()
            train_net(config, model, gen_train, gen_val, model_save_path)
            end_train = time.time() - start_train
            print('Total train duration = ', end_train / 60)

    #######################################################################################################################
    #######################################################################################################################


def predict(config):

    name = config.get_name()

    model_save_path = os.path.join(config.save_dir, 'models', name)

    # Ensure prediction directories
    if not os.path.exists(os.path.join(config.save_dir, 'predictions')):
        os.mkdir(os.path.join(config.save_dir, 'predictions'))
    pred_path = os.path.join(config.save_dir, 'predictions', name)
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)

    # ------------------------------------------------------------
    # Build test subject/record list from SZ2_*_test.tsv
    # ------------------------------------------------------------
    test_pats_list = pd.read_csv(
        os.path.join('net', 'datasets', config.dataset + '_test.tsv'),
        sep='\t',
        header=None,
        skiprows=[0, 1, 2]
    )
    test_pats_list = test_pats_list[0].to_list()
    test_recs_list = [
        [s, r.split('_')[-2]]
        for s in test_pats_list
        for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg'))
        if 'edf' in r
    ]

    # ------------------------------------------------------------
    # Build & cache ONE BIG test generator (meta_test_*, gen_test_*)
    # ------------------------------------------------------------
    gen_name = f"{config.dataset}_frame-{config.frame}_sampletype-{config.sample_type}"
    generators_dir = os.path.join("net", "generators")
    os.makedirs(generators_dir, exist_ok=True)

    meta_test_path = os.path.join(generators_dir, f"meta_test_{gen_name}.pkl")
    gen_test_path  = os.path.join(generators_dir, f"gen_test_{gen_name}.pkl")

    if os.path.exists(meta_test_path) and os.path.exists(gen_test_path):
        print("Using existing meta_test / gen_test pickles for test set...")
        with open(meta_test_path, "rb") as f:
            meta_test = pickle.load(f)
        # We still rebuild the generator to be robust to code changes
        test_recs_list = meta_test["recs_list"]
        test_segments = meta_test["segments"]
    else:
        print("Creating and caching test generators (meta_test / gen_test)...")

        # One call builds segments for *all* test recordings
        test_segments = generate_data_keys_sequential(
            config, test_recs_list, verbose=False
        )

        # Lightweight metadata
        meta_test = {"recs_list": test_recs_list, "segments": test_segments}
        with open(meta_test_path, "wb") as f:
            pickle.dump(meta_test, f, pickle.HIGHEST_PROTOCOL)

        # Also cache a full generator object (mainly for debugging/backwards compatibility)
        gen_test_all_tmp = SequentialGenerator(
            config,
            test_recs_list,
            test_segments,
            batch_size=600,
            shuffle=False,
            verbose=False,
        )
        with open(gen_test_path, "wb") as f:
            pickle.dump(gen_test_all_tmp, f, pickle.HIGHEST_PROTOCOL)
        del gen_test_all_tmp
        gc.collect()

    # Rebuild ONE big test generator from metadata
    test_recs_list = meta_test["recs_list"]
    test_segments = meta_test["segments"]

    if len(test_segments) == 0:
        print("No test segments found – skipping prediction.")
        return

    gen_test_all = SequentialGenerator(
        config,
        test_recs_list,
        test_segments,
        batch_size=600,
        shuffle=False,
        verbose=False,
    )

    if len(gen_test_all) == 0:
        print("Unified test generator is empty – skipping prediction.")
        return

    # ------------------------------------------------------------
    # Build model once, load weights once, predict once
    # ------------------------------------------------------------
    model_weights_path = os.path.join(
        model_save_path, 'Weights', name + '.weights.h5'
    )

    # Reload config used for training
    config.load_config(
        config_path=os.path.join(model_save_path, 'configs'),
        config_name=name + '.cfg',
    )

    # Build same model architecture
    if config.model == 'DeepConvNet':
        from net.DeepConv_Net import net
    elif config.model == 'ChronoNet':
        from net.ChronoNet import net
    elif config.model == 'EEGnet':
        from net.EEGnet import net
    else:
        raise ValueError(f"Unknown model type: {config.model}")

    print("Running unified prediction over entire test set...")
    model = net(config)
    y_pred_all, y_true_all = predict_net(gen_test_all, model_weights_path, model)

    # Safety check
    if y_pred_all.size == 0:
        print("No predictions produced by unified test generator – aborting per-record save.")
        return

    # ------------------------------------------------------------
    # Split unified predictions back into per-record sequences
    # and save HDF5 files matching the original evaluation format.
    # ------------------------------------------------------------
    # We assume the order of samples in y_pred_all/y_true_all matches
    # the order of `test_segments` as built above.
    num_samples = len(y_pred_all)
    if num_samples < len(test_segments):
        print(
            f"Warning: number of predictions ({num_samples}) is smaller than "
            f"number of segments ({len(test_segments)}). Truncating segments list."
        )
        effective_segments = test_segments[:num_samples]
    else:
        effective_segments = test_segments

    # Build mapping from recording index -> list of sample indices
    rec_to_indices = {}
    for idx, seg in enumerate(effective_segments):
        rec_idx = int(seg[0])
        rec_to_indices.setdefault(rec_idx, []).append(idx)

    # Now save one HDF5 per recording as before
    for rec_idx, rec in enumerate(test_recs_list):
        out_file = os.path.join(
            pred_path,
            rec[0] + '_' + rec[1] + '_preds.h5'
        )

        # Get indices for this recording (may be empty)
        idxs = rec_to_indices.get(rec_idx, [])
        if len(idxs) == 0:
            print(f"No valid segments/predictions for {rec[0]} {rec[1]} – skipping file.")
            continue

        y_pred_rec = y_pred_all[idxs]
        y_true_rec = y_true_all[idxs]

        with h5py.File(out_file, 'w') as f:
            f.create_dataset('y_pred', data=y_pred_rec)
            f.create_dataset('y_true', data=y_true_rec)

    gc.collect()

#######################################################################################################################
#######################################################################################################################


def evaluate(config):

    """
    Evaluation for 3-class model (0=interictal, 1=pre-ictal, 2=ictal),
    but metrics focus ONLY on pre-ictal detection (class 1):

    - Sensitivity_preictal
    - False alarms per hour (FA/h) on interictal segments

    Ictal (class 2) samples are ignored in the metric computation.
    """

    name = config.get_name()

    pred_path = os.path.join(config.save_dir, 'predictions', name)
    pred_fs = 1  # 1 Hz, since we have one prediction per second

    thresholds = list(np.around(np.linspace(0, 1, 51), 2))
    x_plot = np.linspace(0, 200, 200)  # kept for compatibility, can be reused for FA ranges

    # Ensure results directory
    if not os.path.exists(os.path.join(config.save_dir, 'results')):
        os.mkdir(os.path.join(config.save_dir, 'results'))

    result_file = os.path.join(config.save_dir, 'results', name + '_results.h5')

    # Metric containers: lists over recordings, each element is list over thresholds
    sens_preictal_all = []
    fa_per_hour_all = []
    spec_interictal_all = []
    score_all = []

    # Optionally keep these for plotting style similar to the challenge
    sens_ovlp_plot = []
    prec_ovlp_plot = []

    pred_files = sorted(os.listdir(pred_path))

    for file in tqdm(pred_files):
        with h5py.File(os.path.join(pred_path, file), 'r') as f:
            y_pred = np.array(f['y_pred'])
            y_true = np.array(f['y_true'])

        # Convert y_true to class indices (0,1,2)
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true_cls = np.argmax(y_true, axis=1)
        else:
            y_true_cls = y_true.astype(int)

        # Sanity check: we expect y_pred to be (N, 3) for 3-class model
        if y_pred.ndim == 1:
            # Degenerate case: treat as probabilities for pre-ictal only
            y_pred = y_pred.reshape(-1, 1)

        # ------------------------------------------------------------------
        # Apply RMSA-based artifact mask (as in original code) to y_pred
        # ------------------------------------------------------------------
        rec = [file.split('_')[0], file.split('_')[1]]
        rec_data = Data.loadData(config.data_path, rec, modalities=['eeg'])

        [ch_focal, ch_cross] = apply_preprocess_eeg(config, rec_data)

        rmsa_f = [
            np.sqrt(np.mean(ch_focal[start:start + 2 * config.fs] ** 2))
            for start in range(0, len(ch_focal) - 2 * config.fs + 1, 1 * config.fs)
        ]
        rmsa_c = [
            np.sqrt(np.mean(ch_cross[start:start + 2 * config.fs] ** 2))
            for start in range(0, len(ch_focal) - 2 * config.fs + 1, 1 * config.fs)
        ]

        rmsa_f = np.array([1 if 13 < rms < 150 else 0 for rms in rmsa_f])
        rmsa_c = np.array([1 if 13 < rms < 150 else 0 for rms in rmsa_c])
        rmsa = np.logical_and(rmsa_f == 1, rmsa_c == 1)

        # Align RMSA mask length with number of predictions
        if len(y_pred) != len(rmsa):
            rmsa = rmsa[:len(y_pred)]
        if len(y_true_cls) != len(rmsa):
            y_true_cls = y_true_cls[:len(rmsa)]
            y_pred = y_pred[:len(rmsa)]

        # Mask "bad" windows (rmsa == 0) by forcing them to non-pre-ictal
        mask_bad = (rmsa == 0)
        if y_pred.ndim == 2:
            # Set all probabilities to zero → no pre-ictal alarms from those windows
            y_pred[mask_bad, :] = 0.0
        else:
            y_pred[mask_bad] = 0.0

        # ------------------------------------------------------------------
        # Extract pre-ictal probability (class index 1)
        # ------------------------------------------------------------------
        if y_pred.ndim == 2 and y_pred.shape[1] >= 2:
            p_pre = y_pred[:, 1]
        else:
            # Fallback if shape unexpected
            p_pre = y_pred.astype(float).flatten()

        # ------------------------------------------------------------------
        # Compute metrics for each threshold
        # ------------------------------------------------------------------
        sens_preictal_th = []
        fa_per_hour_th = []
        spec_interictal_th = []
        score_th = []

        for th in thresholds:
            # Binary decision: pre-ictal vs non-pre-ictal
            pred_bin = (p_pre >= th).astype(int)  # 1 = predicted pre-ictal

            # Ignore ictal (class 2) samples in the metrics
            mask_eval = (y_true_cls != 2)
            yt = y_true_cls[mask_eval]
            yp = pred_bin[mask_eval]

            if yt.size == 0:
                sens_preictal_th.append(np.nan)
                fa_per_hour_th.append(np.nan)
                spec_interictal_th.append(np.nan)
                score_th.append(np.nan)
                continue

            # 0 = interictal, 1 = pre-ictal
            TP = np.sum((yt == 1) & (yp == 1))
            FN = np.sum((yt == 1) & (yp == 0))
            FP = np.sum((yt == 0) & (yp == 1))
            TN = np.sum((yt == 0) & (yp == 0))

            # Sensitivity on pre-ictal (class 1)
            if TP + FN > 0:
                sens_pre = TP / (TP + FN)
            else:
                sens_pre = np.nan

            # False alarms per hour on interictal (class 0) windows
            n_inter = np.sum(yt == 0)
            if n_inter > 0:
                hours_inter = n_inter / (pred_fs * 3600.0)
                fa_h = FP / hours_inter
                spec_inter = TN / (TN + FP) if (TN + FP) > 0 else np.nan
            else:
                fa_h = np.nan
                spec_inter = np.nan

            sens_preictal_th.append(sens_pre)
            fa_per_hour_th.append(fa_h)
            spec_interictal_th.append(spec_inter)

            # Example composite score similar to challenge:
            if np.isfinite(sens_pre) and np.isfinite(fa_h):
                score_th.append(sens_pre * 100.0 - 0.4 * fa_h)
            else:
                score_th.append(np.nan)

        sens_preictal_all.append(sens_preictal_th)
        fa_per_hour_all.append(fa_per_hour_th)
        spec_interictal_all.append(spec_interictal_th)
        score_all.append(score_th)

        # For plotting-style arrays: treat FA/h as "x" and sensitivity as "y"
        # Using the same interpolation trick as original code
        # (optional, mostly kept for backwards compatibility with plotting)
        fa_arr = np.array(fa_per_hour_th, dtype=float)
        sens_arr = np.array(sens_preictal_th, dtype=float)

        # Only keep finite values for interpolation
        valid = np.isfinite(fa_arr) & np.isfinite(sens_arr)
        if np.any(valid):
            # Sort by FA/h
            order = np.argsort(fa_arr[valid])
            x_valid = fa_arr[valid][order]
            y_valid = sens_arr[valid][order]

            # Ensure strictly increasing x for interpolation by removing duplicates
            x_unique, idx_unique = np.unique(x_valid, return_index=True)
            y_unique = y_valid[idx_unique]

            # Clip the interpolation domain if needed
            x_min, x_max = x_unique[0], x_unique[-1]
            x_target = np.clip(x_plot, x_min, x_max)
            y_plot = np.interp(x_target, x_unique, y_unique)
        else:
            y_plot = np.full_like(x_plot, np.nan, dtype=float)

        sens_ovlp_plot.append(y_plot)
        # No precision-based curve here; just reuse sensitivity curve as a placeholder
        prec_ovlp_plot.append(y_plot.copy())

    # Report score at threshold index 25 (approx th = 0.5)
    score_05 = [x[25] for x in score_all if len(x) > 25]
    if len(score_05) > 0:
        print('Score (pre-ictal, th=0.5): ' + "%.2f" % np.nanmean(score_05))
    else:
        print('Score (pre-ictal, th=0.5): NaN')

    # Save metrics
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('sens_preictal', data=sens_preictal_all)
        f.create_dataset('fa_per_hour', data=fa_per_hour_all)
        f.create_dataset('spec_interictal', data=spec_interictal_all)
        f.create_dataset('score', data=score_all)
        f.create_dataset('thresholds', data=np.array(thresholds))

        # Optional plotting arrays (for backwards compatibility)
        f.create_dataset('sens_ovlp_plot', data=sens_ovlp_plot)
        f.create_dataset('prec_ovlp_plot', data=prec_ovlp_plot)
        f.create_dataset('x_plot', data=x_plot)


#######################################################################################################################
#######################################################################################################################


def main():
    """
    Entry point: set seeds, build Config, and run train -> predict -> evaluate.
    """
    import random

    # Reproducibility
    random_seed = 1
    random.seed(random_seed)

    import numpy as np
    np.random.seed(random_seed)

    import tensorflow as tf
    tf.random.set_seed(random_seed)

    # Also seed the key_generator module explicitly
    from net import key_generator
    key_generator.random.seed(random_seed)

    # Import configuration class
    from net.DL_config import Config

    # GPU setup
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('GPU automatically enabled.')
    else:
        print('Running on CPU.')

    ###########################################
    ## Initialize standard config parameters ##
    ###########################################

    config = Config()
    config.data_path = '/Users/rosalouisemarker/Desktop/Digital Media Project/dataset'  # path to data
    config.save_dir = 'net/save_dir'                                                   # save directory for outputs
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    config.fs = 250              # Sampling frequency after post-processing
    config.CH = 2                # Nr of EEG channels
    config.cross_validation = 'fixed'
    config.batch_size = 128
    config.frame = 2             # window size (seconds)
    config.stride = 1            # stride (background EEG, seconds)
    config.stride_s = 0.5        # stride (seizure EEG, seconds)
    config.boundary = 0.5        # proportion of seizure data to mark positive
    config.factor = 5            # class balancing factor

    # Network hyper-parameters
    config.dropoutRate = 0.5
    config.nb_epochs = 80
    config.l2 = 0.01
    config.lr = 0.01

    ###########################################
    #####q INPUT CONFIGS:
    ###########################################
    config.model = 'ChronoNet'          # model architecture
    config.dataset = 'SZ2'              # dataset split (see datasets folder)
    config.sample_type = 'subsample'    # subsample background EEG
    config.add_to_name = 'test'         # suffix for experiment name

    ###########################################
    ###########################################

    print('Training the model...')
    load_generators = True   # load generators from pkl if available
    save_generators = False  # do not resave generators in this run

    train(config, load_generators, save_generators)

    print('Getting predictions on the test set...')
    predict(config)

    print('Getting evaluation metrics...')
    evaluate(config)


if __name__ == "__main__":
    main()