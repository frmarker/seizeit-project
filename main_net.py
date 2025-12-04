# Updated main_net.py with generator caching, consistent segmentation, and lower learning rate
import os
import gc
import time
import pickle
import argparse

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

print('[main_net] File imported and top-level code reached.')

# Remap segment labels for binary task: 0 = interictal and seizure, 1 = pre-ictal
def remap_labels_to_binary_preictal(segments):

    if segments is None:
        return None
    for seg in segments:
        lbl = int(seg[3])
        seg[3] = 1 if lbl == 1 else 0
    return segments

# Helper functions to inspect label distributions in generators
def check_train_labels(train_generator, seizure_class_idx=1, max_batches=None):

    total_pos = 0
    total_neg = 0
    total_samples = 0

    num_batches = len(train_generator) if hasattr(train_generator, "__len__") else None

    for b, (_, y) in enumerate(train_generator):
        if max_batches is not None and b >= max_batches:
            break

        # If y is one-hot, convert to class indices
        labels = np.argmax(y, axis=1)

        total_pos += np.sum(labels == seizure_class_idx)
        total_neg += np.sum(labels != seizure_class_idx)
        total_samples += labels.shape[0]

        if num_batches is not None and (b + 1) >= num_batches:
            break

    print('=== Training generator label check ===')
    print(f'Total windows in train: {total_samples}')
    print(f'Seizure windows (class {seizure_class_idx}): {total_pos}')
    print(f'Non-seizure windows: {total_neg}')
    if total_samples > 0:
        print(f'Seizure fraction: {total_pos / total_samples:.4f}')


def check_val_labels(val_generator, seizure_class_idx=1, max_batches=None):

    total_pos = 0
    total_neg = 0
    total_samples = 0

    num_batches = len(val_generator) if hasattr(val_generator, "__len__") else None

    for b, (_, y) in enumerate(val_generator):
        if max_batches is not None and b >= max_batches:
            break

        # If y is one-hot, convert to class indices
        labels = np.argmax(y, axis=1)

        total_pos += np.sum(labels == seizure_class_idx)
        total_neg += np.sum(labels != seizure_class_idx)
        total_samples += labels.shape[0]

        if num_batches is not None and (b + 1) >= num_batches:
            break

    print('=== Validation generator label check ===')
    print(f'Total windows in val: {total_samples}')
    print(f' Seizure windows (class {seizure_class_idx}): {total_pos}')
    print(f' Non-seizure windows: {total_neg}')
    if total_samples > 0:
        print(f'  Seizure fraction: {total_pos / total_samples:.4f}')

# Helper function to plot EEG with event boundaries and 30s pre-ictal window
def plot_eeg_with_events_and_preictal(config, recs_list, segments, target_rec_idx, pre_len=30.0):

    import matplotlib
    matplotlib.use('Agg')  # headless backend
    import matplotlib.pyplot as plt

    # Get recording ID and load raw EEG
    rec = recs_list[target_rec_idx]
    rec_data = Data.loadData(config.data_path, rec, modalities=['eeg'])

    # Apply standard preprocessing to obtain focal / cross channels
    try:
        ch_focal, ch_cross = apply_preprocess_eeg(config, rec_data)
    except Exception as e:
        print(f'Could not preprocess EEG for plot ({rec[0]} {rec[1]}): {e}')
        return

    fs = config.fs
    n_samples = len(ch_focal)
    if n_samples == 0:
        print(f'No samples in focal channel for {rec[0]} {rec[1]} - skipping plot.')
        return

    t = np.arange(n_samples) / fs

    # Collect ictal segments for this recording
    ictal_segs = [s for s in segments if int(s[0]) == int(target_rec_idx) and int(s[3]) == 2]
    if not ictal_segs:
        print(f'No ictal segments for {rec[0]} {rec[1]} - nothing to plot.')
        return

    seizure_start = min(s[1] for s in ictal_segs)
    seizure_end = max(s[2] for s in ictal_segs)
    pre_start = max(seizure_start - pre_len, 0.0)

    # Select a plotting window around the seizure
    window_start = max(pre_start - 30.0, 0.0)  # 30s padding before pre-ictal
    window_end   = min(seizure_end + 30.0, t[-1])  # 30s padding after seizure

    start_idx = int(window_start * fs)
    end_idx   = int(window_end * fs)

    t_win        = t[start_idx:end_idx]
    ch_focal_win = ch_focal[start_idx:end_idx]
    ch_cross_win = ch_cross[start_idx:end_idx]

    # Create plot
    plt.figure(figsize = (12, 6))
    plt.plot(t_win, ch_focal_win, label='Focal channel')
    plt.plot(t_win, ch_cross_win, label='Cross channel', alpha=0.7)

    # Shade pre-ictal and ictal windows (in absolute time coordinates)
    plt.axvspan(pre_start, seizure_start, alpha = 0.3, label='Pre-ictal window')
    plt.axvspan(seizure_start, seizure_end, alpha = 0.3, label='Ictal period')

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude [a.u.]')
    plt.title(f'EEG with events and 30s pre-ictal window: {rec[0]} {rec[1]}')
    plt.legend(loc = 'upper right')
    plt.tight_layout()

    plots_dir = os.path.join(config.save_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    out_name = f"{rec[0]}_{rec[1]}_preictal_plot.png"
    out_path = os.path.join(plots_dir, out_name)
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f'Saved EEG + event + pre-ictal plot to {out_path}')


def train(config, load_generators, save_generators):

    print(f'[train] Starting train() for exp name: {config.get_name()}')

    name = config.get_name()

    # Select model (ChronoNet ONLY)
    from net.ChronoNet import net
    # Force config.model to ChronoNet for consistency
    config.model = 'ChronoNet'

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

    # Train/val/test
    if config.dataset == 'SZ2':
            # Build subject splits directly from the dataset directory,
            # avoiding any dependency on SZ2_training.tsv / SZ2_validation.tsv.
            print('[train] Building subject splits directly from dataset folder (ignoring TSVs)...', flush=True)

            # Discover all subjects that look like "sub-XXX" and have an EEG folder
            all_subs = []
            for d in sorted(os.listdir(config.data_path)):
                if not d.startswith("sub-"):
                    continue
                eeg_dir = os.path.join(config.data_path, d, "ses-01", "eeg")
                if os.path.isdir(eeg_dir):
                    all_subs.append(d)

            if not all_subs:
                print('[train][ERROR] No usable subjects found in data_path; aborting.')
                return

            n_total = len(all_subs)
            print(f'[train] Found {n_total} subjects with EEG data: {all_subs}')

            # Simple deterministic split: 70% train, 15% val, 15% test
            n_train = max(1, int(0.7 * n_total))
            n_val   = max(1, int(0.15 * n_total))
            if n_train + n_val >= n_total:
                # Ensure at least one test subject if possible
                n_val = max(1, n_total - n_train - 1)

            train_pats_list = all_subs[:n_train]
            val_pats_list   = all_subs[n_train:n_train + n_val]
            test_pats_list  = all_subs[n_train + n_val:]

            print(f'[train] Train subjects ({len(train_pats_list)}): {train_pats_list}')
            print(f'[train] Val subjects   ({len(val_pats_list)}): {val_pats_list}')
            print(f'[train] Test subjects  ({len(test_pats_list)}): {test_pats_list}')

            # Persist this split so predict() can reuse it later
            split_path = os.path.join("net", "datasets", "SZ2_subject_split_auto.pkl")
            try:
                with open(split_path, "wb") as f:
                    pickle.dump(
                        {
                            "train": train_pats_list,
                            "val": val_pats_list,
                            "test": test_pats_list,
                        },
                        f,
                        pickle.HIGHEST_PROTOCOL,
                    )
                print(f"[train] Saved automatic subject split to {split_path}")
            except Exception as e:
                print(f"[train][WARNING] Could not save subject split pickle: {e}")


            # Build list of training recordings, all runs for train subjects
            print('[train] Building train_recs_list from EDF files...')
            train_recs_list = [
                [s, r.split("_")[-2]]
                for s in train_pats_list
                for r in os.listdir(os.path.join(config.data_path, s, "ses-01", "eeg"))
                if 'edf' in r
            ]

            print(f'[train] Total training recordings: {len(train_recs_list)}')

            # Names/paths for generator metadata (segments) and generator pickles
            gen_name = f"{config.dataset}_frame-{config.frame}_sampletype-{config.sample_type}"
            generators_dir = os.path.join('net', 'generators')
            os.makedirs(generators_dir, exist_ok=True)

            meta_train_path = os.path.join(generators_dir, f"meta_train_{gen_name}.pkl")
            meta_val_path   = os.path.join(generators_dir, f"meta_val_{gen_name}.pkl")
            gen_train_path  = os.path.join(generators_dir, f"gen_train_{gen_name}.pkl")
            gen_val_path    = os.path.join(generators_dir, f"gen_val_{gen_name}.pkl")

            # We optionally reuse cached segment metadata (recs_list + segments)
            # to avoid recomputing keys; generators themselves can also be cached.
            train_segments = None
            val_segments = None
            val_recs_list = None

            # Try to reuse cached segment metadata if it exists and is well-formed
            meta_train = None
            meta_val = None
            if os.path.exists(meta_train_path) and os.path.exists(meta_val_path):
                try:
                    with open(meta_train_path, 'rb') as f:
                        meta_train = pickle.load(f)
                    with open(meta_val_path, 'rb') as f:
                        meta_val = pickle.load(f)

                    if (
                        isinstance(meta_train, dict)
                        and isinstance(meta_val, dict)
                        and 'recs_list' in meta_train
                        and 'segments' in meta_train
                        and 'recs_list' in meta_val
                        and 'segments' in meta_val
                    ):
                        # Reuse cached segment keys
                        train_recs_list = meta_train['recs_list']
                        train_segments = meta_train['segments']

                        val_recs_list = meta_val['recs_list']
                        val_segments = meta_val['segments']
                        print(f'Loaded cached segment metadata from meta_* pickles for {gen_name}.')
                    else:
                        print('Cached meta_* pickles are not in the expected format – ignoring cache.')
                        meta_train = None
                        meta_val = None
                except Exception as e:
                    print(f'Error loading meta_* pickles ({e}) – regenerating segment metadata from raw data.')
                    meta_train = None
                    meta_val = None

            # If no valid cached metadata, (re)generate segment keys from raw data
            if train_segments is None or val_segments is None or val_recs_list is None:
                print('Generating training segments from raw data...')
                if config.sample_type == 'subsample':
                    train_segments = generate_data_keys_subsample(config, train_recs_list)
                else:
                    train_segments = generate_data_keys_sequential(config, train_recs_list)

                print('Building validation recording list from automatic subject split...')
                # val_pats_list was already computed above from all_subs
                val_recs_list = [
                    [s, r.split('_')[-2]]
                    for s in val_pats_list
                    for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg'))
                    if 'edf' in r
                ]

                # Use the same window style as training for validation
                print('Generating validation segments from raw data...')
                if config.sample_type == 'subsample':
                    val_segments = generate_data_keys_subsample(config, val_recs_list)
                else:
                    val_segments = generate_data_keys_sequential(config, val_recs_list)

                    print('Saving segment metadata (meta_train_*, meta_val_*) for future runs...')
                    meta_train = {"recs_list": train_recs_list, "segments": train_segments}
                    meta_val   = {"recs_list": val_recs_list, "segments": val_segments}
                    with open(meta_train_path, 'wb') as f:
                        pickle.dump(meta_train, f, pickle.HIGHEST_PROTOCOL)
                    with open(meta_val_path, 'wb') as f:
                        pickle.dump(meta_val, f, pickle.HIGHEST_PROTOCOL)
                    # Invalidate any stale generator pickles since segments changed
                    for p in [gen_train_path, gen_val_path]:
                        if os.path.exists(p):
                            os.remove(p)

            # Enforce binary labeling: 0=non-pre-ictal, 1=pre-ictal
            # (ictal segments are collapsed into class 0)
            train_segments = remap_labels_to_binary_preictal(train_segments)
            val_segments   = remap_labels_to_binary_preictal(val_segments)

            # Auto-check label distribution and (optionally) oversample pre-ictal (class 1)
            # if it is severely under-represented compared to interictal (class 0).
            # Currently oversampling is disabled (line left commented out).
            try:
                labels = np.array([int(seg[3]) for seg in train_segments])
                unique, counts = np.unique(labels, return_counts=True)
                print('Training label distribution before oversampling '
                      '(0=interictal,1=seizure/pre-ictal):')
                for u, c in zip(unique, counts):
                    print(f'  class {u}: {c} samples')

                n0 = counts[unique == 0][0] if np.any(unique == 0) else 0
                n1 = counts[unique == 1][0] if np.any(unique == 1) else 0

                # Target: roughly 30% as many pre-ictal as interictal samples,
                # with a maximum duplication factor of 5x to avoid extremes.
                if n0 > 0 and n1 > 0:
                    target_n1 = 0.3 * n0
                    if n1 < target_n1:
                        factor = int(np.clip(np.ceil(target_n1 / n1), 2, 5))
                        pre_segments = [s for s in train_segments if int(s[3]) == 1]
                        print(
                            f'Auto-oversampling pre-ictal (class 1) by factor {factor} '
                            f'(original n1={n1}, target≈{int(target_n1)})'
                        )
                        # To enable oversampling, uncomment the next line:
                        # train_segments = train_segments + pre_segments * (factor - 1)

                        # Recompute and print distribution after oversampling (currently unchanged)
                        labels = np.array([int(seg[3]) for seg in train_segments])
                        unique, counts = np.unique(labels, return_counts=True)
                        print('Training label distribution after oversampling (effective):')
                        for u, c in zip(unique, counts):
                            print(f'class {u}: {c} samples')
                    else:
                        print('Pre-ictal class proportion is acceptable; no oversampling applied.')
                else:
                    print('Not enough labeled data to compute class 0/1 distribution; '
                          'skipping auto-oversampling.')
            except Exception as e:
                print(f'[Warning] Could not auto-oversample pre-ictal segments: {e}')

            # Instantiate or load training generator
            if load_generators and os.path.exists(gen_train_path):
                print('Loading cached training generator from', gen_train_path)
                with open(gen_train_path, 'rb') as f:
                    gen_train = pickle.load(f)
            else:
                print('Instantiating training generator...')
                gen_train = SegmentedGenerator(config, train_recs_list, train_segments, batch_size = config.batch_size, shuffle = True,)
                if save_generators:
                    print('Saving training generator cache to', gen_train_path)
                    with open(gen_train_path, 'wb') as f:
                        pickle.dump(gen_train, f, pickle.HIGHEST_PROTOCOL)

            # Debug: inspect the label distribution over the entire training generator
            try:
                print('Checking training label distribution...')
                # Use max_batches to limit runtime if desired (e.g., 5); None = all batches
                check_train_labels(gen_train, seizure_class_idx=1, max_batches=5)
            except Exception as e:
                print(f'[Warning] Could not check training labels: {e}')

            # Instantiate or load validation generator
            if load_generators and os.path.exists(gen_val_path):
                print('Loading cached validation generator from', gen_val_path)
                with open(gen_val_path, 'rb') as f:
                    gen_val = pickle.load(f)
            else:
                print('Instantiating validation generator...')
                gen_val = SequentialGenerator(config, val_recs_list, val_segments, batch_size = 600, shuffle = False,)
                if save_generators:
                    print('Saving validation generator cache to', gen_val_path)
                    with open(gen_val_path, 'wb') as f:
                        pickle.dump(gen_val, f, pickle.HIGHEST_PROTOCOL)

            # Debug: check that validation labels actually contain seizures
            try:
                print('Checking validation label distribution...')
                check_val_labels(gen_val, seizure_class_idx = 1, max_batches = None)
            except Exception as e:
                print(f'[Warning] Could not check validation labels: {e}')

            # Optional: plot EEG with event boundaries and 30s pre-ictal window
            try:
                ictal_rec_idxs = sorted({int(seg[0]) for seg in val_segments if int(seg[3]) == 2})
                if ictal_rec_idxs:
                    target_rec_idx = ictal_rec_idxs[0]
                    plot_eeg_with_events_and_preictal(config, val_recs_list, val_segments, target_rec_idx, pre_len = 30.0,)
                else:
                    print('No ictal segments found in validation set - skipping EEG/pre-ictal plot.')
            except Exception as e:
                print(f'Could not generate EEG/pre-ictal plot: {e}')

            # Build ChronoNet model
            print('Training model....')
            model = net(config)

            start_train = time.time()
            train_net(config, model, gen_train, gen_val, model_save_path)
            end_train = time.time() - start_train
            print('Total train duration = ', end_train / 60)


def predict(config): 
    # Predict routine
    print('[predict] Entering predict()')
    # Select model (ChronoNet ONLY)
    name = config.get_name()
    # Force config.model to ChronoNet for consistency
    model_save_path = os.path.join(config.save_dir, 'models', name)

    # Ensure prediction directories
    if not os.path.exists(os.path.join(config.save_dir, 'predictions')):
        os.mkdir(os.path.join(config.save_dir, 'predictions'))
    pred_path = os.path.join(config.save_dir, 'predictions', name)
    if not os.path.exists(pred_path):
        os.mkdir(pred_path)

    # Build test subject/record list
    # Prefer the automatic subject split created during training;
    # fall back to SZ2_test.tsv only if needed.
    split_path = os.path.join("net", "datasets", "SZ2_subject_split_auto.pkl")
    test_pats_list = None

    if os.path.exists(split_path):
        try:
            with open(split_path, "rb") as f:
                split = pickle.load(f)
            if isinstance(split, dict) and 'test' in split:
                test_pats_list = split['test']
                print(f'[predict] Loaded test subjects from {split_path}: {test_pats_list}')
        except Exception as e:
            print(f'[predict][WARNING] Could not load subject split pickle ({e}); falling back to TSV.')

    if test_pats_list is None:
        print('[predict] Falling back to SZ2_test.tsv for test subject list...')
        test_pats_list = pd.read_csv(
            os.path.join("net", "datasets", config.dataset + "_test.tsv"),
            sep="\t",
            header=None,
            skiprows=[0, 1, 2],
        )[0].to_list()
    test_recs_list = [
        [s, r.split('_')[-2]]
        for s in test_pats_list
        for r in os.listdir(os.path.join(config.data_path, s, 'ses-01', 'eeg'))
        if 'edf' in r
    ]

    # Build & cache ONE BIG test generator (meta_test_*, gen_test_*)
    gen_name = f"{config.dataset}_frame-{config.frame}_sampletype-{config.sample_type}"
    generators_dir = os.path.join("net", "generators")
    os.makedirs(generators_dir, exist_ok=True)

    meta_test_path = os.path.join(generators_dir, f"meta_test_{gen_name}.pkl")
    gen_test_path  = os.path.join(generators_dir, f"gen_test_{gen_name}.pkl")

    if os.path.exists(meta_test_path) and os.path.exists(gen_test_path):
        print('Using existing meta_test / gen_test pickles for test set...')
        with open(meta_test_path, "rb") as f:
            meta_test = pickle.load(f)
        with open(gen_test_path, "rb") as f:
            gen_test_all = pickle.load(f)
        test_recs_list = meta_test['recs_list']
        test_segments = meta_test['segments']
    else:
        print('Creating and caching test generators (meta_test / gen_test)...')

        # One call builds segments for *all* test recordings
        test_segments = generate_data_keys_sequential(
            config, test_recs_list, verbose=False
        )

        # Lightweight metadata
        meta_test = {'recs_list': test_recs_list, 'segments': test_segments}
        with open(meta_test_path, "wb") as f:
            pickle.dump(meta_test, f, pickle.HIGHEST_PROTOCOL)

        # Also cache a full generator object
        gen_test_all = SequentialGenerator(config, test_recs_list, test_segments, batch_size = 600, shuffle = False, verbose = False,)
        with open(gen_test_path, "wb") as f:
            pickle.dump(gen_test_all, f, pickle.HIGHEST_PROTOCOL)

    # Safety checks
    if meta_test is None:
        print('No meta_test information found - skipping prediction.')
        return

    test_recs_list = meta_test['recs_list']
    test_segments = meta_test['segments']
    # Enforce binary labeling (0=non-pre-ictal,1=pre-ictal) for test set
    test_segments = remap_labels_to_binary_preictal(test_segments)
    meta_test['segments'] = test_segments

    if len(test_segments) == 0:
        print('No test segments found - skipping prediction.')
        return

    if gen_test_all is None:
        # Rebuild generator from metadata if pickle missing or corrupted
        print("Test generator pickle missing or invalid - rebuilding SequentialGenerator...")
        gen_test_all = SequentialGenerator(
            config,
            test_recs_list,
            test_segments,
            batch_size=600,
            shuffle=False,
            verbose=False,
        )

    if len(gen_test_all) == 0:
        print('Unified test generator is empty - skipping prediction.')
        return

    # Build model once, load weights once, predict once
    model_weights_path = os.path.join(
        model_save_path, 'Weights', name + '.weights.h5'
    )

    # Reload config used for training
    config.load_config(
        config_path=os.path.join(model_save_path, 'configs'),
        config_name=name + '.cfg',
    )

    # Build same model architecture 
    from net.ChronoNet import net
    config.model = 'ChronoNet'

    print("Running unified prediction over entire test set...")
    model = net(config)
    y_pred_all, y_true_all = predict_net(gen_test_all, model_weights_path, model)

    # Safety check
    if y_pred_all.size == 0:
        print('No predictions produced by unified test generator - aborting per-record save.')
        return

    # Split unified predictions back into per-record sequences
    # and save HDF5 files matching the original evaluation format.
    # We assume the order of samples in y_pred_all/y_true_all matches
    # the order of `test_segments` as built above.
    num_samples = len(y_pred_all)
    if num_samples < len(test_segments):
        print(
            f'Warning: number of predictions ({num_samples}) is smaller than '
            f'number of segments ({len(test_segments)}). Truncating segments list.'
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
            print(f'No valid segments/predictions for {rec[0]} {rec[1]} – skipping file.')
            continue

        y_pred_rec = y_pred_all[idxs]
        y_true_rec = y_true_all[idxs]

        with h5py.File(out_file, 'w') as f:
            f.create_dataset('y_pred', data=y_pred_rec)
            f.create_dataset('y_true', data=y_true_rec)

    gc.collect()


def evaluate(config):

    print('[evaluate] Entering evaluate()')

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

    pred_files = sorted(f for f in os.listdir(pred_path) if f.endswith('.h5'))

    # Global confusion matrix accumulators at threshold 0.2 (index 25)
    TP_global = 0
    FP_global = 0
    TN_global = 0
    FN_global = 0

    for file in tqdm(pred_files):
        with h5py.File(os.path.join(pred_path, file), 'r') as f:
            y_pred = np.array(f['y_pred'])
            y_true = np.array(f['y_true'])
            # Skip files with completely empty predictions/labels to avoid
            # downstream shape issues or stalling on RMSA computation.
            if y_pred.size == 0 or y_true.size == 0:
                print(f'[Warning] Empty y_pred or y_true in {file} – skipping from evaluation.')
                sens_preictal_all.append([np.nan] * len(thresholds))
                fa_per_hour_all.append([np.nan] * len(thresholds))
                spec_interictal_all.append([np.nan] * len(thresholds))
                score_all.append([np.nan] * len(thresholds))
                sens_ovlp_plot.append(np.full_like(x_plot, np.nan, dtype=float))
                prec_ovlp_plot.append(np.full_like(x_plot, np.nan, dtype=float))
                continue

        # Convert y_true to class indices (0,1,2)
        if y_true.ndim == 2 and y_true.shape[1] > 1:
            y_true_cls = np.argmax(y_true, axis=1)
        else:
            y_true_cls = y_true.astype(int)

        # Sanity check: we expect y_pred to be (N, 3) for 3-class model
        if y_pred.ndim == 1:
            # Degenerate case: treat as probabilities for pre-ictal only
            y_pred = y_pred.reshape(-1, 1)

        # Apply RMSA-based artifact mask (as in original code) to y_pred
        rec = [file.split('_')[0], file.split('_')[1]]
        rec_data = Data.loadData(config.data_path, rec, modalities=['eeg'])

        # Gracefully skip empty Data objects (missing EEG file) instead of raising
        try:
            [ch_focal, ch_cross] = apply_preprocess_eeg(config, rec_data)
        except ValueError:
            # Append NaN arrays to keep positional alignment
            sens_preictal_all.append([np.nan]*len(thresholds))
            fa_per_hour_all.append([np.nan]*len(thresholds))
            spec_interictal_all.append([np.nan]*len(thresholds))
            score_all.append([np.nan]*len(thresholds))
            sens_ovlp_plot.append(np.full_like(x_plot, np.nan, dtype=float))
            prec_ovlp_plot.append(np.full_like(x_plot, np.nan, dtype=float))
            continue

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

        # Extract pre-ictal probability (class index 1)
        if y_pred.ndim == 2 and y_pred.shape[1] >= 2:
            p_pre = y_pred[:, 1]
        else:
            # Fallback if shape unexpected
            p_pre = y_pred.astype(float).flatten()

        # Compute metrics for each threshold
        sens_preictal_th = []
        fa_per_hour_th = []
        spec_interictal_th = []
        score_th = []

        for th in thresholds:
            # Binary decision: pre-ictal vs non-pre-ictal
            pred_bin = (p_pre >= th).astype(int)  # 1 = predicted pre-ictal

            # All samples are either 0 (non-pre-ictal) or 1 (pre-ictal)
            yt = y_true_cls
            yp = pred_bin

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

            # Aggregate confusion matrix counts for threshold ~0.2
            if th == 0.2:
                TP_global += TP
                FP_global += FP
                TN_global += TN
                FN_global += FN

        # After computing all thresholds for this recording, append metric lists once
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

    # Report score at threshold index 25 (approx th = 0.2)
    score_05 = [x[25] for x in score_all if len(x) > 25]
    if len(score_05) > 0:
        print('Score (pre-ictal, th=0.2): ' + '%.2f' % np.nanmean(score_05))
    else:
        print('Score (pre-ictal, th=0.2): NaN')

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

        # Confusion matrix at threshold 0.2
        cm = np.array([[TP_global, FP_global], [FN_global, TN_global]])
        f.create_dataset('confusion_matrix_th02', data=cm)

    # Print confusion matrix & derived metrics
    if TP_global + FP_global + TN_global + FN_global > 0:
        precision = TP_global / (TP_global + FP_global) if (TP_global + FP_global) > 0 else float('nan')
        recall = TP_global / (TP_global + FN_global) if (TP_global + FN_global) > 0 else float('nan')
        specificity = TN_global / (TN_global + FP_global) if (TN_global + FP_global) > 0 else float('nan')
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else float('nan'))
        print('\nConfusion matrix (threshold=0.2, binary event-related vs inter-ictal):')
        print('Pred Pre  Pred Non')
        print(f'True Pre:   {TP_global:5d}    {FN_global:5d}')
        print(f'True Non:   {FP_global:5d}    {TN_global:5d}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'Specificity: {specificity:.4f}')
        print(f'F1: {f1:.4f}')

    # Plot training curves (loss, accuracy, AUC) if history CSV exists
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')  # headless
        import matplotlib.pyplot as plt

        history_csv = os.path.join(config.save_dir, 'models', name, 'History', name + '.csv')
        if os.path.exists(history_csv):
            df = pd.read_csv(history_csv)
            curves_dir = os.path.join(config.save_dir, 'results')
            # Loss plot
            plt.figure(figsize=(10,5))
            if 'loss' in df.columns and 'val_loss' in df.columns:
                plt.plot(df['loss'], label='Train Loss')
                plt.plot(df['val_loss'], label='Val Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss Curve')
            plt.legend(); plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(curves_dir, 'loss_curve.png'), dpi=150)
            plt.close()

            # Accuracy plot
            if 'accuracy' in df.columns and 'val_accuracy' in df.columns:
                plt.figure(figsize=(10,5))
                plt.plot(df['accuracy'], label='Train Acc')
                plt.plot(df['val_accuracy'], label='Val Acc')
                plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve')
                plt.legend(); plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(curves_dir, 'accuracy_curve.png'), dpi=150)
                plt.close()

            # AUC plot
            if 'auc' in df.columns and 'val_auc' in df.columns:
                plt.figure(figsize=(10,5))
                plt.plot(df['auc'], label='Train AUC')
                plt.plot(df['val_auc'], label='Val AUC')
                plt.xlabel('Epoch'); plt.ylabel('AUC'); plt.title('AUC Curve')
                plt.legend(); plt.grid(True); plt.tight_layout()
                plt.savefig(os.path.join(curves_dir, 'auc_curve.png'), dpi=150)
                plt.close()
            print('Saved training curves (loss_curve.png, accuracy_curve.png, auc_curve.png) to results directory.')
        else:
            print('History CSV not found; skipping curve plotting.')
    except Exception as e:
        print(f'Could not plot training curves: {e}')



def main():
    print('[main_net] Entering main()')
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

    # Import configuration class (robust to different class names in DL_config)
    from net import DL_config as dlc
    Config = getattr(dlc, "Config", None)
    if Config is None:
        Config = getattr(dlc, "DL_Config", None)
    if Config is None:
        raise ImportError('Could not find \'Config\' or \'DL_Config\' in net.DL_config')

    # GPU setup
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        for gpu in physical_gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('GPU automatically enabled.')
    else:
        print('Running on CPU.')

    # Select experiment profile and basic overrides
    parser = argparse.ArgumentParser(description='Seizure detection experiments')
    parser.add_argument('--exp', choices=['eeg', 'eeg_hrv', 'hrv'], default='eeg',
                        help='Which experiment profile to run')
    parser.add_argument('--data-path', default='/Users/rosalouisemarker/Desktop/Digital Media Project/dataset',
                        help='Path to SZ2 BIDS root')
    parser.add_argument('--save-dir', default='net/save_dir',
                        help='Base directory for saving models/results')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')

    args = parser.parse_args()


    # Initialize standard config parameters 
    config = Config()
    # Basic paths from CLI
    config.data_path = args.data_path
    config.save_dir = args.save_dir
    if not os.path.exists(config.save_dir):
        os.mkdir(config.save_dir)

    # Restrict to EEG + ECG data only
    config.modalities = ['eeg', 'ecg']

    # Use EEG+HRV input mode by default
    config.input_mode = 'eeg_hrv'

    # Core data/segmentation parameters
    config.fs = 250              
    config.CH = 2               
    config.cross_validation = 'fixed'
    config.batch_size = args.batch_size if args.batch_size is not None else 128
    config.frame = 2            
    config.stride = 1           
    config.stride_s = 0.5     
    config.boundary = 0.5      
    config.factor = 5           

    # Network hyper-parameters
    config.dropoutRate = 0.5
    config.nb_epochs = 5
    config.l2 = 0.01
    config.lr = 0.001  

    # Input / experiment configs
    config.model = 'ChronoNet'       
    config.dataset = 'SZ2'            
    config.sample_type = 'sequential'  

    if hasattr(config, 'apply_experiment_profile'):
        config.apply_experiment_profile(args.exp)
    else:
        # Fallback: configure input_mode directly based on --exp
        if args.exp == 'eeg':
            config.input_mode = 'eeg'
        elif args.exp == 'eeg_hrv':
            config.input_mode = 'eeg_hrv'  # EEG + HRV combined or multi-branch
        elif args.exp == 'hrv':
            config.input_mode = 'hrv'      # HRV-only model
        # Use experiment name as suffix for saving
        config.add_to_name = args.exp

    print(f'Running experiment: {args.exp} (input_mode={getattr(config, "input_mode", "eeg")})')
    print(f'Data path: {config.data_path}')
    print(f'Save dir : {config.save_dir}')

    # Train the model
    print('Training the model...')
    # Enable caching of generators by default:
    load_generators = False    
    save_generators = True   

    train(config, load_generators, save_generators)

    print('Getting predictions on the test set...')
    predict(config)

    print('Getting evaluation metrics...')
    evaluate(config)


if __name__ == "__main__":
    main()