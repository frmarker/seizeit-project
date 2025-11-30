import os
import gc
import time
import pickle

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

import tensorflow as tf

from net.key_generator import (
    generate_data_keys_sequential,
    generate_data_keys_subsample,
    generate_data_keys_sequential_window,
)
from net.generator_ds import SegmentedGenerator, SequentialGenerator
from net.routines import train_net, predict_net
from net.utils import apply_preprocess_eeg, get_metrics_scoring
from classes.data import Data


# -------------------------------------------------------
# Relabel segments: 3 classes (0 = interictal, 1 = pre-ictal, 2 = ictal)
# -------------------------------------------------------
def relabel_segments_to_threeclass(segments, pre_len=30.0):
    """
    Convert seizure labels (1 = ictal, 0 = non-seizure) into:
        0 = interictal
        1 = pre-ictal (pre_len seconds before ictal onset)
        2 = ictal

    segments: list of [rec_idx, start_s, stop_s, label]
    pre_len: pre-ictal window length in seconds
    """
    segs = [list(s) for s in segments]
    by_rec = {}

    for s in segs:
        rec_idx = int(s[0])
        by_rec.setdefault(rec_idx, []).append(s)

    new_segments = []

    for rec_idx, rec_segs in by_rec.items():
        rec_segs.sort(key=lambda x: x[1])
        n = len(rec_segs)
        i = 0

        # initialize all non-seizure as interictal (0)
        for s in rec_segs:
            if s[3] == 0:
                s[3] = 0  # interictal initial

        while i < n:
            if rec_segs[i][3] == 1:  # ictal
                seiz_start = rec_segs[i][1]

                j = i + 1
                while j < n and rec_segs[j][3] == 1:
                    j += 1
                seiz_end = rec_segs[j - 1][2]

                pre_start = max(0.0, seiz_start - pre_len)

                # relabel segments in this recording
                for k in range(n):
                    seg = rec_segs[k]
                    start = seg[1]
                    stop = seg[2]
                    center = 0.5 * (start + stop)

                    if seiz_start <= center <= seiz_end:
                        seg[3] = 2  # ictal
                    elif pre_start <= center < seiz_start and seg[3] == 0:
                        seg[3] = 1  # pre-ictal
                    # else keep existing (0 or possibly already 1 from other seizures)

                i = j
            else:
                i += 1

        new_segments.extend(rec_segs)

    return new_segments


# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------
def train(config, load_generators: bool, save_generators: bool):
    """
    Main training routine.
    """
    name = config.get_name()

    if config.model == "ChronoNet":
        from net.ChronoNet import net
    elif config.model == "EEGnet":
        from net.EEGnet import net
    elif config.model == "DeepConvNet":
        from net.DeepConv_Net import net
    else:
        raise ValueError(f"Unknown model: {config.model}")

    models_root = os.path.join(config.save_dir, "models")
    os.makedirs(models_root, exist_ok=True)

    model_save_path = os.path.join(models_root, name)
    os.makedirs(model_save_path, exist_ok=True)

    config_path = os.path.join(model_save_path, "configs")
    os.makedirs(config_path, exist_ok=True)

    config.save_config(save_path=config_path)

    # We keep your "fixed" split behaviour, but 3-class relabel.
    if config.cross_validation == "fixed" and config.dataset == "SZ2":
        # --- training subjects ---
        train_pats_list = pd.read_csv(
            os.path.join("net", "datasets", "SZ2_training.tsv"),
            sep="\t",
            header=None,
            skiprows=[0, 1, 2],
        )[0].to_list()

        train_recs_list = [
            [s, r.split("_")[-2]]
            for s in train_pats_list
            for r in os.listdir(os.path.join(config.data_path, s, "ses-01", "eeg"))
            if "edf" in r
        ]

        # --- validation subjects ---
        val_pats_list = pd.read_csv(
            os.path.join("net", "datasets", "SZ2_validation.tsv"),
            sep="\t",
            header=None,
            skiprows=[0, 1, 2],
        )[0].to_list()

        val_recs_list = [
            [s, r.split("_")[-2]]
            for s in val_pats_list
            for r in os.listdir(os.path.join(config.data_path, s, "ses-01", "eeg"))
            if "edf" in r
        ]

        gen_name = f"{config.dataset}_frame-{config.frame}_sampletype-{config.sample_type}"
        gen_dir = os.path.join("net", "generators")
        os.makedirs(gen_dir, exist_ok=True)

        meta_train_path = os.path.join(gen_dir, f"meta_train_{gen_name}.pkl")
        meta_val_path = os.path.join(gen_dir, f"meta_val_{gen_name}.pkl")
        gen_train_path = os.path.join(gen_dir, f"gen_train_{gen_name}.pkl")
        gen_val_path = os.path.join(gen_dir, f"gen_val_{gen_name}.pkl")

        # ----------------------------------------------------------
        # A) LOAD EXISTING GENERATORS / METADATA
        # ----------------------------------------------------------
        if load_generators and os.path.exists(meta_train_path) and os.path.exists(
            meta_val_path
        ):
            print(f"Loading generator metadata from meta_* pickles for {gen_name}.")

            with open(meta_train_path, "rb") as f:
                meta_train = pickle.load(f)
            with open(meta_val_path, "rb") as f:
                meta_val = pickle.load(f)

            train_recs_list = meta_train["recs_list"]
            train_segments = meta_train["segments"]
            val_recs_list = meta_val["recs_list"]
            val_segments = meta_val["segments"]

            # 3-class relabel
            train_segments = relabel_segments_to_threeclass(
                train_segments, pre_len=30.0
            )
            val_segments = relabel_segments_to_threeclass(
                val_segments, pre_len=30.0
            )

            gen_train = SegmentedGenerator(
                config,
                train_recs_list,
                train_segments,
                batch_size=config.batch_size,
                shuffle=True,
            )
            gen_val = SequentialGenerator(
                config,
                val_recs_list,
                val_segments,
                batch_size=600,
                shuffle=False,
            )

        elif load_generators and os.path.exists(gen_train_path) and os.path.exists(
            gen_val_path
        ):
            print(f"Loading full generator objects from gen_* pickles for {gen_name}.")

            with open(gen_train_path, "rb") as f:
                gen_train = pickle.load(f)
            with open(gen_val_path, "rb") as f:
                gen_val = pickle.load(f)

        else:
            # ------------------------------------------------------
            # B) BUILD NEW GENERATORS FROM SCRATCH
            # ------------------------------------------------------
            print("Generating training segments...")
            if config.sample_type == "subsample":
                train_segments = generate_data_keys_subsample(
                    config, train_recs_list
                )
            else:
                raise NotImplementedError(
                    f"sample_type {config.sample_type} not implemented."
                )

            # 3-class relabel
            train_segments = relabel_segments_to_threeclass(
                train_segments, pre_len=30.0
            )

            gen_train = SegmentedGenerator(
                config,
                train_recs_list,
                train_segments,
                batch_size=config.batch_size,
                shuffle=True,
            )

            # Validation segments
            print("Generating validation segments...")
            val_segments = generate_data_keys_sequential_window(
                config, val_recs_list, 5 * 60
            )
            val_segments = relabel_segments_to_threeclass(
                val_segments, pre_len=30.0
            )

            gen_val = SequentialGenerator(
                config,
                val_recs_list,
                val_segments,
                batch_size=600,
                shuffle=False,
            )

            if save_generators:
                print("Saving generator metadata to net/generators/ ...")

                meta_train = {
                    "recs_list": train_recs_list,
                    "segments": train_segments,
                }
                meta_val = {"recs_list": val_recs_list, "segments": val_segments}

                with open(meta_train_path, "wb") as f:
                    pickle.dump(meta_train, f, pickle.HIGHEST_PROTOCOL)
                with open(meta_val_path, "wb") as f:
                    pickle.dump(meta_val, f, pickle.HIGHEST_PROTOCOL)

                with open(gen_train_path, "wb") as f:
                    pickle.dump(gen_train, f, pickle.HIGHEST_PROTOCOL)
                with open(gen_val_path, "wb") as f:
                    pickle.dump(gen_val, f, pickle.HIGHEST_PROTOCOL)

        # ----------------------------------------------------------
        # TRAIN
        # ----------------------------------------------------------
        print("### Training model....")
        model = net(config)

        start_train = time.time()
        train_net(config, model, gen_train, gen_val, model_save_path)
        end_train = time.time() - start_train

        print("Total train duration = ", end_train / 60.0, "minutes")


# -------------------------------------------------------
# PREDICT
# -------------------------------------------------------
def predict(config):
    """
    Run prediction on test set and save y_pred / y_true to HDF5.

    NOTE: we store 1D probabilities and labels for pre-ictal (class 1).
    """
    name = config.get_name()
    model_save_path = os.path.join(config.save_dir, "models", name)

    pred_root = os.path.join(config.save_dir, "predictions")
    os.makedirs(pred_root, exist_ok=True)
    pred_path = os.path.join(pred_root, name)
    os.makedirs(pred_path, exist_ok=True)

    test_pats_list = pd.read_csv(
        os.path.join("net", "datasets", config.dataset + "_test.tsv"),
        sep="\t",
        header=None,
        skiprows=[0, 1, 2],
    )[0].to_list()

    test_recs_list = [
        [s, r.split("_")[-2]]
        for s in test_pats_list
        for r in os.listdir(os.path.join(config.data_path, s, "ses-01", "eeg"))
        if "edf" in r
    ]

    model_weights_path = os.path.join(
        model_save_path, "Weights", name + ".weights.h5"
    )

    # reload training config
    config.load_config(
        config_path=os.path.join(model_save_path, "configs"),
        config_name=name + ".cfg",
    )

    # model definition must match training
    if config.model == "DeepConvNet":
        from net.DeepConv_Net import net
    elif config.model == "ChronoNet":
        from net.ChronoNet import net
    elif config.model == "EEGnet":
        from net.EEGnet import net
    else:
        raise ValueError(f"Unknown model: {config.model}")

    for rec in tqdm(test_recs_list):
        out_file = os.path.join(
            pred_path, rec[0] + "_" + rec[1] + "preds.h5"
        )
        if os.path.isfile(out_file):
            print(rec[0] + " " + rec[1] + " exists. Skipping...")
            continue

        segments = generate_data_keys_sequential(
            config, [rec], verbose=False
        )
        segments = relabel_segments_to_threeclass(
            segments, pre_len=30.0
        )

        gen_test = SequentialGenerator(
            config,
            [rec],
            segments,
            batch_size=len(segments),
            shuffle=False,
            verbose=False,
        )

        model = net(config)

        y_pred, y_true = predict_net(gen_test, model_weights_path, model)

        with h5py.File(out_file, "w") as f:
            f.create_dataset("y_pred", data=y_pred)
            f.create_dataset("y_true", data=y_true)

        gc.collect()


# -------------------------------------------------------
# EVALUATE
# -------------------------------------------------------
def evaluate(config):
    """
    Evaluate predictions and save metrics to results HDF5.

    Uses pre-ictal probability (saved from predict()) and computes:
    - sens_ovlp (any-overlap)
    - FA/h (epoch & overlap)
    - etc., via get_metrics_scoring
    """
    name = config.get_name()

    pred_path = os.path.join(config.save_dir, "predictions", name)
    pred_fs = 1  # 1 Hz (one prediction per second)

    thresholds = list(np.around(np.linspace(0, 1, 51), 2))
    x_plot = np.linspace(0, 200, 200)

    results_root = os.path.join(config.save_dir, "results")
    os.makedirs(results_root, exist_ok=True)
    result_file = os.path.join(results_root, name + ".h5")

    sens_ovlp = []
    prec_ovlp = []
    fah_ovlp = []
    f1_ovlp = []
    sens_ovlp_plot = []
    prec_ovlp_plot = []

    sens_epoch = []
    spec_epoch = []
    prec_epoch = []
    fah_epoch = []
    f1_epoch = []

    score = []

    print("Getting evaluation metrics...")

    pred_files = sorted(os.listdir(pred_path))

    for file in tqdm(pred_files):
        with h5py.File(os.path.join(pred_path, file), "r") as f:
            y_pred = list(f["y_pred"])
            y_true = list(f["y_true"])

        sens_ovlp_th = []
        prec_ovlp_th = []
        fah_ovlp_th = []
        f1_ovlp_th = []

        sens_epoch_th = []
        spec_epoch_th = []
        prec_epoch_th = []
        fah_epoch_th = []
        f1_epoch_th = []
        score_th = []

        # motion artefact masking (same as original)
        rec = [file.split("_")[0], file.split("_")[1]]
        rec_data = Data.loadData(config.data_path, rec, modalities=["eeg"])

        [ch_focal, ch_cross] = apply_preprocess_eeg(config, rec_data)

        rmsa_f = [
            np.sqrt(np.mean(ch_focal[start : start + 2 * config.fs] ** 2))
            for start in range(0, len(ch_focal) - 2 * config.fs + 1, config.fs)
        ]
        rmsa_c = [
            np.sqrt(np.mean(ch_cross[start : start + 2 * config.fs] ** 2))
            for start in range(0, len(ch_focal) - 2 * config.fs + 1, config.fs)
        ]

        rmsa_f = [1 if 13 < rms < 150 else 0 for rms in rmsa_f]
        rmsa_c = [1 if 13 < rms < 150 else 0 for rms in rmsa_c]
        rmsa = [f & c for f, c in zip(rmsa_f, rmsa_c)]

        if len(y_pred) != len(rmsa):
            rmsa = rmsa[: len(y_pred)]

        y_pred = np.where(np.array(rmsa) == 0, 0, y_pred)

        for th in thresholds:
            (
                sens_ovlp_rec,
                prec_ovlp_rec,
                FA_ovlp_rec,
                f1_ovlp_rec,
                sens_epoch_rec,
                spec_epoch_rec,
                prec_epoch_rec,
                FA_epoch_rec,
                f1_epoch_rec,
            ) = get_metrics_scoring(y_pred, y_true, pred_fs, th)

            sens_ovlp_th.append(sens_ovlp_rec)
            prec_ovlp_th.append(prec_ovlp_rec)
            fah_ovlp_th.append(FA_ovlp_rec)
            f1_ovlp_th.append(f1_ovlp_rec)

            sens_epoch_th.append(sens_epoch_rec)
            spec_epoch_th.append(spec_epoch_rec)
            prec_epoch_th.append(prec_epoch_rec)
            fah_epoch_th.append(FA_epoch_rec)
            f1_epoch_th.append(f1_epoch_rec)

            score_th.append(sens_ovlp_rec * 100 - 0.4 * FA_epoch_rec)

        sens_ovlp.append(sens_ovlp_th)
        prec_ovlp.append(prec_ovlp_th)
        fah_ovlp.append(fah_ovlp_th)
        f1_ovlp.append(f1_ovlp_th)

        sens_epoch.append(sens_epoch_th)
        spec_epoch.append(spec_epoch_th)
        prec_epoch.append(prec_epoch_th)
        fah_epoch.append(fah_epoch_th)
        f1_epoch.append(f1_epoch_th)

        score.append(score_th)

        # For plotting curves
        to_cut = int(np.argmax(fah_ovlp_th))
        fah_ovlp_plot_rec = fah_ovlp_th[to_cut:]
        sens_ovlp_plot_rec = sens_ovlp_th[to_cut:]
        prec_ovlp_plot_rec = prec_ovlp_th[to_cut:]

        y_plot = np.interp(
            x_plot, fah_ovlp_plot_rec[::-1], sens_ovlp_plot_rec[::-1]
        )
        sens_ovlp_plot.append(y_plot)
        y_plot = np.interp(
            x_plot, sens_ovlp_plot_rec[::-1], prec_ovlp_plot_rec[::-1]
        )
        prec_ovlp_plot.append(y_plot)

    score_05 = [x[25] for x in score]
    print("Score: " + "%.2f" % np.nanmean(score_05))

    with h5py.File(result_file, "w") as f:
        f.create_dataset("sens_ovlp", data=sens_ovlp)
        f.create_dataset("prec_ovlp", data=prec_ovlp)
        f.create_dataset("fah_ovlp", data=fah_ovlp)
        f.create_dataset("f1_ovlp", data=f1_ovlp)
        f.create_dataset("sens_ovlp_plot", data=sens_ovlp_plot)
        f.create_dataset("prec_ovlp_plot", data=prec_ovlp_plot)
        f.create_dataset("x_plot", data=x_plot)
        f.create_dataset("sens_epoch", data=sens_epoch)
        f.create_dataset("spec_epoch", data=spec_epoch)
        f.create_dataset("prec_epoch", data=prec_epoch)
        f.create_dataset("fah_epoch", data=fah_epoch)
        f.create_dataset("f1_epoch", data=f1_epoch)
        f.create_dataset("score", data=score)