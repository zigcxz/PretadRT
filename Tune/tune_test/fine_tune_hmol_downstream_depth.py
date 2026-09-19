# To perform a fine-tuning-depth ablation, only change TUNE_MODE:
#
#   "output_only"
#   "64_output"
#   "256_64_output"
#   "512_256_64_output"
#   "full_downstream"
#
# If you explicitly want USP_embedding to update too:
#   FREEZE_USP = False
#
# For a clean depth ablation, keep FREEZE_USP = True for ALL modes.
# ============================================================

import os
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import spearmanr


# ============================================================
# 1. Paths
# ============================================================

TUNE_ROOT = r"F:\RepoRT-master (2)\Tune"

PRE_ROOT = r"F:\RepoRT-master (2)\Pre_data"


# Output of:
# split_all_data_for_tune_hmol_mordred_standalone.py
GEN_DIR = os.path.join(
    TUNE_ROOT,
    "gen_hmol_mordred",
    "np_seed29"
)


# Hmol + Mordred source-domain RT model.
BASE_MODEL_DIR = os.path.join(
    PRE_ROOT,
    "base_model_hmol_mordred_molecule_split"
)

BASE_MODEL = os.path.join(
    BASE_MODEL_DIR,
    "base_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_MODEL_DIR,
    "scaler.pkl"
)


# ============================================================
# 2. Fine-tuning configuration
# ============================================================

# ------------------------------------------------------------
# Choose ONE:
#
# output_only
#     update RT output layer only
#
# 64_output
#     update Dense(64) + RT
#
# 256_64_output
#     update Dense(256) + Dense(64) + RT
#
# 512_256_64_output
#     update Dense(512) + Dense(256) + Dense(64) + RT
#
# full_downstream
#     update Dense(1024) + 512 + 256 + 64 + RT
# ------------------------------------------------------------

TUNE_MODE = "full_downstream"


# Keep this True for a clean DNN-depth ablation.
FREEZE_USP = False


LR = 1e-5

EPOCHS = 500

BATCH_SIZE = 32

SEED = 29


# Optional. False reproduces the spirit of your previous code:
# always train the requested 100 epochs.
USE_EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 15


OUTPUT_DIR = os.path.join(
    TUNE_ROOT,
    "fine_tune_hmol_downstream_result",
    (
        TUNE_MODE
        +
        (
            "_freeze_USP"
            if FREEZE_USP
            else
            "_tune_USP"
        )
    )
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 3. Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    tf.keras.utils.set_random_seed(seed)


# ============================================================
# 4. Dataset loading
# ============================================================

def _load_csv_matrix(path, dtype=np.float32):

    if not os.path.exists(path):

        raise FileNotFoundError(path)

    if os.path.getsize(path) == 0:

        return np.empty(
            (0, 0),
            dtype=dtype
        )

    return pd.read_csv(
        path,
        header=None
    ).values.astype(dtype)


def _load_csv_vector(path, dtype=np.float32):

    if not os.path.exists(path):

        raise FileNotFoundError(path)

    if os.path.getsize(path) == 0:

        return np.array(
            [],
            dtype=dtype
        )

    return (
        pd.read_csv(
            path,
            header=None
        )
        .values
        .reshape(-1)
        .astype(dtype)
    )


def load_dataset(dataset):

    path = os.path.join(
        GEN_DIR,
        dataset
    )

    X_train = _load_csv_matrix(
        os.path.join(
            path,
            "X_train.csv"
        )
    )

    y_train = _load_csv_vector(
        os.path.join(
            path,
            "y_train.csv"
        )
    )

    usp_train = _load_csv_vector(
        os.path.join(
            path,
            "usp_train.csv"
        ),
        dtype=np.int64
    )

    X_val = _load_csv_matrix(
        os.path.join(
            path,
            "X_val.csv"
        )
    )

    y_val = _load_csv_vector(
        os.path.join(
            path,
            "y_val.csv"
        )
    )

    usp_val = _load_csv_vector(
        os.path.join(
            path,
            "usp_val.csv"
        ),
        dtype=np.int64
    )

    X_test = _load_csv_matrix(
        os.path.join(
            path,
            "X_test.csv"
        )
    )

    y_test = _load_csv_vector(
        os.path.join(
            path,
            "y_test.csv"
        )
    )

    usp_test = _load_csv_vector(
        os.path.join(
            path,
            "usp_test.csv"
        ),
        dtype=np.int64
    )

    smiles_test = None

    smiles_path = os.path.join(
        path,
        "smiles_test.csv"
    )

    if os.path.exists(
        smiles_path
    ):

        smiles_df = pd.read_csv(
            smiles_path
        )

        if (
            "canonical_smiles"
            in smiles_df.columns
        ):

            smiles_test = (
                smiles_df[
                    "canonical_smiles"
                ]
                .astype(str)
                .values
            )

    return (
        X_train,
        y_train,
        usp_train,
        X_val,
        y_val,
        usp_val,
        X_test,
        y_test,
        usp_test,
        smiles_test
    )


# ============================================================
# 5. Metrics
# ============================================================

def safe_spearman(
    y_true,
    y_pred
):

    if len(y_true) < 2:

        return np.nan

    rho, _ = spearmanr(
        y_true,
        y_pred
    )

    return float(rho)


def evaluate(
    y_true,
    y_pred
):

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred
        )
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )

    median = float(
        np.median(
            np.abs(
                y_true
                -
                y_pred
            )
        )
    )

    r2 = (
        float(
            r2_score(
                y_true,
                y_pred
            )
        )
        if len(y_true) >= 2
        else
        np.nan
    )

    rho = safe_spearman(
        y_true,
        y_pred
    )

    return (
        mae,
        rmse,
        median,
        r2,
        rho
    )


# ============================================================
# 6. Configure trainable layers
# ============================================================

MODE_TO_DENSE_UNITS = {

    "output_only":
        set(),

    "64_output":
        {
            64
        },

    "256_64_output":
        {
            256,
            64
        },

    "512_256_64_output":
        {
            512,
            256,
            64
        },

    "full_downstream":
        {
            1024,
            512,
            256,
            64
        },
}


def configure_trainable_layers(
    model,
    mode,
    freeze_usp=True
):
    """
    Freeze everything first, then selectively unfreeze the requested
    Dense layers.

    The RT output layer is ALWAYS trainable.

    The current source model architecture is:
        1024 -> 512 -> 256 -> 64 -> RT(1)

    Using layer.units makes this robust to auto-generated Keras names
    such as dense, dense_1, dense_2, ...
    """

    if mode not in MODE_TO_DENSE_UNITS:

        raise ValueError(
            "Unknown TUNE_MODE: "
            f"{mode}\n"
            f"Allowed: {list(MODE_TO_DENSE_UNITS.keys())}"
        )

    requested_units = (
        MODE_TO_DENSE_UNITS[
            mode
        ]
    )

    # --------------------------------------------------------
    # Freeze all layers first.
    # --------------------------------------------------------

    for layer in model.layers:

        layer.trainable = False


    # --------------------------------------------------------
    # USP branch.
    # --------------------------------------------------------

    usp_layer = model.get_layer(
        "USP_embedding"
    )

    usp_layer.trainable = (
        not freeze_usp
    )


    # --------------------------------------------------------
    # Dense RT head.
    # --------------------------------------------------------

    found_hidden_units = []

    found_rt = False

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.Dense
        ):

            # RT output is explicitly named "RT".
            if layer.name == "RT":

                layer.trainable = True

                found_rt = True

                continue

            units = int(
                layer.units
            )

            if units in requested_units:

                layer.trainable = True

                found_hidden_units.append(
                    units
                )


    if not found_rt:

        raise RuntimeError(
            "Could not find Dense output layer named 'RT'."
        )


    missing_units = (
        requested_units
        -
        set(
            found_hidden_units
        )
    )

    if missing_units:

        raise RuntimeError(
            "Requested Dense layers were not found: "
            f"{sorted(missing_units)}"
        )


    # --------------------------------------------------------
    # Print + return an auditable table.
    # --------------------------------------------------------

    rows = []

    print(
        "\nTrainable-layer configuration"
    )

    print(
        "=" * 72
    )

    print(
        "TUNE_MODE:",
        mode
    )

    print(
        "FREEZE_USP:",
        freeze_usp
    )

    print(
        "-" * 72
    )

    for layer in model.layers:

        n_params = int(
            layer.count_params()
        )

        rows.append(
            {
                "layer_name":
                    layer.name,

                "layer_class":
                    layer.__class__.__name__,

                "trainable":
                    bool(
                        layer.trainable
                    ),

                "parameters":
                    n_params
            }
        )

        print(
            f"{layer.name:30s} "
            f"{layer.__class__.__name__:18s} "
            f"trainable={str(layer.trainable):5s} "
            f"params={n_params}"
        )


    trainable_params = int(
        np.sum(
            [
                np.prod(
                    v.shape
                )
                for v in model.trainable_weights
            ]
        )
    )

    non_trainable_params = int(
        np.sum(
            [
                np.prod(
                    v.shape
                )
                for v in model.non_trainable_weights
            ]
        )
    )

    print(
        "-" * 72
    )

    print(
        "Trainable parameters:",
        trainable_params
    )

    print(
        "Non-trainable parameters:",
        non_trainable_params
    )

    print(
        "=" * 72
    )

    return (
        pd.DataFrame(
            rows
        ),
        trainable_params,
        non_trainable_params
    )


# ============================================================
# 7. Fine-tune one unknown dataset
# ============================================================

def finetune_one(
    dataset
):

    print(
        "\n" + "=" * 80
    )

    print(
        "Hmol downstream fine-tuning:",
        dataset
    )

    print(
        "=" * 80
    )


    # Each target dataset starts from the SAME source base model.
    set_seed(
        SEED
    )


    (
        X_train,
        y_train,
        usp_train,
        X_val,
        y_val,
        usp_val,
        X_test,
        y_test,
        usp_test,
        smiles_test

    ) = load_dataset(
        dataset
    )


    if (
        len(X_train) == 0
        or
        len(X_val) == 0
        or
        len(X_test) == 0
    ):

        print(
            f"[SKIP] {dataset}: empty train/val/test split"
        )

        return None


    # ========================================================
    # 7.1 Source scaler only
    #
    # DO NOT fit a new scaler on target data.
    # ========================================================

    scaler = joblib.load(
        SCALER_PATH
    )


    if hasattr(
        scaler,
        "n_features_in_"
    ):

        expected_dim = int(
            scaler.n_features_in_
        )

        if X_train.shape[
            1
        ] != expected_dim:

            raise RuntimeError(
                f"{dataset}: feature dimension mismatch.\n"
                f"Tune Hmol X dim = {X_train.shape[1]}\n"
                f"Source scaler dim = {expected_dim}\n"
                "Check that GEN_DIR and BASE_MODEL_DIR both belong "
                "to Hmol + Mordred."
            )


    X_train = scaler.transform(
        X_train
    ).astype(
        np.float32
    )

    X_val = scaler.transform(
        X_val
    ).astype(
        np.float32
    )

    X_test = scaler.transform(
        X_test
    ).astype(
        np.float32
    )


    usp_train = usp_train.reshape(
        -1,
        1
    )

    usp_val = usp_val.reshape(
        -1,
        1
    )

    usp_test = usp_test.reshape(
        -1,
        1
    )


    # ========================================================
    # 7.2 Reload the SAME source model for every target dataset
    # ========================================================

    model = load_model(
        BASE_MODEL
    )


    (
        layer_df,
        trainable_params,
        non_trainable_params
    ) = configure_trainable_layers(
        model=
            model,

        mode=
            TUNE_MODE,

        freeze_usp=
            FREEZE_USP
    )


    # IMPORTANT:
    # compile AFTER changing trainable flags.
    model.compile(
        optimizer=Adam(
            learning_rate=
                LR
        ),
        loss="mae"
    )


    # ========================================================
    # 7.3 Optional callbacks
    # ========================================================

    callbacks = []

    if USE_EARLY_STOPPING:

        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=
                    EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            )
        )


    # ========================================================
    # 7.4 Fine-tune
    # ========================================================

    history = model.fit(
        [
            X_train,
            usp_train
        ],
        y_train,

        validation_data=(
            [
                X_val,
                usp_val
            ],
            y_val
        ),

        epochs=
            EPOCHS,

        batch_size=
            BATCH_SIZE,

        callbacks=
            callbacks,

        verbose=2,

        shuffle=True
    )


    # ========================================================
    # 7.5 Held-out test
    # ========================================================

    pred = (
        model.predict(
            [
                X_test,
                usp_test
            ],
            verbose=0
        )
        .reshape(-1)
    )


    (
        mae,
        rmse,
        median,
        r2,
        rho
    ) = evaluate(
        y_test,
        pred
    )


    print(
        f"\n{dataset} test:"
    )

    print(
        f"MAE      = {mae:.6f}"
    )

    print(
        f"RMSE     = {rmse:.6f}"
    )

    print(
        f"MedianAE = {median:.6f}"
    )

    print(
        f"R2       = {r2:.6f}"
    )

    print(
        f"Spearman = {rho:.6f}"
    )


    # ========================================================
    # 7.6 Save
    # ========================================================

    save_dir = os.path.join(
        OUTPUT_DIR,
        dataset
    )

    os.makedirs(
        save_dir,
        exist_ok=True
    )


    model.save(
        os.path.join(
            save_dir,
            "fine_tuned_model.keras"
        )
    )


    prediction_dict = {
        "RT_true":
            y_test,

        "RT_pred":
            pred,

        "abs_error":
            np.abs(
                y_test
                -
                pred
            )
    }

    if (
        smiles_test is not None
        and
        len(
            smiles_test
        ) == len(
            y_test
        )
    ):

        prediction_dict[
            "canonical_smiles"
        ] = smiles_test


    pd.DataFrame(
        prediction_dict
    ).to_csv(
        os.path.join(
            save_dir,
            "prediction.csv"
        ),
        index=False
    )


    pd.DataFrame(
        {
            "MAE":
                [
                    mae
                ],

            "RMSE":
                [
                    rmse
                ],

            "Median":
                [
                    median
                ],

            "R2":
                [
                    r2
                ],

            "Spearman":
                [
                    rho
                ]
        }
    ).to_csv(
        os.path.join(
            save_dir,
            "result.csv"
        ),
        index=False
    )


    pd.DataFrame(
        history.history
    ).to_csv(
        os.path.join(
            save_dir,
            "loss_history.csv"
        ),
        index=False
    )


    layer_df.to_csv(
        os.path.join(
            save_dir,
            "layer_trainability.csv"
        ),
        index=False
    )


    config = {
        "dataset":
            dataset,

        "representation":
            (
                "fixed pretrained Hmol + Mordred + "
                "column + gradient + meta"
            ),

        "gin_encoder":
            "fixed / not in downstream Keras graph",

        "tune_mode":
            TUNE_MODE,

        "freeze_usp":
            bool(
                FREEZE_USP
            ),

        "learning_rate":
            LR,

        "epochs_requested":
            EPOCHS,

        "epochs_completed":
            len(
                history.history[
                    "loss"
                ]
            ),

        "batch_size":
            BATCH_SIZE,

        "source_scaler_reused":
            True,

        "trainable_parameters":
            trainable_params,

        "non_trainable_parameters":
            non_trainable_params,

        "early_stopping":
            bool(
                USE_EARLY_STOPPING
            )
    }


    with open(
        os.path.join(
            save_dir,
            "fine_tune_config.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False
        )


    return {
        "dataset":
            dataset,

        "n_test":
            len(
                y_test
            ),

        "MAE":
            mae,

        "RMSE":
            rmse,

        "Median":
            median,

        "R2":
            r2,

        "Spearman":
            rho,

        "trainable_parameters":
            trainable_params
    }


# ============================================================
# 8. Main
# ============================================================

def main():

    if not os.path.exists(
        GEN_DIR
    ):

        raise FileNotFoundError(
            "Hmol Tune directory not found:\n"
            f"{GEN_DIR}\n\n"
            "Run split_all_data_for_tune_hmol_mordred_standalone.py "
            "first."
        )


    if not os.path.exists(
        BASE_MODEL
    ):

        raise FileNotFoundError(
            BASE_MODEL
        )


    if not os.path.exists(
        SCALER_PATH
    ):

        raise FileNotFoundError(
            SCALER_PATH
        )


    datasets = sorted(
        d
        for d in os.listdir(
            GEN_DIR
        )
        if os.path.isdir(
            os.path.join(
                GEN_DIR,
                d
            )
        )
    )


    print(
        "Datasets:",
        datasets
    )

    print(
        "TUNE_MODE:",
        TUNE_MODE
    )

    print(
        "FREEZE_USP:",
        FREEZE_USP
    )


    results = []

    for dataset in datasets:

        result = finetune_one(
            dataset
        )

        if result is not None:

            results.append(
                result
            )


    if not results:

        raise RuntimeError(
            "No target dataset was fine-tuned."
        )


    result_df = pd.DataFrame(
        results
    )


    result_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "all_result.csv"
        ),
        index=False
    )


    # Macro average across target datasets.
    macro_df = pd.DataFrame(
        [
            {
                "n_datasets":
                    len(
                        result_df
                    ),

                "mean_MAE":
                    float(
                        result_df[
                            "MAE"
                        ].mean()
                    ),

                "mean_RMSE":
                    float(
                        result_df[
                            "RMSE"
                        ].mean()
                    ),

                "mean_Median":
                    float(
                        result_df[
                            "Median"
                        ].mean()
                    ),

                "mean_R2":
                    float(
                        result_df[
                            "R2"
                        ].mean()
                    ),

                "mean_Spearman":
                    float(
                        result_df[
                            "Spearman"
                        ].mean()
                    )
            }
        ]
    )


    macro_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "macro_result.csv"
        ),
        index=False
    )


    print(
        "\nAll fine-tuning finished."
    )

    print(
        "Results:",
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()
