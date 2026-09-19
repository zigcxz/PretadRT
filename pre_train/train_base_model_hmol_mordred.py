import os
import json
import random

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.preprocessing import (
    StandardScaler
)

import joblib

import torch


# ============================================================
# Local modules
# ============================================================

from model_hmol_mordred import (
    build_DNN
)

from config_hmol_mordred import *


# ============================================================
# 1. Reproducibility
# ============================================================

SEED = 29


random.seed(
    SEED
)


np.random.seed(
    SEED
)


tf.random.set_seed(
    SEED
)


# ============================================================
# Optional TensorFlow deterministic behavior
# ============================================================

try:

    tf.config.experimental.enable_op_determinism()

except Exception:

    pass


# ============================================================
# 2. Check required files
# ============================================================

def check_required_files():

    required_files = [

        X_PATH,
        VAL_X_PATH,

        USP_PATH,
        VAL_USP_PATH,

        Y_PATH,
        VAL_Y_PATH

    ]


    for path in required_files:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(

                f"Required file does not exist:\n"
                f"{path}\n\n"

                f"Please run "
                f"train_test_valid_hmol_mordred_global_molecule.py "
                f"first."

            )


# ============================================================
# 2.1 Molecule-split integrity check
# ============================================================

def check_molecule_split_integrity():

    if not os.path.exists(
        TRAIN_SMILES_PATH
    ):
        raise FileNotFoundError(
            f"Missing train molecule manifest:\n"
            f"{TRAIN_SMILES_PATH}"
        )

    if not os.path.exists(
        VAL_SMILES_PATH
    ):
        raise FileNotFoundError(
            f"Missing validation molecule manifest:\n"
            f"{VAL_SMILES_PATH}"
        )

    train_smiles = set(
        pd.read_csv(
            TRAIN_SMILES_PATH
        )[
            "canonical_smiles"
        ]
        .dropna()
        .astype(str)
    )

    val_smiles = set(
        pd.read_csv(
            VAL_SMILES_PATH
        )[
            "canonical_smiles"
        ]
        .dropna()
        .astype(str)
    )

    overlap = (
        train_smiles
        &
        val_smiles
    )

    if overlap:

        examples = list(
            overlap
        )[
            :20
        ]

        raise RuntimeError(
            "GLOBAL train/validation molecule overlap detected.\n"
            f"Overlap count = {len(overlap)}\n"
            f"Examples = {examples}"
        )

    print(
        "\nGlobal molecule-split check: PASS"
    )

    print(
        f"Train unique molecules: "
        f"{len(train_smiles)}"
    )

    print(
        f"Val unique molecules  : "
        f"{len(val_smiles)}"
    )

    print(
        "Train/val molecule overlap: 0"
    )


# ============================================================
# 3. Load dataset
# ============================================================

def load_data():

    # ========================================================
    # Train continuous features
    # ========================================================

    X_train = pd.read_csv(
        X_PATH,
        header=None
    ).values.astype(
        np.float32
    )


    # ========================================================
    # Train USP
    # ========================================================

    USP_train = pd.read_csv(
        USP_PATH,
        header=None
    ).values.astype(
        np.int32
    )


    # ========================================================
    # Train RT
    # ========================================================

    y_train = pd.read_csv(
        Y_PATH,
        header=None
    ).values[
        :,
        0
    ].astype(
        np.float32
    )


    # ========================================================
    # Validation continuous features
    # ========================================================

    X_val = pd.read_csv(
        VAL_X_PATH,
        header=None
    ).values.astype(
        np.float32
    )


    # ========================================================
    # Validation USP
    # ========================================================

    USP_val = pd.read_csv(
        VAL_USP_PATH,
        header=None
    ).values.astype(
        np.int32
    )


    # ========================================================
    # Validation RT
    # ========================================================

    y_val = pd.read_csv(
        VAL_Y_PATH,
        header=None
    ).values[
        :,
        0
    ].astype(
        np.float32
    )


    # ========================================================
    # Shape checks
    # ========================================================

    if not (
        len(X_train)
        ==
        len(USP_train)
        ==
        len(y_train)
    ):

        raise RuntimeError(

            "Train sample number mismatch:\n"

            f"X_train   = {len(X_train)}\n"
            f"USP_train = {len(USP_train)}\n"
            f"y_train   = {len(y_train)}"

        )


    if not (
        len(X_val)
        ==
        len(USP_val)
        ==
        len(y_val)
    ):

        raise RuntimeError(

            "Validation sample number mismatch:\n"

            f"X_val   = {len(X_val)}\n"
            f"USP_val = {len(USP_val)}\n"
            f"y_val   = {len(y_val)}"

        )


    # ========================================================
    # Feature dimension must be identical
    # ========================================================

    if (
        X_train.shape[1]
        !=
        X_val.shape[1]
    ):

        raise RuntimeError(

            "Train / validation feature dimensions differ:\n"

            f"train dim = "
            f"{X_train.shape[1]}\n"

            f"val dim = "
            f"{X_val.shape[1]}"

        )


    # ========================================================
    # NaN / Inf checks
    # ========================================================

    if not np.isfinite(
        X_train
    ).all():

        bad = int(
            np.sum(
                ~np.isfinite(
                    X_train
                )
            )
        )

        raise RuntimeError(

            f"X_train contains "
            f"{bad} NaN/Inf values."

        )


    if not np.isfinite(
        X_val
    ).all():

        bad = int(
            np.sum(
                ~np.isfinite(
                    X_val
                )
            )
        )

        raise RuntimeError(

            f"X_val contains "
            f"{bad} NaN/Inf values."

        )


    if not np.isfinite(
        y_train
    ).all():

        raise RuntimeError(
            "y_train contains NaN/Inf."
        )


    if not np.isfinite(
        y_val
    ).all():

        raise RuntimeError(
            "y_val contains NaN/Inf."
        )


    # ========================================================
    # Keras embedding expects shape [N, 1]
    # ========================================================

    USP_train = USP_train.reshape(
        -1,
        1
    )


    USP_val = USP_val.reshape(
        -1,
        1
    )


    return (

        X_train,
        USP_train,
        y_train,

        X_val,
        USP_val,
        y_val

    )


# ============================================================
# 4. Read feature layout
# ============================================================

def load_feature_layout():

    if not os.path.exists(
        FEATURE_LAYOUT_PATH
    ):

        print(
            "[WARNING] "
            "feature_layout.json not found."
        )

        return None


    with open(
        FEATURE_LAYOUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        layout = json.load(
            f
        )


    return layout


# ============================================================
# 5. Train
# ============================================================

def train():

    # ========================================================
    # File check
    # ========================================================

    check_required_files()

    check_molecule_split_integrity()


    # ========================================================
    # Load data
    # ========================================================

    (
        X_train,
        USP_train,
        y_train,

        X_val,
        USP_val,
        y_val

    ) = load_data()


    print(
        "=" * 75
    )

    print(
        "Hmol + Mordred + Hcond "
        "MDL-TL Base Model Training"
    )

    print(
        "=" * 75
    )


    print(
        "\nRaw train shape:",
        X_train.shape
    )


    print(
        "Raw val shape:",
        X_val.shape
    )


    print(
        "Train RT shape:",
        y_train.shape
    )


    print(
        "Val RT shape:",
        y_val.shape
    )


    print(
        "USP train shape:",
        USP_train.shape
    )


    print(
        "USP val shape:",
        USP_val.shape
    )


    # ========================================================
    # Feature layout
    # ========================================================

    feature_layout = (
        load_feature_layout()
    )


    if feature_layout is not None:

        print(
            "\nFeature layout:"
        )

        print(
            json.dumps(
                feature_layout,
                indent=2,
                ensure_ascii=False
            )
        )


        expected_dim = (
            feature_layout.get(
                "total_dim"
            )
        )


        if (
            expected_dim is not None
            and
            int(expected_dim)
            !=
            X_train.shape[1]
        ):

            raise RuntimeError(

                "Feature layout dimension "
                "does not match X_train.\n"

                f"layout total_dim = "
                f"{expected_dim}\n"

                f"X_train dim = "
                f"{X_train.shape[1]}"

            )


    # ========================================================
    # 6. NEW StandardScaler
    #
    # IMPORTANT:
    #
    # This scaler belongs ONLY to:
    #
    # Hmol + Mordred + Hcond
    #
    # Do NOT reuse:
    #
    # base_model/scaler.pkl
    #
    # or
    #
    # base_model_contrastive/scaler.pkl
    # ========================================================

    scaler = StandardScaler()


    scaler.fit(
        X_train
    )


    X_train_scaled = scaler.transform(
        X_train
    ).astype(
        np.float32
    )


    X_val_scaled = scaler.transform(
        X_val
    ).astype(
        np.float32
    )


    print(
        "\nScaled train:",
        X_train_scaled.shape
    )


    print(
        "Scaled val:",
        X_val_scaled.shape
    )


    print(
        "\nContinuous feature dimension:",
        X_train_scaled.shape[1]
    )


    # ========================================================
    # Sanity check after scaling
    # ========================================================

    if not np.isfinite(
        X_train_scaled
    ).all():

        raise RuntimeError(
            "Scaled X_train contains NaN/Inf."
        )


    if not np.isfinite(
        X_val_scaled
    ).all():

        raise RuntimeError(
            "Scaled X_val contains NaN/Inf."
        )


    # ========================================================
    # 7. USP class number
    #
    # Same USP mapping as Original MDL-TL
    # ========================================================

    usp_embedding_path = os.path.join(

        ROOT,

        "processed",

        "usp_embedding.pth"

    )


    if not os.path.exists(
        usp_embedding_path
    ):

        raise FileNotFoundError(
            usp_embedding_path
        )


    checkpoint = torch.load(
        usp_embedding_path,
        map_location="cpu"
    )


    if (
        "word2id"
        not in checkpoint
    ):

        raise KeyError(
            "usp_embedding.pth "
            "does not contain word2id."
        )


    usp_num = len(
        checkpoint[
            "word2id"
        ]
    )


    print(
        "\nUSP classes:",
        usp_num
    )


    print(
        "USP train range:",
        int(
            USP_train.min()
        ),
        "to",
        int(
            USP_train.max()
        )
    )


    print(
        "USP val range:",
        int(
            USP_val.min()
        ),
        "to",
        int(
            USP_val.max()
        )
    )


    # ========================================================
    # USP safety check
    # ========================================================

    if (
        USP_train.min() < 0
        or
        USP_val.min() < 0
    ):

        raise RuntimeError(
            "Negative USP index found."
        )


    if (
        USP_train.max()
        >=
        usp_num
    ):

        raise RuntimeError(

            "USP_train index exceeds "
            "embedding vocabulary size."

        )


    if (
        USP_val.max()
        >=
        usp_num
    ):

        raise RuntimeError(

            "USP_val index exceeds "
            "embedding vocabulary size."

        )


    # ========================================================
    # 8. Build SAME MDL-TL DNN
    #
    # feature_dim is dynamically determined.
    # ========================================================

    model = build_DNN(

        feature_dim=
            X_train_scaled.shape[1],

        usp_num=
            usp_num,

        lr=
            0.001

    )


    print(
        "\nModel summary:"
    )


    model.summary()


    # ========================================================
    # 9. Training
    #
    # SAME protocol as Original:
    #
    # epochs = 320
    # batch_size = 64
    # optimizer = Adam
    # LR = 0.001
    # loss = MAE
    #
    # Do not alter these yet.
    # ========================================================

    print(
        "\nStarting training..."
    )


    history = model.fit(

        [
            X_train_scaled,
            USP_train
        ],

        y_train,


        validation_data=(

            [
                X_val_scaled,
                USP_val
            ],

            y_val

        ),


        epochs=320,


        batch_size=64,


        verbose=2

    )


    # ========================================================
    # 10. Save model
    # ========================================================

    model_path = os.path.join(

        MODEL_DIR,

        "base_model.keras"

    )


    model.save(
        model_path
    )


    # ========================================================
    # 11. Save NEW scaler
    # ========================================================

    scaler_path = os.path.join(

        MODEL_DIR,

        "scaler.pkl"

    )


    joblib.dump(

        scaler,

        scaler_path

    )


    # ========================================================
    # 12. Save training history
    # ========================================================

    history_path = os.path.join(

        MODEL_DIR,

        "loss_history.csv"

    )


    pd.DataFrame(

        history.history

    ).to_csv(

        history_path,

        index=False

    )


    # ========================================================
    # 13. Save training configuration
    # ========================================================

    config = {

        "seed":
            SEED,


        "feature_dim":
            int(
                X_train_scaled.shape[
                    1
                ]
            ),


        "usp_num":
            int(
                usp_num
            ),


        "molecular_representation":
            (
                "chemical-prior contrastive "
                "Hmol + Mordred"
            ),


        "condition_representation":
            (
                "column + gradient + meta "
                "+ USP embedding"
            ),


        "loss":
            "mae",


        "optimizer":
            "Adam",


        "learning_rate":
            0.001,


        "epochs":
            320,


        "batch_size":
            64,


        "rt_split":
            "global_canonical_molecule_disjoint",


        "gin_pretraining":
            "all_molecules_transductive"

    }


    if feature_layout is not None:

        config[
            "feature_layout"
        ] = feature_layout


    config_path = os.path.join(

        MODEL_DIR,

        "training_config.json"

    )


    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(

            config,

            f,

            indent=2,

            ensure_ascii=False

        )


    # ========================================================
    # 14. Final report
    # ========================================================

    print(
        "\n" + "=" * 75
    )


    print(
        "Hmol + Mordred + Hcond "
        "base model training finished"
    )


    print(
        "=" * 75
    )


    print(
        "\nModel saved:"
    )

    print(
        model_path
    )


    print(
        "\nScaler saved:"
    )

    print(
        scaler_path
    )


    print(
        "\nLoss history saved:"
    )

    print(
        history_path
    )


    print(
        "\nTraining config saved:"
    )

    print(
        config_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()