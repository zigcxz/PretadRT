import os

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from scipy.stats import spearmanr


# ============================================================
# 1. Paths
# ============================================================

ROOT = r"F:\RepoRT-master (2)\Tune"

PRE_DATA_ROOT = r"F:\RepoRT-master (2)\Pre_data"


GEN_DIR = os.path.join(
    ROOT,
    "gen_hmol_mordred",
    "np_seed29"
)


# ------------------------------------------------------------
# Must be the pretrained Hmol+Mordred RT base model.
# ------------------------------------------------------------

BASE_MODEL_DIR = os.path.join(
    PRE_DATA_ROOT,
    "base_model_hmol_mordred_molecule_split"
)


# For the older non-molecule-disjoint Hmol downstream model:
#
# BASE_MODEL_DIR = os.path.join(
#     PRE_DATA_ROOT,
#     "base_model_hmol_mordred"
# )


MODEL_PATH = os.path.join(
    BASE_MODEL_DIR,
    "base_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_MODEL_DIR,
    "scaler.pkl"
)


# "heldout_test" or "all_rows"
EVAL_MODE = "heldout_test"


OUTPUT_DIR = os.path.join(
    BASE_MODEL_DIR,
    "unknown_dataset_evaluation_standalone_split"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# 2. Load model/scaler
# ============================================================

def load_model_and_scaler():

    if not os.path.exists(
        MODEL_PATH
    ):

        raise FileNotFoundError(
            MODEL_PATH
        )

    if not os.path.exists(
        SCALER_PATH
    ):

        raise FileNotFoundError(
            SCALER_PATH
        )

    scaler = joblib.load(
        SCALER_PATH
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return (
        model,
        scaler
    )


# ============================================================
# 3. Load one dataset
# ============================================================

def load_eval_data(
    dataset_id
):

    d = os.path.join(
        GEN_DIR,
        dataset_id
    )

    if EVAL_MODE == "heldout_test":

        x_name = "X_test.csv"

        y_name = "y_test.csv"

        usp_name = "usp_test.csv"

        smiles_name = "smiles_test.csv"

    elif EVAL_MODE == "all_rows":

        x_name = "X_all.csv"

        y_name = "y_all.csv"

        usp_name = "usp_all.csv"

        smiles_name = "smiles_all.csv"

    else:

        raise ValueError(
            f"Unknown EVAL_MODE: {EVAL_MODE}"
        )

    x_path = os.path.join(
        d,
        x_name
    )

    y_path = os.path.join(
        d,
        y_name
    )

    usp_path = os.path.join(
        d,
        usp_name
    )

    smiles_path = os.path.join(
        d,
        smiles_name
    )

    for path in [
        x_path,
        y_path,
        usp_path
    ]:

        if not os.path.exists(
            path
        ):

            raise FileNotFoundError(
                path
            )

    if os.path.getsize(
        x_path
    ) == 0:

        return None

    X = pd.read_csv(
        x_path,
        header=None
    ).values.astype(
        np.float32
    )

    USP = pd.read_csv(
        usp_path,
        header=None
    ).values.astype(
        np.int64
    )

    y = (
        pd.read_csv(
            y_path,
            header=None
        )
        .values[
            :,
            0
        ]
        .astype(
            np.float32
        )
    )

    smiles_df = None

    if os.path.exists(
        smiles_path
    ):

        smiles_df = pd.read_csv(
            smiles_path
        )

    return (
        X,
        USP,
        y,
        smiles_df
    )


# ============================================================
# 4. Metrics
# ============================================================

def median_error(
    y_true,
    y_pred
):

    return float(
        np.median(
            np.abs(
                y_true
                -
                y_pred
            )
        )
    )


def safe_spearman(
    y_true,
    y_pred
):

    if len(
        y_true
    ) < 2:

        return np.nan

    rho, _ = spearmanr(
        y_true,
        y_pred
    )

    return float(
        rho
    )


# ============================================================
# 5. Evaluate one dataset
# ============================================================

def evaluate_one(
    model,
    scaler,
    dataset_id
):

    loaded = load_eval_data(
        dataset_id
    )

    if loaded is None:

        print(
            f"[SKIP] {dataset_id}: empty evaluation split"
        )

        return None

    (
        X,
        USP,
        y_true,
        smiles_df
    ) = loaded

    if hasattr(
        scaler,
        "n_features_in_"
    ):

        expected_dim = int(
            scaler.n_features_in_
        )

        if X.shape[
            1
        ] != expected_dim:

            raise RuntimeError(
                f"{dataset_id}: feature dimension mismatch.\n"
                f"Tune X dim = {X.shape[1]}\n"
                f"Scaler dim = {expected_dim}\n"
                "Check GIN/Hmol, Mordred columns, and "
                "condition-vector dimensions."
            )

    X_scaled = scaler.transform(
        X
    )

    y_pred = (
        model.predict(
            [
                X_scaled,
                USP
            ],
            verbose=0
        )
        .reshape(
            -1
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred
        )
    )

    med = median_error(
        y_true,
        y_pred
    )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
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
        if len(
            y_true
        ) >= 2
        else
        np.nan
    )

    rho = safe_spearman(
        y_true,
        y_pred
    )

    print(
        f"  {dataset_id}: "
        f"n={len(y_true)}, "
        f"MAE={mae:.4f}, "
        f"MedAE={med:.4f}, "
        f"RMSE={rmse:.4f}, "
        f"R2={r2:.4f}, "
        f"Spearman={rho:.4f}"
    )

    result_df = pd.DataFrame(
        {
            "RT_true":
                y_true,

            "RT_pred":
                y_pred,

            "abs_error":
                np.abs(
                    y_true
                    -
                    y_pred
                )
        }
    )

    if (
        smiles_df is not None
        and
        "canonical_smiles"
        in smiles_df.columns
        and
        len(
            smiles_df
        )
        ==
        len(
            result_df
        )
    ):

        result_df.insert(
            0,
            "canonical_smiles",
            smiles_df[
                "canonical_smiles"
            ].values
        )

    dataset_output = os.path.join(
        OUTPUT_DIR,
        dataset_id
    )

    os.makedirs(
        dataset_output,
        exist_ok=True
    )

    result_df.to_csv(
        os.path.join(
            dataset_output,
            "test_prediction.csv"
        ),
        index=False
    )

    return {
        "dataset_id":
            dataset_id,

        "n":
            len(
                y_true
            ),

        "MAE":
            mae,

        "Median":
            med,

        "RMSE":
            rmse,

        "R2":
            r2,

        "Spearman":
            rho,

        "y_true":
            y_true,

        "y_pred":
            y_pred,

        "predictions":
            result_df
    }


# ============================================================
# 6. Main
# ============================================================

def evaluate():

    model, scaler = (
        load_model_and_scaler()
    )

    if not os.path.exists(
        GEN_DIR
    ):

        raise FileNotFoundError(
            "Tune Hmol data not found:\n"
            f"{GEN_DIR}\n"
            "Run split_all_data_for_tune_hmol_mordred_standalone.py first."
        )

    dataset_ids = sorted(
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
        "=" * 80
    )

    print(
        "Unknown-dataset Hmol+Mordred NO-TUNE evaluation"
    )

    print(
        "=" * 80
    )

    print(
        "Evaluation mode:",
        EVAL_MODE
    )

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Datasets:",
        dataset_ids
    )

    print(
        "=" * 80
    )

    rows = []

    pooled_true = []

    pooled_pred = []

    pooled_dfs = []

    for dataset_id in dataset_ids:

        result = evaluate_one(
            model,
            scaler,
            dataset_id
        )

        if result is None:

            continue

        rows.append(
            {
                "dataset_id":
                    result[
                        "dataset_id"
                    ],

                "n":
                    result[
                        "n"
                    ],

                "MAE":
                    result[
                        "MAE"
                    ],

                "RMSE":
                    result[
                        "RMSE"
                    ],

                "Median":
                    result[
                        "Median"
                    ],

                "R2":
                    result[
                        "R2"
                    ],

                "Spearman":
                    result[
                        "Spearman"
                    ]
            }
        )

        pooled_true.append(
            result[
                "y_true"
            ]
        )

        pooled_pred.append(
            result[
                "y_pred"
            ]
        )

        tmp = (
            result[
                "predictions"
            ]
            .copy()
        )

        tmp.insert(
            0,
            "dataset_id",
            dataset_id
        )

        pooled_dfs.append(
            tmp
        )

    if not rows:

        raise RuntimeError(
            "No dataset was evaluated."
        )

    summary_df = pd.DataFrame(
        rows
    )

    y_true_all = np.concatenate(
        pooled_true
    )

    y_pred_all = np.concatenate(
        pooled_pred
    )

    pooled_mae = float(
        mean_absolute_error(
            y_true_all,
            y_pred_all
        )
    )

    pooled_medae = median_error(
        y_true_all,
        y_pred_all
    )

    pooled_rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true_all,
                y_pred_all
            )
        )
    )

    pooled_r2 = float(
        r2_score(
            y_true_all,
            y_pred_all
        )
    )

    pooled_rho = safe_spearman(
        y_true_all,
        y_pred_all
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "Overall"
    )

    print(
        "=" * 80
    )

    print(
        f"n = {len(y_true_all)}"
    )

    print(
        f"MAE = {pooled_mae:.4f}"
    )

    print(
        f"MedAE = {pooled_medae:.4f}"
    )

    print(
        f"RMSE = {pooled_rmse:.4f}"
    )

    print(
        f"R2 = {pooled_r2:.4f}"
    )

    print(
        f"Spearman = {pooled_rho:.4f}"
    )

    summary_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "unknown_dataset_summary.csv"
        ),
        index=False
    )

    pd.concat(
        pooled_dfs,
        ignore_index=True
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "unknown_dataset_pooled_predictions.csv"
        ),
        index=False
    )

    pd.DataFrame(
        [
            {
                "evaluation_mode":
                    EVAL_MODE,

                "n":
                    len(
                        y_true_all
                    ),

                "MAE":
                    pooled_mae,

                "MedAE":
                    pooled_medae,

                "RMSE":
                    pooled_rmse,

                "R2":
                    pooled_r2,

                "Spearman":
                    pooled_rho
            }
        ]
    ).to_csv(
        os.path.join(
            OUTPUT_DIR,
            "unknown_dataset_overall.csv"
        ),
        index=False
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    evaluate()
