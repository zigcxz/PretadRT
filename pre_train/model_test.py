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
# 1. PATH
# ============================================================

ROOT = r"F:\RepoRT-master (2)\Pre_data"


# ============================================================
# Original MDL-TL
# ============================================================

GEN_DIR = os.path.join(
    ROOT,
    "gen_molecule_split",
    "np_seed29"
)


MODEL_PATH = os.path.join(
    ROOT,
    "base_model_maccs_mordred_molecule_split",
    "base_model.keras"
)


SCALER_PATH = os.path.join(
    ROOT,
    "base_model_maccs_mordred_molecule_split",
    "scaler.pkl"
)


# ============================================================
# 单独保存 Original MDL-TL test result
# 不要覆盖 gen/ 中的数据
# ============================================================

OUTPUT_DIR = os.path.join(
    ROOT,
    "base_model_maccs_mordred_molecule_split",
    "test_evaluation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# Global train / validation molecule manifests
# ============================================================

TRAIN_SMILES_PATH = os.path.join(
    GEN_DIR,
    "smiles_train.csv"
)

VAL_SMILES_PATH = os.path.join(
    GEN_DIR,
    "smiles_val.csv"
)


def load_train_val_molecule_sets():

    train_set = set()
    val_set = set()

    if os.path.exists(TRAIN_SMILES_PATH):
        train_set = set(
            pd.read_csv(TRAIN_SMILES_PATH)["canonical_smiles"]
            .dropna()
            .astype(str)
        )

    if os.path.exists(VAL_SMILES_PATH):
        val_set = set(
            pd.read_csv(VAL_SMILES_PATH)["canonical_smiles"]
            .dropna()
            .astype(str)
        )

    overlap = train_set & val_set

    if overlap:
        raise RuntimeError(
            "Global train/validation molecule overlap detected "
            "before test evaluation."
        )

    return train_set, val_set


GLOBAL_TRAIN_MOLECULES, GLOBAL_VAL_MOLECULES = (
    load_train_val_molecule_sets()
)


# ============================================================
# 2. LOAD MODEL & SCALER
# ============================================================

def load_model_and_scaler():

    print("=" * 70)
    print("Loading MACCS + Mordred molecule-disjoint baseline")
    print("=" * 70)

    print("Model:")
    print(MODEL_PATH)

    print("\nScaler:")
    print(SCALER_PATH)


    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )


    if not os.path.exists(SCALER_PATH):

        raise FileNotFoundError(
            f"Scaler not found:\n{SCALER_PATH}"
        )


    scaler = joblib.load(
        SCALER_PATH
    )


    model = tf.keras.models.load_model(
        MODEL_PATH
    )


    print("\nModel loaded successfully.")

    print(
        "Scaler feature dimension:",
        scaler.n_features_in_
    )


    # ========================================================
    # 检查 model continuous input dimension
    # ========================================================

    model_feature_dim = int(
        model.inputs[0].shape[-1]
    )


    print(
        "Model feature dimension:",
        model_feature_dim
    )


    if (
        model_feature_dim
        !=
        scaler.n_features_in_
    ):

        raise RuntimeError(
            "Model feature dimension "
            "does not match scaler dimension!"
        )


    return (
        model,
        scaler
    )


# ============================================================
# 3. LOAD TEST DATA
# ============================================================

def load_test(dataset_id):

    dataset_dir = os.path.join(
        GEN_DIR,
        dataset_id
    )


    X_path = os.path.join(
        dataset_dir,
        "X_test.csv"
    )


    usp_path = os.path.join(
        dataset_dir,
        "usp_test.csv"
    )


    y_path = os.path.join(
        dataset_dir,
        "y_test.csv"
    )


    required_files = [
        X_path,
        usp_path,
        y_path
    ]


    for path in required_files:

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"Missing test file:\n{path}"
            )


    X_test = pd.read_csv(
        X_path,
        header=None
    ).values.astype(
        np.float32
    )


    USP_test = pd.read_csv(
        usp_path,
        header=None
    ).values.astype(
        np.int32
    )


    y_test = pd.read_csv(
        y_path,
        header=None
    ).values[:, 0].astype(
        np.float32
    )


    # ========================================================
    # Sample number check
    # ========================================================

    if not (
        len(X_test)
        ==
        len(USP_test)
        ==
        len(y_test)
    ):

        raise RuntimeError(
            f"{dataset_id}: "
            "X / USP / y sample counts do not match."
        )


    # ========================================================
    # NaN / Inf check
    # ========================================================

    if not np.isfinite(
        X_test
    ).all():

        raise RuntimeError(
            f"{dataset_id}: "
            "X_test contains NaN/Inf."
        )


    if not np.isfinite(
        y_test
    ).all():

        raise RuntimeError(
            f"{dataset_id}: "
            "y_test contains NaN/Inf."
        )


    return (
        X_test,
        USP_test,
        y_test
    )


# ============================================================
# 4. METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_true,
        y_pred
    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )


    # --------------------------------------------------------
    # Median Absolute Error
    # --------------------------------------------------------

    medae = np.median(
        np.abs(
            y_true
            -
            y_pred
        )
    )


    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    if len(y_true) >= 2:

        r2 = r2_score(
            y_true,
            y_pred
        )

    else:

        r2 = np.nan


    # --------------------------------------------------------
    # Spearman rank correlation
    #
    # 对保留顺序很有价值
    # --------------------------------------------------------

    if (
        len(y_true) >= 2
        and
        np.std(y_true) > 0
        and
        np.std(y_pred) > 0
    ):

        rho, pvalue = spearmanr(
            y_true,
            y_pred
        )

    else:

        rho = np.nan
        pvalue = np.nan


    return {

        "MAE":
            float(mae),

        "MedAE":
            float(medae),

        "RMSE":
            float(rmse),

        "R2":
            float(r2),

        "Spearman":
            float(rho),

        "Spearman_p":
            float(pvalue)
    }


# ============================================================
# 5. EVALUATE ONE DATASET
# ============================================================

def evaluate_one(
    model,
    scaler,
    dataset_id
):

    (
        X_test,
        USP_test,
        y_test
    ) = load_test(
        dataset_id
    )


    # ========================================================
    # Feature dimension check
    # ========================================================

    if (
        X_test.shape[1]
        !=
        scaler.n_features_in_
    ):

        raise RuntimeError(

            f"{dataset_id}: "
            f"X_test dim={X_test.shape[1]}, "
            f"scaler dim={scaler.n_features_in_}"

        )


    # ========================================================
    # 使用 Original MDL-TL 自己的 scaler
    # ========================================================

    X_test_scaled = scaler.transform(
        X_test
    ).astype(
        np.float32
    )


    # ========================================================
    # Prediction
    # ========================================================

    y_pred = model.predict(
        [
            X_test_scaled,
            USP_test
        ],
        verbose=0
    ).reshape(-1)


    # ========================================================
    # Metrics
    # ========================================================

    metrics = calculate_metrics(
        y_test,
        y_pred
    )


    n = len(
        y_test
    )


    print(

        f"{dataset_id:25s} "
        f"n={n:4d} | "
        f"MAE={metrics['MAE']:.4f} | "
        f"MedAE={metrics['MedAE']:.4f} | "
        f"RMSE={metrics['RMSE']:.4f} | "
        f"R2={metrics['R2']:.4f} | "
        f"rho={metrics['Spearman']:.4f}"

    )


    # ========================================================
    # Load test SMILES and verify molecule-disjointness
    # ========================================================

    smiles_path = os.path.join(
        GEN_DIR,
        dataset_id,
        "smiles_test.csv"
    )

    test_smiles_values = None

    if os.path.exists(smiles_path):

        smiles_df = pd.read_csv(smiles_path)

        if (
            "canonical_smiles" in smiles_df.columns
            and
            len(smiles_df) == n
        ):

            test_smiles_values = (
                smiles_df["canonical_smiles"]
                .astype(str)
                .values
            )

            test_set = set(test_smiles_values)

            overlap_train = (
                test_set
                &
                GLOBAL_TRAIN_MOLECULES
            )

            overlap_val = (
                test_set
                &
                GLOBAL_VAL_MOLECULES
            )

            if overlap_train:
                raise RuntimeError(
                    f"{dataset_id}: test molecules overlap "
                    f"GLOBAL training molecules. "
                    f"Count={len(overlap_train)}"
                )

            if overlap_val:
                raise RuntimeError(
                    f"{dataset_id}: test molecules overlap "
                    f"GLOBAL validation molecules. "
                    f"Count={len(overlap_val)}"
                )

    # ========================================================
    # Save dataset-level predictions
    # ========================================================

    prediction_df = pd.DataFrame({

        "RT_true":
            y_test,

        "RT_pred":
            y_pred,

        "error":
            np.abs(
                y_test
                -
                y_pred
            ),

        "squared_error":
            (
                y_test
                -
                y_pred
            ) ** 2

    })


    if test_smiles_values is not None:

        prediction_df.insert(
            0,
            "canonical_smiles",
            test_smiles_values
        )


    dataset_output_dir = os.path.join(
        OUTPUT_DIR,
        dataset_id
    )


    os.makedirs(
        dataset_output_dir,
        exist_ok=True
    )


    prediction_df.to_csv(

        os.path.join(
            dataset_output_dir,
            "test_prediction.csv"
        ),

        index=False

    )


    return (
        y_test,
        y_pred,
        metrics
    )


# ============================================================
# 6. MAIN
# ============================================================

def evaluate():

    # ========================================================
    # Load model
    # ========================================================

    model, scaler = (
        load_model_and_scaler()
    )


    # ========================================================
    # Find all dataset directories
    # ========================================================

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


    print("\n" + "=" * 70)

    print(
        f"Found {len(dataset_ids)} "
        f"dataset directories."
    )

    print("=" * 70)


    # ========================================================
    # Results
    # ========================================================

    summary_rows = []


    pooled_true = []

    pooled_pred = []

    pooled_dataset_id = []

    pooled_smiles = []


    # ========================================================
    # Evaluate each dataset
    # ========================================================

    for dataset_id in dataset_ids:


        test_path = os.path.join(
            GEN_DIR,
            dataset_id,
            "X_test.csv"
        )


        if not os.path.exists(
            test_path
        ):

            print(
                f"[SKIP] {dataset_id}: "
                "no X_test.csv"
            )

            continue

        # Under a GLOBAL molecule split, a small LC dataset can
        # legitimately contain zero test molecules.
        if os.path.getsize(test_path) == 0:

            print(
                f"[SKIP] {dataset_id}: "
                "empty test split under global molecule split"
            )

            continue


        try:

            (
                y_true,
                y_pred,
                metrics
            ) = evaluate_one(

                model,
                scaler,
                dataset_id

            )


        except Exception as e:

            print(
                f"[ERROR] {dataset_id}: "
                f"{e}"
            )

            continue


        n = len(
            y_true
        )


        # ====================================================
        # dataset-level summary
        # ====================================================

        summary_rows.append({

            "dataset_id":
                dataset_id,

            "n":
                n,

            "MAE":
                metrics[
                    "MAE"
                ],

            "MedAE":
                metrics[
                    "MedAE"
                ],

            "RMSE":
                metrics[
                    "RMSE"
                ],

            "R2":
                metrics[
                    "R2"
                ],

            "Spearman":
                metrics[
                    "Spearman"
                ],

            "Spearman_p":
                metrics[
                    "Spearman_p"
                ]

        })


        # ====================================================
        # pooled
        # ====================================================

        pooled_true.append(
            y_true
        )

        pooled_pred.append(
            y_pred
        )


        pooled_dataset_id.extend(
            [dataset_id] * n
        )

        smiles_path = os.path.join(
            GEN_DIR,
            dataset_id,
            "smiles_test.csv"
        )

        if os.path.exists(smiles_path):

            smiles_df = pd.read_csv(
                smiles_path
            )

            if (
                "canonical_smiles" in smiles_df.columns
                and
                len(smiles_df) == n
            ):

                pooled_smiles.extend(
                    smiles_df["canonical_smiles"]
                    .astype(str)
                    .tolist()
                )

            else:

                pooled_smiles.extend(
                    [None] * n
                )

        else:

            pooled_smiles.extend(
                [None] * n
            )


    # ========================================================
    # Check
    # ========================================================

    if len(
        summary_rows
    ) == 0:

        raise RuntimeError(
            "No datasets were successfully evaluated."
        )


    # ========================================================
    # 7. Save dataset-level summary
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )


    summary_path = os.path.join(
        OUTPUT_DIR,
        "test_summary_by_dataset.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    # ========================================================
    # 8. Pooled metrics
    #
    # 注意：
    # 这个才是真正的整体RMSE。
    # ========================================================

    pooled_true = np.concatenate(
        pooled_true
    )


    pooled_pred = np.concatenate(
        pooled_pred
    )


    global_metrics = calculate_metrics(
        pooled_true,
        pooled_pred
    )


    total_n = len(
        pooled_true
    )


    # ========================================================
    # 9. Macro metrics
    #
    # 每个dataset权重相同
    # ========================================================

    macro_mae = (
        summary_df[
            "MAE"
        ].mean()
    )


    macro_medae = (
        summary_df[
            "MedAE"
        ].mean()
    )


    macro_rmse = (
        summary_df[
            "RMSE"
        ].mean()
    )


    macro_r2 = (
        summary_df[
            "R2"
        ].mean()
    )


    macro_spearman = (
        summary_df[
            "Spearman"
        ].mean()
    )


    # ========================================================
    # 10. Sample-weighted MAE
    #
    # 应和 pooled MAE 基本一致
    # ========================================================

    weighted_mae = np.average(

        summary_df[
            "MAE"
        ],

        weights=
            summary_df[
                "n"
            ]

    )


    # ========================================================
    # 11. Overall summary
    # ========================================================

    overall_df = pd.DataFrame({

        "metric": [

            "number_of_datasets",

            "total_test_samples",

            "pooled_MAE",

            "pooled_MedAE",

            "pooled_RMSE",

            "pooled_R2",

            "pooled_Spearman",

            "macro_MAE",

            "macro_MedAE",

            "macro_RMSE",

            "macro_R2",

            "macro_Spearman",

            "sample_weighted_MAE"

        ],


        "value": [

            len(
                summary_df
            ),

            total_n,

            global_metrics[
                "MAE"
            ],

            global_metrics[
                "MedAE"
            ],

            global_metrics[
                "RMSE"
            ],

            global_metrics[
                "R2"
            ],

            global_metrics[
                "Spearman"
            ],

            macro_mae,

            macro_medae,

            macro_rmse,

            macro_r2,

            macro_spearman,

            weighted_mae

        ]

    })


    overall_path = os.path.join(
        OUTPUT_DIR,
        "overall_test_metrics.csv"
    )


    overall_df.to_csv(
        overall_path,
        index=False
    )


    # ========================================================
    # 12. Save pooled predictions
    #
    # dataset_id也一起保存
    # 后面可以直接和Contrastive模型做paired comparison
    # ========================================================

    pooled_df = pd.DataFrame({

        "dataset_id":
            pooled_dataset_id,

        "canonical_smiles":
            pooled_smiles,

        "RT_true":
            pooled_true,

        "RT_pred":
            pooled_pred,

        "error":
            np.abs(
                pooled_true
                -
                pooled_pred
            ),

        "squared_error":
            (
                pooled_true
                -
                pooled_pred
            ) ** 2

    })


    pooled_prediction_path = os.path.join(
        OUTPUT_DIR,
        "pooled_test_predictions.csv"
    )


    pooled_df.to_csv(
        pooled_prediction_path,
        index=False
    )


    # ========================================================
    # 13. Print
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "MACCS + Mordred Molecule-Disjoint Baseline Test Results"
    )

    print("=" * 70)


    print(
        f"Datasets evaluated: "
        f"{len(summary_df)}"
    )


    print(
        f"Total test samples: "
        f"{total_n}"
    )


    print(
        "\n--- Pooled sample-level metrics ---"
    )


    print(
        f"MAE      : "
        f"{global_metrics['MAE']:.4f}"
    )


    print(
        f"MedAE    : "
        f"{global_metrics['MedAE']:.4f}"
    )


    print(
        f"RMSE     : "
        f"{global_metrics['RMSE']:.4f}"
    )


    print(
        f"R2       : "
        f"{global_metrics['R2']:.4f}"
    )


    print(
        f"Spearman : "
        f"{global_metrics['Spearman']:.4f}"
    )


    print(
        "\n--- Dataset-level macro metrics ---"
    )


    print(
        f"Macro MAE      : "
        f"{macro_mae:.4f}"
    )


    print(
        f"Macro MedAE    : "
        f"{macro_medae:.4f}"
    )


    print(
        f"Macro RMSE     : "
        f"{macro_rmse:.4f}"
    )


    print(
        f"Macro R2       : "
        f"{macro_r2:.4f}"
    )


    print(
        f"Macro Spearman : "
        f"{macro_spearman:.4f}"
    )


    print(
        "\nFiles saved:"
    )


    print(
        summary_path
    )


    print(
        overall_path
    )


    print(
        pooled_prediction_path
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    evaluate()