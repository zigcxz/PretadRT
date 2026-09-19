import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

ROOT = r"F:\RepoRT-master (2)\Tune"
PRE_DATA_ROOT = r"F:\RepoRT-master (2)\Pre_data"

GEN_DIR = os.path.join(
    ROOT,
    "gen_maccs_mordred_same_split",
    "np_seed29",
)

# Must be the MACCS + Mordred base model
BASE_MODEL_DIR = os.path.join(
    PRE_DATA_ROOT,
    "base_model_maccs_mordred_molecule_split",
)

MODEL_PATH = os.path.join(BASE_MODEL_DIR, "base_model.keras")
SCALER_PATH = os.path.join(BASE_MODEL_DIR, "scaler.pkl")

# "heldout_test" or "all_rows"
EVAL_MODE = "heldout_test"

OUTPUT_DIR = os.path.join(
    BASE_MODEL_DIR,
    "unknown_dataset_evaluation_same_split",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model_and_scaler():
    return (
        tf.keras.models.load_model(MODEL_PATH),
        joblib.load(SCALER_PATH),
    )


def load_eval(dataset_id):
    d = os.path.join(GEN_DIR, dataset_id)

    if EVAL_MODE == "heldout_test":
        suffix = "test"
    elif EVAL_MODE == "all_rows":
        suffix = "all"
    else:
        raise ValueError(EVAL_MODE)

    x_path = os.path.join(d, f"X_{suffix}.csv")
    if not os.path.exists(x_path) or os.path.getsize(x_path) == 0:
        return None

    X = pd.read_csv(x_path, header=None).values.astype(np.float32)
    y = pd.read_csv(
        os.path.join(d, f"y_{suffix}.csv"),
        header=None,
    ).values[:, 0].astype(np.float32)
    usp = pd.read_csv(
        os.path.join(d, f"usp_{suffix}.csv"),
        header=None,
    ).values.astype(np.int64)

    smiles_path = os.path.join(d, f"smiles_{suffix}.csv")
    smiles = None
    if os.path.exists(smiles_path):
        smiles = pd.read_csv(smiles_path)["canonical_smiles"].astype(str).values

    return X, y, usp, smiles


def safe_spearman(y_true, y_pred):
    if len(y_true) < 2:
        return np.nan
    return float(spearmanr(y_true, y_pred)[0])


def evaluate():
    model, scaler = load_model_and_scaler()

    dataset_ids = sorted(
        d for d in os.listdir(GEN_DIR)
        if os.path.isdir(os.path.join(GEN_DIR, d))
    )

    rows = []
    pooled_true, pooled_pred, pooled_df = [], [], []

    for dataset_id in dataset_ids:
        loaded = load_eval(dataset_id)
        if loaded is None:
            print(f"[SKIP] {dataset_id}: empty split")
            continue

        X, y_true, usp, smiles = loaded

        if hasattr(scaler, "n_features_in_"):
            expected = int(scaler.n_features_in_)
            if X.shape[1] != expected:
                raise RuntimeError(
                    f"{dataset_id}: feature dimension mismatch "
                    f"(Tune={X.shape[1]}, scaler={expected})"
                )

        Xs = scaler.transform(X)
        y_pred = model.predict([Xs, usp], verbose=0).reshape(-1)

        mae = float(mean_absolute_error(y_true, y_pred))
        med = float(np.median(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
        rho = safe_spearman(y_true, y_pred)

        print(
            f"{dataset_id}: n={len(y_true)}, "
            f"MAE={mae:.4f}, MedAE={med:.4f}, "
            f"RMSE={rmse:.4f}, R2={r2:.4f}, Spearman={rho:.4f}"
        )

        rows.append(
            {
                "dataset_id": dataset_id,
                "n": len(y_true),
                "MAE": mae,
                "Median": med,
                "RMSE": rmse,
                "R2": r2,
                "Spearman": rho,
            }
        )

        tmp = pd.DataFrame(
            {
                "dataset_id": dataset_id,
                "RT_true": y_true,
                "RT_pred": y_pred,
                "abs_error": np.abs(y_true - y_pred),
            }
        )
        if smiles is not None and len(smiles) == len(tmp):
            tmp.insert(1, "canonical_smiles", smiles)

        pooled_df.append(tmp)
        pooled_true.append(y_true)
        pooled_pred.append(y_pred)

    if not rows:
        raise RuntimeError("No dataset evaluated.")

    y_true_all = np.concatenate(pooled_true)
    y_pred_all = np.concatenate(pooled_pred)

    overall = {
        "evaluation_mode": EVAL_MODE,
        "n": len(y_true_all),
        "MAE": float(mean_absolute_error(y_true_all, y_pred_all)),
        "MedAE": float(np.median(np.abs(y_true_all - y_pred_all))),
        "RMSE": float(np.sqrt(mean_squared_error(y_true_all, y_pred_all))),
        "R2": float(r2_score(y_true_all, y_pred_all)),
        "Spearman": safe_spearman(y_true_all, y_pred_all),
    }

    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "unknown_dataset_summary.csv"),
        index=False,
    )
    pd.DataFrame([overall]).to_csv(
        os.path.join(OUTPUT_DIR, "unknown_dataset_overall.csv"),
        index=False,
    )
    pd.concat(pooled_df, ignore_index=True).to_csv(
        os.path.join(OUTPUT_DIR, "unknown_dataset_pooled_predictions.csv"),
        index=False,
    )

    print("\nOverall:")
    print(overall)
    print("\nSaved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    evaluate()
