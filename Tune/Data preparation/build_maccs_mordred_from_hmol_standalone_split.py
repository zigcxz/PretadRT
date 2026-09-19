import os
import json
import numpy as np
import pandas as pd
import torch

ROOT = r"F:\RepoRT-master (2)\Tune"
PRE_DATA_ROOT = r"F:\RepoRT-master (2)\Pre_data"

# Reuse the cleaned rows + split created by:
# split_all_data_for_tune_hmol_mordred_standalone.py
TARGET_DIR = os.path.join(ROOT, "target_hmol_mordred")

# Separate MACCS baseline output
GEN_DIR = os.path.join(ROOT, "gen_maccs_mordred_same_split")

COL_VEC_DIR = os.path.join(ROOT, "processed", "column_name_vector")
GRAD_VEC_DIR = os.path.join(ROOT, "processed", "gradient_vector")
META_VEC_DIR = os.path.join(ROOT, "processed", "meta_vector")
USP_DIR = os.path.join(ROOT, "origin", "original_columns_usp_code")
USP_MAP_PATH = os.path.join(PRE_DATA_ROOT, "processed", "usp_embedding.pth")

SEEDS = [29]


def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)


def load_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) == 0:
        return np.array([], dtype=int)
    return np.loadtxt(path, delimiter=",", ndmin=1).astype(int)


def save_matrix(path, arr):
    np.savetxt(path, np.asarray(arr), delimiter=",")


def save_vector(path, arr, fmt=None):
    arr = np.asarray(arr)
    if fmt is None:
        np.savetxt(path, arr, delimiter=",")
    else:
        np.savetxt(path, arr, delimiter=",", fmt=fmt)


checkpoint = torch.load(USP_MAP_PATH, map_location="cpu")
WORD2ID = checkpoint["word2id"]


def build_features(dataset_id):
    d = os.path.join(TARGET_DIR, dataset_id)

    maccs = pd.read_csv(
        os.path.join(d, f"{dataset_id}_maccs.csv"),
        header=None,
    )
    mordred = pd.read_csv(
        os.path.join(d, f"{dataset_id}_mordred_std.csv"),
        header=None,
    )

    if len(maccs) != len(mordred):
        raise RuntimeError(f"{dataset_id}: MACCS/Mordred row mismatch")

    smiles = maccs.iloc[:, 0].astype(str).values
    y = maccs.iloc[:, 1].values.astype(np.float32)

    y2 = mordred.iloc[:, 1].values.astype(np.float32)
    if not np.allclose(y, y2, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{dataset_id}: RT mismatch")

    X_maccs = maccs.iloc[:, 2:].values.astype(np.float32)
    X_mordred = mordred.iloc[:, 2:].values.astype(np.float32)

    n = len(y)

    col = np.loadtxt(
        os.path.join(COL_VEC_DIR, f"{dataset_id}.csv"),
        delimiter=",",
    )
    col = np.atleast_1d(col).astype(np.float32).reshape(-1)
    X_col = np.tile(col.reshape(1, -1), (n, 1))

    grad = np.loadtxt(
        os.path.join(GRAD_VEC_DIR, f"{dataset_id}_gradient.csv"),
        delimiter=",",
    )
    grad = np.atleast_1d(grad).astype(np.float32).reshape(-1)
    X_grad = np.tile(grad.reshape(1, -1), (n, 1))

    meta_df = pd.read_csv(os.path.join(META_VEC_DIR, f"{dataset_id}.csv"))
    if len(meta_df) == 0:
        raise RuntimeError(f"{dataset_id}: empty meta")
    meta = meta_df.iloc[0].values.astype(np.float32).reshape(-1)
    X_meta = np.tile(meta.reshape(1, -1), (n, 1))

    # Baseline feature order must match the baseline training code.
    X = np.hstack(
        [X_maccs, X_mordred, X_col, X_grad, X_meta]
    ).astype(np.float32)

    if not np.isfinite(X).all():
        raise RuntimeError(f"{dataset_id}: X contains NaN/Inf")

    with open(
        os.path.join(USP_DIR, f"{dataset_id}.txt"),
        "r",
        encoding="utf-8",
    ) as f:
        usp = f.readline().strip()

    if usp not in WORD2ID:
        raise KeyError(f"{dataset_id}: unknown USP code: {usp}")

    usp_index = np.full(n, int(WORD2ID[usp]), dtype=np.int64)

    layout = {
        "maccs_dim": int(X_maccs.shape[1]),
        "mordred_dim": int(X_mordred.shape[1]),
        "column_dim": int(X_col.shape[1]),
        "gradient_dim": int(X_grad.shape[1]),
        "meta_dim": int(X_meta.shape[1]),
        "total_dim": int(X.shape[1]),
    }

    return X, y, usp_index, smiles, layout


def generate():
    if not os.path.exists(TARGET_DIR):
        raise FileNotFoundError(
            f"{TARGET_DIR}\n"
            "Run split_all_data_for_tune_hmol_mordred_standalone.py first."
        )

    dataset_ids = sorted(
        d for d in os.listdir(TARGET_DIR)
        if os.path.isdir(os.path.join(TARGET_DIR, d))
    )

    for seed in SEEDS:
        X_train_all, y_train_all, usp_train_all, smi_train_all = [], [], [], []
        X_val_all, y_val_all, usp_val_all, smi_val_all = [], [], [], []
        feature_layout = None
        summary = []

        for dataset_id in dataset_ids:
            X, y, usp, smiles, dims = build_features(dataset_id)

            if feature_layout is None:
                feature_layout = dims
            elif dims != feature_layout:
                raise RuntimeError(
                    f"{dataset_id}: feature layout mismatch\n"
                    f"{dims}\nvs\n{feature_layout}"
                )

            split_dir = os.path.join(
                TARGET_DIR, dataset_id, "split_index", f"np_seed{seed}"
            )
            tr = load_index(os.path.join(split_dir, "train_index.csv"))
            va = load_index(os.path.join(split_dir, "val_index.csv"))
            te = load_index(os.path.join(split_dir, "test_index.csv"))

            all_idx = np.concatenate([tr, va, te])
            if len(all_idx) != len(X) or len(np.unique(all_idx)) != len(all_idx):
                raise RuntimeError(f"{dataset_id}: invalid split indices")

            out = os.path.join(GEN_DIR, f"np_seed{seed}", dataset_id)
            safe_mkdir(out)

            split_map = {
                "train": tr,
                "val": va,
                "test": te,
            }

            for split_name, idx in split_map.items():
                save_matrix(os.path.join(out, f"X_{split_name}.csv"), X[idx])
                save_vector(os.path.join(out, f"y_{split_name}.csv"), y[idx])
                save_vector(
                    os.path.join(out, f"usp_{split_name}.csv"),
                    usp[idx],
                    fmt="%d",
                )
                pd.DataFrame(
                    {"canonical_smiles": smiles[idx]}
                ).to_csv(
                    os.path.join(out, f"smiles_{split_name}.csv"),
                    index=False,
                )

            # all rows for pure zero-shot evaluation
            save_matrix(os.path.join(out, "X_all.csv"), X)
            save_vector(os.path.join(out, "y_all.csv"), y)
            save_vector(os.path.join(out, "usp_all.csv"), usp, fmt="%d")
            pd.DataFrame(
                {"canonical_smiles": smiles}
            ).to_csv(
                os.path.join(out, "smiles_all.csv"),
                index=False,
            )

            if len(tr):
                X_train_all.append(X[tr])
                y_train_all.append(y[tr])
                usp_train_all.append(usp[tr])
                smi_train_all.append(smiles[tr])

            if len(va):
                X_val_all.append(X[va])
                y_val_all.append(y[va])
                usp_val_all.append(usp[va])
                smi_val_all.append(smiles[va])

            summary.append(
                {
                    "dataset_id": dataset_id,
                    "n_total": len(X),
                    "n_train": len(tr),
                    "n_val": len(va),
                    "n_test": len(te),
                    "feature_dim": X.shape[1],
                }
            )

            print(
                f"{dataset_id}: X={X.shape}, "
                f"train={len(tr)}, val={len(va)}, test={len(te)}"
            )

        global_dir = os.path.join(GEN_DIR, f"np_seed{seed}")
        safe_mkdir(global_dir)

        if X_train_all:
            Xtr = np.vstack(X_train_all)
            ytr = np.concatenate(y_train_all)
            utr = np.concatenate(usp_train_all)
            str_ = np.concatenate(smi_train_all)
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(Xtr))
            save_matrix(os.path.join(global_dir, "X_train.csv"), Xtr[order])
            save_vector(os.path.join(global_dir, "y_train.csv"), ytr[order])
            save_vector(
                os.path.join(global_dir, "usp_train.csv"),
                utr[order],
                fmt="%d",
            )
            pd.DataFrame(
                {"canonical_smiles": str_[order]}
            ).to_csv(
                os.path.join(global_dir, "smiles_train.csv"),
                index=False,
            )

        if X_val_all:
            Xva = np.vstack(X_val_all)
            yva = np.concatenate(y_val_all)
            uva = np.concatenate(usp_val_all)
            sva = np.concatenate(smi_val_all)
            rng = np.random.RandomState(seed)
            order = rng.permutation(len(Xva))
            save_matrix(os.path.join(global_dir, "X_val.csv"), Xva[order])
            save_vector(os.path.join(global_dir, "y_val.csv"), yva[order])
            save_vector(
                os.path.join(global_dir, "usp_val.csv"),
                uva[order],
                fmt="%d",
            )
            pd.DataFrame(
                {"canonical_smiles": sva[order]}
            ).to_csv(
                os.path.join(global_dir, "smiles_val.csv"),
                index=False,
            )

        with open(
            os.path.join(global_dir, "feature_layout.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(feature_layout, f, indent=2, ensure_ascii=False)

        pd.DataFrame(summary).to_csv(
            os.path.join(global_dir, "dataset_split_summary.csv"),
            index=False,
        )

        print("\nFinished:")
        print(global_dir)
        print(feature_layout)


if __name__ == "__main__":
    generate()
