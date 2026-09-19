import os
import json
import numpy as np
import pandas as pd
import torch

from rdkit import Chem


# ============================================================
# 1. Paths / configuration
# ============================================================

DATA_PATH = r"F:\RepoRT-master (2)\Pre_data"

SMILES_RT_DIR = os.path.join(
    DATA_PATH,
    "smiles_rt_data"
)

MOL_VEC_DIR = os.path.join(
    DATA_PATH,
    "processed",
    "molecular_vector"
)

GRAD_VEC_DIR = os.path.join(
    DATA_PATH,
    "processed",
    "gradient_vector"
)

META_VEC_DIR = os.path.join(
    DATA_PATH,
    "processed",
    "meta_vector"
)

COL_VEC_DIR = os.path.join(
    DATA_PATH,
    "processed",
    "column_name_vector"
)

USP_MAP_PATH = os.path.join(
    DATA_PATH,
    "processed",
    "usp_embedding.pth"
)

USP_DIR = os.path.join(
    DATA_PATH,
    "origin",
    "original_columns_usp_code"
)

TARGET_DIR = os.path.join(
    DATA_PATH,
    "target"
)

# Use a NEW output directory.
# Do not overwrite the original random-split outputs.
GEN_DIR = os.path.join(
    DATA_PATH,
    "gen_molecule_split"
)

GLOBAL_SPLIT_DIR = os.path.join(
    TARGET_DIR,
    "global_molecule_split"
)

SEEDS = [29]

# Keep the original ratio:
# first 90% = train+val
# then 90% of that = train
# => ~81% train, 9% val, 10% test
TRAIN_VAL_RATIO = 0.90
TRAIN_WITHIN_TRAINVAL_RATIO = 0.90


# ============================================================
# 2. Utilities
# ============================================================

def safe_mkdir(path):
    os.makedirs(
        path,
        exist_ok=True
    )


def read_csv_safe(
    path,
    header=None
):
    if not os.path.exists(path):
        print(
            f"[SKIP] File does not exist: {path}"
        )
        return None

    try:
        return pd.read_csv(
            path,
            header=header
        )

    except Exception as exc:
        print(
            f"[ERROR] Failed to read: {path}\n"
            f"{exc}"
        )
        return None


def canonicalize_smiles(smi):
    """
    Convert SMILES to canonical isomeric SMILES.

    This definition should remain consistent across:
      - dataset splitting
      - Hmol lookup
      - downstream analysis
    """

    if pd.isna(smi):
        return None

    try:
        mol = Chem.MolFromSmiles(
            str(smi)
        )

        if mol is None:
            return None

        return Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True
        )

    except Exception:
        return None


def save_vector(
    path,
    arr,
    fmt=None
):
    """
    Save 1D array robustly, including empty arrays.
    """

    arr = np.asarray(arr)

    if fmt is None:
        np.savetxt(
            path,
            arr,
            delimiter=","
        )
    else:
        np.savetxt(
            path,
            arr,
            fmt=fmt,
            delimiter=","
        )


def save_matrix(
    path,
    arr
):
    """
    Save 2D matrix robustly.
    """

    arr = np.asarray(arr)

    np.savetxt(
        path,
        arr,
        delimiter=","
    )


def validate_global_disjointness(
    split_df
):
    train_set = set(
        split_df.loc[
            split_df["split"] == "train",
            "canonical_smiles"
        ]
    )

    val_set = set(
        split_df.loc[
            split_df["split"] == "val",
            "canonical_smiles"
        ]
    )

    test_set = set(
        split_df.loc[
            split_df["split"] == "test",
            "canonical_smiles"
        ]
    )

    if train_set & val_set:
        raise RuntimeError(
            "Global train/val molecule overlap detected."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Global train/test molecule overlap detected."
        )

    if val_set & test_set:
        raise RuntimeError(
            "Global val/test molecule overlap detected."
        )

    print(
        "\nGlobal molecule disjointness check: PASS"
    )

    print(
        f"  train unique molecules = {len(train_set)}"
    )

    print(
        f"  val unique molecules   = {len(val_set)}"
    )

    print(
        f"  test unique molecules  = {len(test_set)}"
    )


# ============================================================
# 3. f1:
#    per-dataset filtering + canonicalization + deduplication
# ============================================================

def f1():

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 1: Per-dataset cleaning / canonicalization"
    )

    print(
        "=" * 80
    )

    if not os.path.exists(
        SMILES_RT_DIR
    ):
        raise FileNotFoundError(
            SMILES_RT_DIR
        )

    files = sorted(
        f
        for f in os.listdir(
            SMILES_RT_DIR
        )
        if f.endswith(
            ".txt"
        )
    )

    for file in files:

        dataset_id = file.replace(
            ".txt",
            ""
        )

        print(
            "\n" + "-" * 70
        )

        print(
            f"Dataset: {dataset_id}"
        )

        maccs_path = os.path.join(
            MOL_VEC_DIR,
            f"{dataset_id}_MACCS.csv"
        )

        mordred_path = os.path.join(
            MOL_VEC_DIR,
            f"{dataset_id}_Mordred_std.csv"
        )

        maccs = read_csv_safe(
            maccs_path,
            header=None
        )

        mordred = read_csv_safe(
            mordred_path,
            header=None
        )

        if (
            maccs is None
            or
            mordred is None
        ):
            print(
                f"[SKIP] {dataset_id}: molecular features missing."
            )
            continue

        if len(maccs) != len(mordred):
            raise RuntimeError(
                f"{dataset_id}: MACCS/Mordred row mismatch.\n"
                f"MACCS   = {len(maccs)}\n"
                f"Mordred = {len(mordred)}"
            )

        # ----------------------------------------------------
        # Reference columns
        # ----------------------------------------------------
        smiles_maccs_raw = (
            maccs.iloc[
                :,
                0
            ]
            .astype(str)
            .tolist()
        )

        smiles_mordred_raw = (
            mordred.iloc[
                :,
                0
            ]
            .astype(str)
            .tolist()
        )

        rt_maccs = (
            maccs.iloc[
                :,
                1
            ]
            .astype(float)
            .values
        )

        rt_mordred = (
            mordred.iloc[
                :,
                1
            ]
            .astype(float)
            .values
        )

        if not np.allclose(
            rt_maccs,
            rt_mordred,
            rtol=1e-6,
            atol=1e-6
        ):
            raise RuntimeError(
                f"{dataset_id}: RT mismatch between "
                "MACCS and Mordred tables."
            )

        canonical_maccs = [
            canonicalize_smiles(
                s
            )
            for s in smiles_maccs_raw
        ]

        canonical_mordred = [
            canonicalize_smiles(
                s
            )
            for s in smiles_mordred_raw
        ]

        # ----------------------------------------------------
        # Check molecular row alignment after canonicalization
        # ----------------------------------------------------
        mismatch = []

        for i, (
            smi_a,
            smi_b
        ) in enumerate(
            zip(
                canonical_maccs,
                canonical_mordred
            )
        ):

            if smi_a != smi_b:
                mismatch.append(
                    i
                )

        if mismatch:
            raise RuntimeError(
                f"{dataset_id}: canonical SMILES mismatch "
                "between MACCS and Mordred.\n"
                f"First indices: {mismatch[:10]}"
            )

        # ----------------------------------------------------
        # Keep:
        #   - valid canonical SMILES
        #   - RT != 0
        #   - first occurrence of each molecule WITHIN dataset
        # ----------------------------------------------------
        seen = set()
        keep = []
        kept_canonical = []

        invalid_count = 0
        zero_rt_count = 0
        duplicate_count = 0

        for i, (
            can_smi,
            rt
        ) in enumerate(
            zip(
                canonical_maccs,
                rt_maccs
            )
        ):

            if can_smi is None:
                invalid_count += 1
                continue

            if float(rt) == 0.0:
                zero_rt_count += 1
                continue

            if can_smi in seen:
                duplicate_count += 1
                continue

            seen.add(
                can_smi
            )

            keep.append(
                i
            )

            kept_canonical.append(
                can_smi
            )

        print(
            f"  Original rows         : {len(maccs)}"
        )

        print(
            f"  Invalid SMILES removed: {invalid_count}"
        )

        print(
            f"  RT==0 removed         : {zero_rt_count}"
        )

        print(
            f"  Within-dataset dupes  : {duplicate_count}"
        )

        print(
            f"  Final rows            : {len(keep)}"
        )

        if len(keep) == 0:
            print(
                f"[SKIP] {dataset_id}: no rows remain."
            )
            continue

        # ----------------------------------------------------
        # Filter both molecular feature tables using same rows.
        # Replace first column by canonical SMILES so every
        # downstream script uses the same molecule identity.
        # ----------------------------------------------------
        maccs_f = (
            maccs.iloc[
                keep
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        mordred_f = (
            mordred.iloc[
                keep
            ]
            .reset_index(
                drop=True
            )
            .copy()
        )

        maccs_f.iloc[
            :,
            0
        ] = kept_canonical

        mordred_f.iloc[
            :,
            0
        ] = kept_canonical

        save_dir = os.path.join(
            TARGET_DIR,
            dataset_id
        )

        safe_mkdir(
            save_dir
        )

        maccs_f.to_csv(
            os.path.join(
                save_dir,
                f"{dataset_id}_maccs.csv"
            ),
            index=False,
            header=False
        )

        mordred_f.to_csv(
            os.path.join(
                save_dir,
                f"{dataset_id}_mordred_std.csv"
            ),
            index=False,
            header=False
        )

        pd.DataFrame(
            {
                "canonical_smiles":
                    kept_canonical
            }
        ).to_csv(
            os.path.join(
                save_dir,
                "canonical_smiles.csv"
            ),
            index=False
        )


# ============================================================
# 4. Global unique-molecule split
# ============================================================

def build_global_molecule_table():

    if not os.path.exists(
        TARGET_DIR
    ):
        raise FileNotFoundError(
            TARGET_DIR
        )

    dataset_ids = sorted(
        d
        for d in os.listdir(
            TARGET_DIR
        )
        if (
            os.path.isdir(
                os.path.join(
                    TARGET_DIR,
                    d
                )
            )
            and
            d != "global_molecule_split"
        )
    )

    rows = []

    for dataset_id in dataset_ids:

        maccs_path = os.path.join(
            TARGET_DIR,
            dataset_id,
            f"{dataset_id}_maccs.csv"
        )

        if not os.path.exists(
            maccs_path
        ):
            continue

        maccs = pd.read_csv(
            maccs_path,
            header=None
        )

        for row_idx, smi in enumerate(
            maccs.iloc[
                :,
                0
            ]
            .astype(str)
            .tolist()
        ):

            can_smi = canonicalize_smiles(
                smi
            )

            if can_smi is None:
                raise RuntimeError(
                    f"{dataset_id}: invalid SMILES remained "
                    f"at row {row_idx}: {smi}"
                )

            rows.append(
                {
                    "dataset_id":
                        dataset_id,

                    "row_idx":
                        int(
                            row_idx
                        ),

                    "canonical_smiles":
                        can_smi
                }
            )

    membership_df = pd.DataFrame(
        rows
    )

    if len(
        membership_df
    ) == 0:
        raise RuntimeError(
            "No molecule rows found in TARGET_DIR."
        )

    # --------------------------------------------------------
    # A molecule may occur in multiple datasets.
    # Keep one global molecular identity but count how many
    # datasets / rows contain it.
    # --------------------------------------------------------
    molecule_stats = (
        membership_df
        .groupby(
            "canonical_smiles"
        )
        .agg(
            dataset_count=(
                "dataset_id",
                "nunique"
            ),

            measurement_count=(
                "dataset_id",
                "size"
            )
        )
        .reset_index()
        .sort_values(
            "canonical_smiles"
        )
        .reset_index(
            drop=True
        )
    )

    molecule_stats[
        "global_mol_id"
    ] = [
        f"GM{i:06d}"
        for i in range(
            len(
                molecule_stats
            )
        )
    ]

    return (
        molecule_stats,
        membership_df,
        dataset_ids
    )


def split_unique_molecules(
    molecule_stats,
    seed
):
    n = len(
        molecule_stats
    )

    if n < 3:
        raise RuntimeError(
            "Need at least 3 unique molecules for "
            "train/val/test split."
        )

    train_val_num = int(
        n
        *
        TRAIN_VAL_RATIO
    )

    train_num = int(
        train_val_num
        *
        TRAIN_WITHIN_TRAINVAL_RATIO
    )

    rng = np.random.RandomState(
        seed
    )

    perm = rng.permutation(
        n
    )

    train_idx = perm[
        :train_num
    ]

    val_idx = perm[
        train_num:
        train_val_num
    ]

    test_idx = perm[
        train_val_num:
    ]

    split_labels = np.empty(
        n,
        dtype=object
    )

    split_labels[
        train_idx
    ] = "train"

    split_labels[
        val_idx
    ] = "val"

    split_labels[
        test_idx
    ] = "test"

    out = molecule_stats.copy()

    out[
        "split"
    ] = split_labels

    return out


def f2():

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 2: Global molecule-level train/val/test split"
    )

    print(
        "=" * 80
    )

    (
        molecule_stats,
        membership_df,
        dataset_ids
    ) = build_global_molecule_table()

    print(
        f"Global unique molecules: "
        f"{len(molecule_stats)}"
    )

    print(
        "Molecules occurring in >1 dataset: "
        f"{int((molecule_stats['dataset_count'] > 1).sum())}"
    )

    for seed in SEEDS:

        print(
            "\n" + "-" * 70
        )

        print(
            f"Seed = {seed}"
        )

        global_split_df = split_unique_molecules(
            molecule_stats=
                molecule_stats,

            seed=
                seed
        )

        validate_global_disjointness(
            global_split_df
        )

        split_lookup = dict(
            zip(
                global_split_df[
                    "canonical_smiles"
                ],
                global_split_df[
                    "split"
                ]
            )
        )

        global_seed_dir = os.path.join(
            GLOBAL_SPLIT_DIR,
            f"np_seed{seed}"
        )

        safe_mkdir(
            global_seed_dir
        )

        global_split_df.to_csv(
            os.path.join(
                global_seed_dir,
                "global_molecule_split.csv"
            ),
            index=False
        )

        summary_rows = []

        # ----------------------------------------------------
        # Convert global molecule assignment back into
        # row indices for every dataset.
        # ----------------------------------------------------
        for dataset_id in dataset_ids:

            maccs_path = os.path.join(
                TARGET_DIR,
                dataset_id,
                f"{dataset_id}_maccs.csv"
            )

            if not os.path.exists(
                maccs_path
            ):
                continue

            maccs = pd.read_csv(
                maccs_path,
                header=None
            )

            smiles = [
                canonicalize_smiles(
                    s
                )
                for s in (
                    maccs.iloc[
                        :,
                        0
                    ]
                    .astype(str)
                    .tolist()
                )
            ]

            if any(
                s is None
                for s in smiles
            ):
                raise RuntimeError(
                    f"{dataset_id}: invalid canonical SMILES "
                    "while generating split indices."
                )

            row_split = []

            for smi in smiles:

                if smi not in split_lookup:
                    raise RuntimeError(
                        f"{dataset_id}: molecule missing from "
                        f"global split: {smi}"
                    )

                row_split.append(
                    split_lookup[
                        smi
                    ]
                )

            row_split = np.asarray(
                row_split,
                dtype=object
            )

            tr = np.where(
                row_split
                ==
                "train"
            )[0]

            va = np.where(
                row_split
                ==
                "val"
            )[0]

            te = np.where(
                row_split
                ==
                "test"
            )[0]

            split_dir = os.path.join(
                TARGET_DIR,
                dataset_id,
                "split_index_molecule",
                f"np_seed{seed}"
            )

            safe_mkdir(
                split_dir
            )

            save_vector(
                os.path.join(
                    split_dir,
                    "train_index.csv"
                ),
                tr,
                fmt="%d"
            )

            save_vector(
                os.path.join(
                    split_dir,
                    "val_index.csv"
                ),
                va,
                fmt="%d"
            )

            save_vector(
                os.path.join(
                    split_dir,
                    "test_index.csv"
                ),
                te,
                fmt="%d"
            )

            split_manifest = pd.DataFrame(
                {
                    "row_idx":
                        np.arange(
                            len(
                                smiles
                            )
                        ),

                    "canonical_smiles":
                        smiles,

                    "split":
                        row_split
                }
            )

            split_manifest.to_csv(
                os.path.join(
                    split_dir,
                    "split_manifest.csv"
                ),
                index=False
            )

            # Verify within this dataset.
            train_smiles = set(
                split_manifest.loc[
                    split_manifest[
                        "split"
                    ] == "train",
                    "canonical_smiles"
                ]
            )

            val_smiles = set(
                split_manifest.loc[
                    split_manifest[
                        "split"
                    ] == "val",
                    "canonical_smiles"
                ]
            )

            test_smiles = set(
                split_manifest.loc[
                    split_manifest[
                        "split"
                    ] == "test",
                    "canonical_smiles"
                ]
            )

            if (
                train_smiles & val_smiles
                or
                train_smiles & test_smiles
                or
                val_smiles & test_smiles
            ):
                raise RuntimeError(
                    f"{dataset_id}: molecule overlap "
                    "between local splits."
                )

            summary_rows.append(
                {
                    "dataset_id":
                        dataset_id,

                    "n_total":
                        len(
                            smiles
                        ),

                    "n_train":
                        len(
                            tr
                        ),

                    "n_val":
                        len(
                            va
                        ),

                    "n_test":
                        len(
                            te
                        ),

                    "unique_train":
                        len(
                            train_smiles
                        ),

                    "unique_val":
                        len(
                            val_smiles
                        ),

                    "unique_test":
                        len(
                            test_smiles
                        )
                }
            )

            print(
                f"{dataset_id}: "
                f"train={len(tr)}, "
                f"val={len(va)}, "
                f"test={len(te)}"
            )

        summary_df = pd.DataFrame(
            summary_rows
        )

        summary_df.to_csv(
            os.path.join(
                global_seed_dir,
                "dataset_split_summary.csv"
            ),
            index=False
        )


# ============================================================
# 5. Original feature building
#    MACCS + Mordred + LC condition
# ============================================================

def load_usp_mapping():

    if not os.path.exists(
        USP_MAP_PATH
    ):
        raise FileNotFoundError(
            USP_MAP_PATH
        )

    checkpoint = torch.load(
        USP_MAP_PATH,
        map_location="cpu"
    )

    if "word2id" not in checkpoint:
        raise KeyError(
            "usp_embedding.pth does not contain word2id."
        )

    return checkpoint[
        "word2id"
    ]


WORD2ID = load_usp_mapping()


def build_features(
    dataset_id
):

    save_dir = os.path.join(
        TARGET_DIR,
        dataset_id
    )

    maccs = pd.read_csv(
        os.path.join(
            save_dir,
            f"{dataset_id}_maccs.csv"
        ),
        header=None
    )

    mordred = pd.read_csv(
        os.path.join(
            save_dir,
            f"{dataset_id}_mordred_std.csv"
        ),
        header=None
    )

    if len(maccs) != len(mordred):
        raise RuntimeError(
            f"{dataset_id}: MACCS/Mordred row mismatch."
        )

    y = (
        maccs.iloc[
            :,
            1
        ]
        .values
        .astype(
            np.float32
        )
    )

    canonical_smiles = np.asarray(
        [
            canonicalize_smiles(
                s
            )
            for s in (
                maccs.iloc[
                    :,
                    0
                ]
                .astype(str)
                .tolist()
            )
        ],
        dtype=object
    )

    # --------------------------------------------------------
    # Molecular representation:
    # original MACCS + Mordred
    # --------------------------------------------------------
    X = np.concatenate(
        [
            maccs.iloc[
                :,
                2:
            ].values.astype(
                np.float32
            ),

            mordred.iloc[
                :,
                2:
            ].values.astype(
                np.float32
            )
        ],
        axis=1
    )

    n = X.shape[
        0
    ]

    # --------------------------------------------------------
    # Column vector
    # --------------------------------------------------------
    col_path = os.path.join(
        COL_VEC_DIR,
        f"{dataset_id}.csv"
    )

    col = np.loadtxt(
        col_path,
        delimiter=","
    )

    col = np.atleast_1d(
        col
    ).astype(
        np.float32
    ).reshape(
        -1
    )

    X = np.hstack(
        [
            X,

            np.tile(
                col.reshape(
                    1,
                    -1
                ),
                (
                    n,
                    1
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------
    grad = np.loadtxt(
        os.path.join(
            GRAD_VEC_DIR,
            f"{dataset_id}_gradient.csv"
        ),
        delimiter=","
    )

    grad = np.atleast_1d(
        grad
    ).astype(
        np.float32
    ).reshape(
        -1
    )

    X = np.hstack(
        [
            X,

            np.tile(
                grad.reshape(
                    1,
                    -1
                ),
                (
                    n,
                    1
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Meta
    # --------------------------------------------------------
    meta_df = pd.read_csv(
        os.path.join(
            META_VEC_DIR,
            f"{dataset_id}.csv"
        )
    )

    if len(meta_df) == 0:
        raise RuntimeError(
            f"{dataset_id}: meta file is empty."
        )

    meta = (
        meta_df.iloc[
            0
        ]
        .values
        .astype(
            np.float32
        )
        .reshape(
            -1
        )
    )

    X = np.hstack(
        [
            X,

            np.tile(
                meta.reshape(
                    1,
                    -1
                ),
                (
                    n,
                    1
                )
            )
        ]
    )

    # --------------------------------------------------------
    # USP
    # --------------------------------------------------------
    usp_txt = os.path.join(
        USP_DIR,
        f"{dataset_id}.txt"
    )

    with open(
        usp_txt,
        "r",
        encoding="utf-8"
    ) as f:

        usp = (
            f.readline()
            .strip()
        )

    if usp not in WORD2ID:
        raise KeyError(
            f"{dataset_id}: USP '{usp}' not found in word2id."
        )

    usp_id = int(
        WORD2ID[
            usp
        ]
    )

    usp_index = np.full(
        n,
        usp_id,
        dtype=np.int64
    )

    if not np.isfinite(
        X
    ).all():
        raise RuntimeError(
            f"{dataset_id}: X contains NaN/Inf."
        )

    return (
        X,
        y,
        usp_index,
        canonical_smiles
    )


# ============================================================
# 6. Build per-dataset and global output using
#    GLOBAL MOLECULE SPLIT
# ============================================================

def load_index_file(
    path
):
    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            path
        )

    if os.path.getsize(
        path
    ) == 0:
        return np.array(
            [],
            dtype=int
        )

    return np.loadtxt(
        path,
        delimiter=",",
        ndmin=1
    ).astype(
        int
    )


def f3():

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 3: Build features using global molecule split"
    )

    print(
        "=" * 80
    )

    dataset_ids = sorted(
        d
        for d in os.listdir(
            TARGET_DIR
        )
        if (
            os.path.isdir(
                os.path.join(
                    TARGET_DIR,
                    d
                )
            )
            and
            d != "global_molecule_split"
        )
    )

    for seed in SEEDS:

        X_train_all = []
        y_train_all = []
        usp_train_all = []
        smiles_train_all = []

        X_val_all = []
        y_val_all = []
        usp_val_all = []
        smiles_val_all = []

        processed_datasets = []

        for dataset_id in dataset_ids:

            maccs_path = os.path.join(
                TARGET_DIR,
                dataset_id,
                f"{dataset_id}_maccs.csv"
            )

            if not os.path.exists(
                maccs_path
            ):
                continue

            print(
                "\n" + "-" * 70
            )

            print(
                f"Dataset: {dataset_id}"
            )

            (
                X,
                y,
                usp_index,
                smiles
            ) = build_features(
                dataset_id
            )

            split_dir = os.path.join(
                TARGET_DIR,
                dataset_id,
                "split_index_molecule",
                f"np_seed{seed}"
            )

            tr = load_index_file(
                os.path.join(
                    split_dir,
                    "train_index.csv"
                )
            )

            va = load_index_file(
                os.path.join(
                    split_dir,
                    "val_index.csv"
                )
            )

            te = load_index_file(
                os.path.join(
                    split_dir,
                    "test_index.csv"
                )
            )

            # ------------------------------------------------
            # Safe slicing also works for empty index arrays.
            # ------------------------------------------------
            X_tr = X[
                tr
            ]

            y_tr = y[
                tr
            ]

            usp_tr = usp_index[
                tr
            ]

            smiles_tr = smiles[
                tr
            ]

            X_va = X[
                va
            ]

            y_va = y[
                va
            ]

            usp_va = usp_index[
                va
            ]

            smiles_va = smiles[
                va
            ]

            X_te = X[
                te
            ]

            y_te = y[
                te
            ]

            usp_te = usp_index[
                te
            ]

            smiles_te = smiles[
                te
            ]

            # ------------------------------------------------
            # Molecule overlap check for this dataset
            # ------------------------------------------------
            tr_set = set(
                smiles_tr.tolist()
            )

            va_set = set(
                smiles_va.tolist()
            )

            te_set = set(
                smiles_te.tolist()
            )

            if (
                tr_set & va_set
                or
                tr_set & te_set
                or
                va_set & te_set
            ):
                raise RuntimeError(
                    f"{dataset_id}: molecule overlap "
                    "after split."
                )

            # ------------------------------------------------
            # Dataset-specific output
            # ------------------------------------------------
            out_dir = os.path.join(
                GEN_DIR,
                f"np_seed{seed}",
                dataset_id
            )

            safe_mkdir(
                out_dir
            )

            save_matrix(
                os.path.join(
                    out_dir,
                    "X_train.csv"
                ),
                X_tr
            )

            save_vector(
                os.path.join(
                    out_dir,
                    "y_train.csv"
                ),
                y_tr
            )

            save_vector(
                os.path.join(
                    out_dir,
                    "usp_train.csv"
                ),
                usp_tr,
                fmt="%d"
            )

            save_matrix(
                os.path.join(
                    out_dir,
                    "X_val.csv"
                ),
                X_va
            )

            save_vector(
                os.path.join(
                    out_dir,
                    "y_val.csv"
                ),
                y_va
            )

            save_vector(
                os.path.join(
                    out_dir,
                    "usp_val.csv"
                ),
                usp_va,
                fmt="%d"
            )

            save_matrix(
                os.path.join(
                    out_dir,
                    "X_test.csv"
                ),
                X_te
            )

            save_vector(
                os.path.join(
                    out_dir,
                    "y_test.csv"
                ),
                y_te
            )

            save_vector(
                os.path.join(
                    out_dir,
                    "usp_test.csv"
                ),
                usp_te,
                fmt="%d"
            )

            pd.DataFrame(
                {
                    "canonical_smiles":
                        smiles_tr
                }
            ).to_csv(
                os.path.join(
                    out_dir,
                    "smiles_train.csv"
                ),
                index=False
            )

            pd.DataFrame(
                {
                    "canonical_smiles":
                        smiles_va
                }
            ).to_csv(
                os.path.join(
                    out_dir,
                    "smiles_val.csv"
                ),
                index=False
            )

            pd.DataFrame(
                {
                    "canonical_smiles":
                        smiles_te
                }
            ).to_csv(
                os.path.join(
                    out_dir,
                    "smiles_test.csv"
                ),
                index=False
            )

            # ------------------------------------------------
            # Global RT base-model pools:
            # only train / validation.
            # ------------------------------------------------
            if len(
                tr
            ) > 0:

                X_train_all.append(
                    X_tr
                )

                y_train_all.append(
                    y_tr
                )

                usp_train_all.append(
                    usp_tr
                )

                smiles_train_all.append(
                    smiles_tr
                )

            if len(
                va
            ) > 0:

                X_val_all.append(
                    X_va
                )

                y_val_all.append(
                    y_va
                )

                usp_val_all.append(
                    usp_va
                )

                smiles_val_all.append(
                    smiles_va
                )

            processed_datasets.append(
                dataset_id
            )

            print(
                f"  train={len(tr)}, "
                f"val={len(va)}, "
                f"test={len(te)}"
            )

        if not X_train_all:
            raise RuntimeError(
                "Global train pool is empty."
            )

        if not X_val_all:
            raise RuntimeError(
                "Global validation pool is empty."
            )

        X_train_all = np.vstack(
            X_train_all
        )

        y_train_all = np.concatenate(
            y_train_all
        )

        usp_train_all = np.concatenate(
            usp_train_all
        )

        smiles_train_all = np.concatenate(
            smiles_train_all
        )

        X_val_all = np.vstack(
            X_val_all
        )

        y_val_all = np.concatenate(
            y_val_all
        )

        usp_val_all = np.concatenate(
            usp_val_all
        )

        smiles_val_all = np.concatenate(
            smiles_val_all
        )

        # ----------------------------------------------------
        # Global overlap assertion.
        # Repeated train molecule across datasets is allowed,
        # but no molecule may appear in both train and val.
        # ----------------------------------------------------
        train_set = set(
            smiles_train_all.tolist()
        )

        val_set = set(
            smiles_val_all.tolist()
        )

        if train_set & val_set:
            raise RuntimeError(
                "Global train/val molecule overlap detected."
            )

        # ----------------------------------------------------
        # Shuffle rows only AFTER molecule assignment.
        # This does not alter split membership.
        # ----------------------------------------------------
        rng = np.random.RandomState(
            seed
        )

        tr_shuffle = rng.permutation(
            len(
                X_train_all
            )
        )

        va_shuffle = rng.permutation(
            len(
                X_val_all
            )
        )

        X_train_all = X_train_all[
            tr_shuffle
        ]

        y_train_all = y_train_all[
            tr_shuffle
        ]

        usp_train_all = usp_train_all[
            tr_shuffle
        ]

        smiles_train_all = smiles_train_all[
            tr_shuffle
        ]

        X_val_all = X_val_all[
            va_shuffle
        ]

        y_val_all = y_val_all[
            va_shuffle
        ]

        usp_val_all = usp_val_all[
            va_shuffle
        ]

        smiles_val_all = smiles_val_all[
            va_shuffle
        ]

        global_dir = os.path.join(
            GEN_DIR,
            f"np_seed{seed}"
        )

        safe_mkdir(
            global_dir
        )

        save_matrix(
            os.path.join(
                global_dir,
                "X_train.csv"
            ),
            X_train_all
        )

        save_vector(
            os.path.join(
                global_dir,
                "y_train.csv"
            ),
            y_train_all
        )

        save_vector(
            os.path.join(
                global_dir,
                "usp_train.csv"
            ),
            usp_train_all,
            fmt="%d"
        )

        save_matrix(
            os.path.join(
                global_dir,
                "X_val.csv"
            ),
            X_val_all
        )

        save_vector(
            os.path.join(
                global_dir,
                "y_val.csv"
            ),
            y_val_all
        )

        save_vector(
            os.path.join(
                global_dir,
                "usp_val.csv"
            ),
            usp_val_all,
            fmt="%d"
        )

        pd.DataFrame(
            {
                "canonical_smiles":
                    smiles_train_all
            }
        ).to_csv(
            os.path.join(
                global_dir,
                "smiles_train.csv"
            ),
            index=False
        )

        pd.DataFrame(
            {
                "canonical_smiles":
                    smiles_val_all
            }
        ).to_csv(
            os.path.join(
                global_dir,
                "smiles_val.csv"
            ),
            index=False
        )

        pd.DataFrame(
            {
                "dataset_id":
                    processed_datasets
            }
        ).to_csv(
            os.path.join(
                global_dir,
                "processed_datasets.csv"
            ),
            index=False
        )

        print(
            "\n" + "=" * 80
        )

        print(
            f"Global train rows: {len(X_train_all)}"
        )

        print(
            f"Global val rows  : {len(X_val_all)}"
        )

        print(
            f"Train unique molecules: "
            f"{len(set(smiles_train_all.tolist()))}"
        )

        print(
            f"Val unique molecules  : "
            f"{len(set(smiles_val_all.tolist()))}"
        )

        print(
            "Global train/val molecule overlap: 0"
        )

        print(
            "=" * 80
        )


# ============================================================
# 7. Main
# ============================================================

if __name__ == "__main__":

    f1()

    f2()

    f3()

    print(
        "\nAll finished."
    )
