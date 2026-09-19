import os
import sys
import json
import random

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from torch_geometric.data import Batch


# ============================================================
# 1. Paths / config
# ============================================================

ROOT = r"F:\RepoRT-master (2)\Tune"

PRE_DATA_ROOT = r"F:\RepoRT-master (2)\Pre_data"

CONTRAST_ROOT = r"F:\RepoRT-master (2)\Contrast_learning"


SMILES_RT_DIR = os.path.join(
    ROOT,
    "smiles_rt_data"
)

MOL_VEC_DIR = os.path.join(
    ROOT,
    "processed",
    "molecular_vector"
)

COL_VEC_DIR = os.path.join(
    ROOT,
    "processed",
    "column_name_vector"
)

GRAD_VEC_DIR = os.path.join(
    ROOT,
    "processed",
    "gradient_vector"
)

META_VEC_DIR = os.path.join(
    ROOT,
    "processed",
    "meta_vector"
)

USP_DIR = os.path.join(
    ROOT,
    "origin",
    "original_columns_usp_code"
)


# Independent target/output directories.
# They do NOT depend on Tune/target or Tune/gen from MACCS baseline.
TARGET_DIR = os.path.join(
    ROOT,
    "target_hmol_mordred"
)

GEN_DIR = os.path.join(
    ROOT,
    "gen_hmol_mordred"
)


# ------------------------------------------------------------
# Frozen GIN encoder
#
# MUST correspond to the Hmol representation used to train the
# downstream Hmol+Mordred base model.
# ------------------------------------------------------------

ENCODER_PATH = os.path.join(
    CONTRAST_ROOT,
    "outputs",
    "gin_encoder_last.pt"
)

# If using scaffold-weighted pretraining instead:
#
# ENCODER_PATH = os.path.join(
#     CONTRAST_ROOT,
#     "outputs_scaffold_weighted",
#     "gin_encoder_last.pt"
# )


# ------------------------------------------------------------
# Used only for overlap diagnostics.
# ------------------------------------------------------------

PRETRAIN_MOLECULE_CSV = os.path.join(
    CONTRAST_ROOT,
    "data",
    "unique_molecules.csv"
)

SUPERVISED_TRAIN_SMILES_PATH = os.path.join(
    PRE_DATA_ROOT,
    "gen_hmol_mordred_molecule_split",
    "np_seed29",
    "smiles_train.csv"
)


# ------------------------------------------------------------
# USP vocabulary MUST be the vocabulary of the pretrained
# downstream RT model.
# ------------------------------------------------------------

USP_MAP_PATH = os.path.join(
    PRE_DATA_ROOT,
    "processed",
    "usp_embedding.pth"
)


SEEDS = [29]

TRAIN_VAL_RATIO = 0.85

TRAIN_WITHIN_TRAINVAL_RATIO = 0.70 / 0.85

HMOL_BATCH_SIZE = 256


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


set_seed(
    SEEDS[
        0
    ]
)


# ============================================================
# 3. Import exact pretraining graph/model code
# ============================================================

if CONTRAST_ROOT not in sys.path:

    sys.path.insert(
        0,
        CONTRAST_ROOT
    )


from Contrast_learning.dataset.molecular_dataset import (  # noqa: E402
    smiles_to_graph
)

from Contrast_learning.models.gin_encoder import (  # noqa: E402
    GINEncoder
)


# ============================================================
# 4. Utilities
# ============================================================

def safe_mkdir(
    path
):

    os.makedirs(
        path,
        exist_ok=True
    )


def canonicalize_smiles(
    smi
):

    if pd.isna(
        smi
    ):

        return None

    try:

        mol = Chem.MolFromSmiles(
            str(
                smi
            )
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


def read_csv_safe(
    path,
    header=None
):

    if not os.path.exists(
        path
    ):

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
            f"[ERROR] Failed to read {path}: {exc}"
        )

        return None


def save_matrix(
    path,
    arr
):

    np.savetxt(
        path,
        np.asarray(
            arr
        ),
        delimiter=","
    )


def save_vector(
    path,
    arr,
    fmt=None
):

    arr = np.asarray(
        arr
    )

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
            delimiter=",",
            fmt=fmt
        )


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


# ============================================================
# 5. f1:
#    independent preprocessing
#
#    - canonicalize SMILES
#    - remove invalid SMILES
#    - remove RT==0
#    - remove duplicate canonical molecules WITHIN dataset
#
# We keep only the first measurement of a duplicated molecule,
# matching the broad intent of the original Tune preprocessing.
# ============================================================

def f1():

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 1: Independent Hmol+Mordred Tune preprocessing"
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

    if not files:

        raise RuntimeError(
            "No Tune dataset .txt files found."
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
                f"[SKIP] {dataset_id}: molecular files missing."
            )

            continue

        if len(
            maccs
        ) != len(
            mordred
        ):

            raise RuntimeError(
                f"{dataset_id}: MACCS/Mordred row mismatch.\n"
                f"MACCS={len(maccs)}, Mordred={len(mordred)}"
            )

        smiles_maccs = (
            maccs.iloc[
                :,
                0
            ]
            .astype(str)
            .tolist()
        )

        smiles_mordred = (
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
                "MACCS and Mordred."
            )

        canonical_maccs = [
            canonicalize_smiles(
                s
            )
            for s in smiles_maccs
        ]

        canonical_mordred = [
            canonicalize_smiles(
                s
            )
            for s in smiles_mordred
        ]

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

                raise RuntimeError(
                    f"{dataset_id}: SMILES mismatch after "
                    f"canonicalization at row {i}.\n"
                    f"MACCS={smi_a}\n"
                    f"Mordred={smi_b}"
                )

        seen = set()

        keep = []

        kept_smiles = []

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

            if float(
                rt
            ) == 0.0:

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

            kept_smiles.append(
                can_smi
            )

        print(
            f"  original rows          = {len(maccs)}"
        )

        print(
            f"  invalid SMILES removed = {invalid_count}"
        )

        print(
            f"  RT==0 removed          = {zero_rt_count}"
        )

        print(
            f"  duplicate molecules    = {duplicate_count}"
        )

        print(
            f"  kept rows              = {len(keep)}"
        )

        if not keep:

            print(
                f"[SKIP] {dataset_id}: no valid rows."
            )

            continue

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

        # Ensure the saved identity is canonical.
        maccs_f.iloc[
            :,
            0
        ] = kept_smiles

        mordred_f.iloc[
            :,
            0
        ] = kept_smiles

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
                    kept_smiles
            }
        ).to_csv(
            os.path.join(
                save_dir,
                "canonical_smiles.csv"
            ),
            index=False
        )


# ============================================================
# 6. f2:
#    independent random 81:9:10 split
# ============================================================

def my_split(
    n,
    seed
):

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

    train_index = perm[
        :train_num
    ]

    val_index = perm[
        train_num:
        train_val_num
    ]

    test_index = perm[
        train_val_num:
    ]

    return (
        train_index,
        val_index,
        test_index
    )


def f2():

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 2: Independent 81:9:10 Tune split"
    )

    print(
        "=" * 80
    )

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
        if os.path.isdir(
            os.path.join(
                TARGET_DIR,
                d
            )
        )
    )

    for seed in SEEDS:

        for dataset_id in dataset_ids:

            save_dir = os.path.join(
                TARGET_DIR,
                dataset_id
            )

            maccs_file = os.path.join(
                save_dir,
                f"{dataset_id}_maccs.csv"
            )

            if not os.path.exists(
                maccs_file
            ):

                continue

            n = len(
                pd.read_csv(
                    maccs_file,
                    header=None
                )
            )

            (
                tr,
                va,
                te
            ) = my_split(
                n,
                seed
            )

            split_dir = os.path.join(
                save_dir,
                "split_index",
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

            print(
                f"{dataset_id}: "
                f"train={len(tr)}, "
                f"val={len(va)}, "
                f"test={len(te)}"
            )


# ============================================================
# 7. Load frozen GIN
# ============================================================

def normalize_state_dict_keys(
    state_dict
):

    if (
        state_dict
        and
        all(
            str(
                k
            ).startswith(
                "module."
            )
            for k in state_dict.keys()
        )
    ):

        return {
            str(
                k
            )[
                len(
                    "module."
                ):
            ]:
                v
            for k, v in state_dict.items()
        }

    return state_dict


def extract_encoder_state_dict(
    checkpoint
):

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "encoder_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "encoder_state_dict"
        ]

    else:

        state_dict = checkpoint

    if not isinstance(
        state_dict,
        dict
    ):

        raise RuntimeError(
            "Could not interpret encoder checkpoint."
        )

    return normalize_state_dict_keys(
        state_dict
    )


def infer_encoder_dims(
    state_dict
):

    if (
        "fc.weight"
        not in state_dict
    ):

        raise RuntimeError(
            "fc.weight was not found in GIN checkpoint."
        )

    output_dim = int(
        state_dict[
            "fc.weight"
        ].shape[
            0
        ]
    )

    hidden_dim = int(
        state_dict[
            "fc.weight"
        ].shape[
            1
        ]
    )

    example_graph = smiles_to_graph(
        "CC"
    )

    input_dim = int(
        example_graph.x.shape[
            1
        ]
    )

    return (
        input_dim,
        hidden_dim,
        output_dim
    )


def load_frozen_encoder():

    if not os.path.exists(
        ENCODER_PATH
    ):

        raise FileNotFoundError(
            ENCODER_PATH
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else
        "cpu"
    )

    checkpoint = torch.load(
        ENCODER_PATH,
        map_location=device
    )

    state_dict = extract_encoder_state_dict(
        checkpoint
    )

    (
        input_dim,
        hidden_dim,
        output_dim
    ) = infer_encoder_dims(
        state_dict
    )

    encoder = GINEncoder(
        input_dim=
            input_dim,

        hidden_dim=
            hidden_dim,

        output_dim=
            output_dim
    ).to(
        device
    )

    encoder.load_state_dict(
        state_dict,
        strict=True
    )

    encoder.eval()

    print(
        "\nFrozen GIN loaded:"
    )

    print(
        "  encoder:",
        ENCODER_PATH
    )

    print(
        "  device:",
        device
    )

    print(
        "  dims:",
        {
            "input_dim":
                input_dim,

            "hidden_dim":
                hidden_dim,

            "hmol_dim":
                output_dim
        }
    )

    return (
        encoder,
        device,
        output_dim
    )


ENCODER, DEVICE, HMOL_DIM = (
    load_frozen_encoder()
)


# ============================================================
# 8. Hmol inference for any new molecule
# ============================================================

def encode_smiles_to_hmol(
    canonical_smiles
):

    graphs = [
        smiles_to_graph(
            smi
        )
        for smi in canonical_smiles
    ]

    all_h = []

    with torch.no_grad():

        for start in range(
            0,
            len(
                graphs
            ),
            HMOL_BATCH_SIZE
        ):

            batch = Batch.from_data_list(
                graphs[
                    start:
                    start
                    +
                    HMOL_BATCH_SIZE
                ]
            ).to(
                DEVICE
            )

            h = ENCODER(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            all_h.append(
                h.cpu()
            )

    Hmol = (
        torch.cat(
            all_h,
            dim=0
        )
        .numpy()
        .astype(
            np.float32
        )
    )

    if Hmol.shape[
        1
    ] != HMOL_DIM:

        raise RuntimeError(
            f"Unexpected Hmol dimension: {Hmol.shape}"
        )

    return Hmol


# ============================================================
# 9. Reference molecule sets for diagnostics only
# ============================================================

def load_smiles_set_from_csv(
    path
):

    if not os.path.exists(
        path
    ):

        return set()

    df = pd.read_csv(
        path
    )

    if (
        "canonical_smiles"
        not in df.columns
    ):

        return set()

    out = set()

    for smi in df[
        "canonical_smiles"
    ].dropna():

        can_smi = canonicalize_smiles(
            smi
        )

        if can_smi is not None:

            out.add(
                can_smi
            )

    return out


PRETRAIN_MOLECULES = (
    load_smiles_set_from_csv(
        PRETRAIN_MOLECULE_CSV
    )
)

SUPERVISED_TRAIN_MOLECULES = (
    load_smiles_set_from_csv(
        SUPERVISED_TRAIN_SMILES_PATH
    )
)


# ============================================================
# 10. USP vocabulary
# ============================================================

if not os.path.exists(
    USP_MAP_PATH
):

    raise FileNotFoundError(
        USP_MAP_PATH
    )


usp_checkpoint = torch.load(
    USP_MAP_PATH,
    map_location="cpu"
)


if (
    "word2id"
    not in usp_checkpoint
):

    raise KeyError(
        "usp_embedding.pth does not contain word2id."
    )


WORD2ID = usp_checkpoint[
    "word2id"
]


# ============================================================
# 11. Build one dataset's Hmol+Mordred features
# ============================================================

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

    if len(
        maccs
    ) != len(
        mordred
    ):

        raise RuntimeError(
            f"{dataset_id}: MACCS/Mordred row mismatch."
        )

    canonical_smiles = (
        maccs.iloc[
            :,
            0
        ]
        .astype(str)
        .values
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

    # MACCS descriptors are NOT used here.
    X_hmol = encode_smiles_to_hmol(
        canonical_smiles
    )

    X_mordred = (
        mordred.iloc[
            :,
            2:
        ]
        .values
        .astype(
            np.float32
        )
    )

    if not np.isfinite(
        X_mordred
    ).all():

        raise RuntimeError(
            f"{dataset_id}: Mordred contains NaN/Inf."
        )

    n = len(
        y
    )

    # Column
    col = np.loadtxt(
        os.path.join(
            COL_VEC_DIR,
            f"{dataset_id}.csv"
        ),
        delimiter=","
    )

    col = (
        np.atleast_1d(
            col
        )
        .astype(
            np.float32
        )
        .reshape(
            -1
        )
    )

    X_col = np.tile(
        col.reshape(
            1,
            -1
        ),
        (
            n,
            1
        )
    )

    # Gradient
    grad = np.loadtxt(
        os.path.join(
            GRAD_VEC_DIR,
            f"{dataset_id}_gradient.csv"
        ),
        delimiter=","
    )

    grad = (
        np.atleast_1d(
            grad
        )
        .astype(
            np.float32
        )
        .reshape(
            -1
        )
    )

    X_grad = np.tile(
        grad.reshape(
            1,
            -1
        ),
        (
            n,
            1
        )
    )

    # Meta
    meta_df = pd.read_csv(
        os.path.join(
            META_VEC_DIR,
            f"{dataset_id}.csv"
        )
    )

    if len(
        meta_df
    ) == 0:

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

    X_meta = np.tile(
        meta.reshape(
            1,
            -1
        ),
        (
            n,
            1
        )
    )

    X = np.hstack(
        [
            X_hmol,
            X_mordred,
            X_col,
            X_grad,
            X_meta
        ]
    ).astype(
        np.float32
    )

    if not np.isfinite(
        X
    ).all():

        raise RuntimeError(
            f"{dataset_id}: X contains NaN/Inf."
        )

    # USP
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
            f"{dataset_id}: USP '{usp}' is not in the "
            "training USP vocabulary."
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

    seen_pretrain = np.asarray(
        [
            smi in PRETRAIN_MOLECULES
            for smi in canonical_smiles
        ],
        dtype=bool
    )

    seen_rt_train = np.asarray(
        [
            smi in SUPERVISED_TRAIN_MOLECULES
            for smi in canonical_smiles
        ],
        dtype=bool
    )

    layout = {
        "hmol_dim":
            int(
                X_hmol.shape[
                    1
                ]
            ),

        "mordred_dim":
            int(
                X_mordred.shape[
                    1
                ]
            ),

        "column_dim":
            int(
                X_col.shape[
                    1
                ]
            ),

        "gradient_dim":
            int(
                X_grad.shape[
                    1
                ]
            ),

        "meta_dim":
            int(
                X_meta.shape[
                    1
                ]
            ),

        "total_dim":
            int(
                X.shape[
                    1
                ]
            )
    }

    return (
        X,
        y,
        usp_index,
        canonical_smiles,
        X_hmol,
        seen_pretrain,
        seen_rt_train,
        layout
    )


# ============================================================
# 12. f3:
#     build/save features using the NEW independently generated
#     Tune split
# ============================================================

def f3():

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 3: Build Hmol+Mordred Tune features"
    )

    print(
        "=" * 80
    )

    dataset_ids = sorted(
        d
        for d in os.listdir(
            TARGET_DIR
        )
        if os.path.isdir(
            os.path.join(
                TARGET_DIR,
                d
            )
        )
    )

    for seed in SEEDS:

        X_train_all = []

        y_train_all = []

        usp_train_all = []

        X_val_all = []

        y_val_all = []

        usp_val_all = []

        feature_layout = None

        overlap_rows = []

        processed_datasets = []

        for dataset_id in dataset_ids:

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
                canonical_smiles,
                X_hmol,
                seen_pretrain,
                seen_rt_train,
                dims
            ) = build_features(
                dataset_id
            )

            if feature_layout is None:

                feature_layout = dims

            elif dims != feature_layout:

                raise RuntimeError(
                    f"{dataset_id}: feature layout mismatch.\n"
                    f"Current={dims}\n"
                    f"Expected={feature_layout}"
                )

            split_dir = os.path.join(
                TARGET_DIR,
                dataset_id,
                "split_index",
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

            X_tr = X[
                tr
            ]

            y_tr = y[
                tr
            ]

            usp_tr = usp_index[
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

            X_te = X[
                te
            ]

            y_te = y[
                te
            ]

            usp_te = usp_index[
                te
            ]

            smiles_tr = canonical_smiles[
                tr
            ]

            smiles_va = canonical_smiles[
                va
            ]

            smiles_te = canonical_smiles[
                te
            ]

            out_dir = os.path.join(
                GEN_DIR,
                f"np_seed{seed}",
                dataset_id
            )

            safe_mkdir(
                out_dir
            )

            # train
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

            # val
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

            # test
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

            # all rows for pure no-tune external evaluation
            save_matrix(
                os.path.join(
                    out_dir,
                    "X_all.csv"
                ),
                X
            )

            save_vector(
                os.path.join(
                    out_dir,
                    "y_all.csv"
                ),
                y
            )

            save_vector(
                os.path.join(
                    out_dir,
                    "usp_all.csv"
                ),
                usp_index,
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

            pd.DataFrame(
                {
                    "canonical_smiles":
                        canonical_smiles,

                    "seen_in_gin_pretraining":
                        seen_pretrain,

                    "seen_in_supervised_rt_train":
                        seen_rt_train
                }
            ).to_csv(
                os.path.join(
                    out_dir,
                    "smiles_all.csv"
                ),
                index=False
            )

            hmol_cols = [
                f"Hmol_{i:03d}"
                for i in range(
                    X_hmol.shape[
                        1
                    ]
                )
            ]

            hmol_df = pd.DataFrame(
                X_hmol,
                columns=
                    hmol_cols
            )

            hmol_df.insert(
                0,
                "canonical_smiles",
                canonical_smiles
            )

            hmol_df.to_csv(
                os.path.join(
                    out_dir,
                    "Hmol_all.csv"
                ),
                index=False
            )

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

            overlap_rows.append(
                {
                    "dataset_id":
                        dataset_id,

                    "n_total":
                        len(
                            X
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

                    "seen_in_gin_pretraining_n":
                        int(
                            seen_pretrain.sum()
                        ),

                    "seen_in_gin_pretraining_fraction":
                        float(
                            seen_pretrain.mean()
                        ),

                    "seen_in_supervised_rt_train_n":
                        int(
                            seen_rt_train.sum()
                        ),

                    "seen_in_supervised_rt_train_fraction":
                        float(
                            seen_rt_train.mean()
                        )
                }
            )

            processed_datasets.append(
                dataset_id
            )

            print(
                f"  X={X.shape}, "
                f"train={len(tr)}, "
                f"val={len(va)}, "
                f"test={len(te)}"
            )

        global_dir = os.path.join(
            GEN_DIR,
            f"np_seed{seed}"
        )

        safe_mkdir(
            global_dir
        )

        # pooled train
        if X_train_all:

            X_train_all = np.vstack(
                X_train_all
            )

            y_train_all = np.concatenate(
                y_train_all
            )

            usp_train_all = np.concatenate(
                usp_train_all
            )

            rng = np.random.RandomState(
                seed
            )

            order = rng.permutation(
                len(
                    X_train_all
                )
            )

            X_train_all = X_train_all[
                order
            ]

            y_train_all = y_train_all[
                order
            ]

            usp_train_all = usp_train_all[
                order
            ]

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

        # pooled val
        if X_val_all:

            X_val_all = np.vstack(
                X_val_all
            )

            y_val_all = np.concatenate(
                y_val_all
            )

            usp_val_all = np.concatenate(
                usp_val_all
            )

            rng = np.random.RandomState(
                seed
            )

            order = rng.permutation(
                len(
                    X_val_all
                )
            )

            X_val_all = X_val_all[
                order
            ]

            y_val_all = y_val_all[
                order
            ]

            usp_val_all = usp_val_all[
                order
            ]

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

        with open(
            os.path.join(
                global_dir,
                "feature_layout.json"
            ),
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                feature_layout,
                f,
                indent=2,
                ensure_ascii=False
            )

        pd.DataFrame(
            overlap_rows
        ).to_csv(
            os.path.join(
                global_dir,
                "overlap_summary.csv"
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
            "\nFinished."
        )

        print(
            "Output:",
            global_dir
        )

        print(
            "Feature layout:",
            feature_layout
        )


# ============================================================
# 13. Main
# ============================================================

if __name__ == "__main__":

    f1()

    f2()

    f3()

    print(
        "\nAll finished."
    )
