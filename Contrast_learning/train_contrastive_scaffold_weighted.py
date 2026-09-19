# ============================================================
# Contrastive molecular representation pretraining
#
# Chemical-prior Contrastive-MDL-TL
#
# Inputs:
#   data/unique_molecules.csv
#   data/chemical_positive_topk.csv
#
# Outputs:
#   outputs/gin_encoder_last.pt
#   outputs/projector_last.pt
#   outputs/contrastive_checkpoint.pt
#   outputs/hmol_embeddings.csv
#   outputs/hmol_embeddings.npy
#   outputs/training_log.csv
# ============================================================


import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from torch_geometric.loader import DataLoader


from dataset.molecular_dataset import (
    MoleculeGraphDataset
)

from sampler.chemical_positive_sampler import (
    ChemicalPositiveBatchSampler
)

from models.gin_encoder import (
    GINEncoder
)

from models.projector import (
    ProjectionHead
)

from losses.weighted_supcon import (
    weighted_supcon_loss
)


# ============================================================
# 1. Configuration
# ============================================================

SEED = 42

EPOCHS = 200

ANCHOR_BATCH_SIZE = 64

MAX_CHEMICAL_POSITIVE = 2

MASK_RATIO = 0.10

HIDDEN_DIM = 256

HMOL_DIM = 256

PROJECTION_DIM = 128

TEMPERATURE = 0.07

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

GRAD_CLIP = 5.0

# Scaffold-frequency anchor weighting
USE_SCAFFOLD_ANCHOR_WEIGHT = True
SCAFFOLD_WEIGHT_POWER = 0.5
EMPTY_SCAFFOLD_WEIGHT = 1.0


# ============================================================
# 2. Paths
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

DATA_DIR = (
    BASE_DIR
    /
    "data"
)

OUTPUT_DIR = (
    BASE_DIR
    /
    "outputs_scaffold_weighted"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MOLECULE_CSV = (
    DATA_DIR
    /
    "unique_molecules.csv"
)

POSITIVE_CSV = (
    DATA_DIR
    /
    "chemical_positive_topk.csv"
)


# ============================================================
# 3. Random seed
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


set_seed(SEED)


# ============================================================
# 4. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else
    "cpu"
)

print("=" * 70)
print("Chemical-prior Contrastive Pretraining")
print("=" * 70)

print(
    f"Device: {device}"
)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# 5. Dataset
# ============================================================

dataset = MoleculeGraphDataset(
    str(MOLECULE_CSV)
)

print(
    f"\nDataset size: "
    f"{len(dataset)} molecules"
)


# 自动读取输入维数
INPUT_DIM = int(
    dataset[0].x.shape[1]
)

print(
    f"Atom feature dimension: "
    f"{INPUT_DIM}"
)


# ============================================================
# 6. Build scaffold-frequency anchor weights
# ============================================================

def compute_murcko_scaffold(smiles):
    """Fallback if unique_molecules.csv has no scaffold column."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(
            mol=mol, includeChirality=True
        )
    except Exception:
        return ""


def build_scaffold_anchor_weights(
    molecule_csv,
    dataset,
    power=0.5,
    empty_scaffold_weight=1.0
):
    """
    For each non-empty scaffold s:
        a_i = 1 / n_s**power

    With power=0.5:
        a_i = 1 / sqrt(n_s)

    Empty scaffold is assigned weight 1.0 instead of treating all
    acyclic/no-Murcko molecules as one giant scaffold group.
    """
    molecule_df = pd.read_csv(molecule_csv).reset_index(drop=True)

    if len(molecule_df) != len(dataset):
        raise RuntimeError(
            f"CSV rows={len(molecule_df)} but dataset={len(dataset)}"
        )

    required = {"mol_id", "canonical_smiles"}
    missing = required - set(molecule_df.columns)
    if missing:
        raise ValueError(f"unique_molecules.csv missing columns: {missing}")

    if molecule_df["mol_id"].tolist() != list(dataset.mol_ids):
        raise RuntimeError(
            "mol_id order mismatch between CSV and MoleculeGraphDataset."
        )

    if "scaffold" not in molecule_df.columns:
        print("\n[INFO] Computing Bemis-Murcko scaffolds...")
        molecule_df["scaffold"] = molecule_df["canonical_smiles"].apply(
            compute_murcko_scaffold
        )

    molecule_df["scaffold"] = molecule_df["scaffold"].fillna("").astype(str)

    non_empty = molecule_df["scaffold"] != ""
    scaffold_counts = molecule_df.loc[non_empty, "scaffold"].value_counts()

    sizes = np.ones(len(molecule_df), dtype=np.int64)
    weights = np.full(
        len(molecule_df), float(empty_scaffold_weight), dtype=np.float32
    )

    for i, scaffold in enumerate(molecule_df["scaffold"].tolist()):
        if scaffold == "":
            sizes[i] = 1
            weights[i] = float(empty_scaffold_weight)
        else:
            n_s = int(scaffold_counts[scaffold])
            sizes[i] = n_s
            weights[i] = float(n_s ** (-power))

    if not np.isfinite(weights).all():
        raise RuntimeError("Scaffold weights contain NaN/Inf.")
    if np.any(weights <= 0):
        raise RuntimeError("Scaffold weights must be positive.")

    weight_df = molecule_df[["mol_id", "canonical_smiles", "scaffold"]].copy()
    weight_df["scaffold_size_for_weight"] = sizes
    weight_df["anchor_weight"] = weights
    weight_df.to_csv(OUTPUT_DIR / "scaffold_anchor_weights.csv", index=False)

    print("\n" + "=" * 70)
    print("Scaffold-frequency anchor weights")
    print("=" * 70)
    print(f"Formula: w_i = 1 / n_scaffold^{power}")
    print(f"Non-empty scaffold types: {len(scaffold_counts)}")
    print(f"Empty-scaffold molecules: {int((~non_empty).sum())}")
    print(f"Empty-scaffold weight: {empty_scaffold_weight}")
    print("\nWeight summary:")
    print(pd.Series(weights).describe(
        percentiles=[0.01,0.05,0.25,0.50,0.75,0.95,0.99]
    ).to_string())

    if len(scaffold_counts) > 0:
        top_df = scaffold_counts.head(20).rename("molecule_count").reset_index()
        top_df["anchor_weight"] = top_df["molecule_count"].astype(float) ** (-power)
        top_df.to_csv(OUTPUT_DIR / "top_scaffold_weight_diagnostics.csv", index=False)
        print("\nTop scaffold sizes and weights:")
        print(top_df.to_string(index=False))

    return torch.tensor(weights, dtype=torch.float32)


molecule_scaffold_weights = build_scaffold_anchor_weights(
    molecule_csv=MOLECULE_CSV,
    dataset=dataset,
    power=SCAFFOLD_WEIGHT_POWER,
    empty_scaffold_weight=EMPTY_SCAFFOLD_WEIGHT
).to(device)


# ============================================================
# 7. Chemical-aware sampler
# No sampler modification is needed; it already returns graph_mol_ids.
# ============================================================

sampler = (
    ChemicalPositiveBatchSampler(
        dataset=dataset,

        positive_csv=
            str(POSITIVE_CSV),

        anchor_batch_size=
            ANCHOR_BATCH_SIZE,

        max_positive=
            MAX_CHEMICAL_POSITIVE,

        augmentation_ratio=
            MASK_RATIO,

        seed=SEED
    )
)

print(
    f"Anchor batches / epoch: "
    f"{len(sampler)}"
)


# ============================================================
# 7. Model
# ============================================================

encoder = GINEncoder(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=HMOL_DIM
).to(device)


projector = ProjectionHead(
    dim=HMOL_DIM,
    projection_dim=PROJECTION_DIM
).to(device)


print("\nEncoder:")
print(encoder)

print("\nProjection head:")
print(projector)


# ============================================================
# 8. Optimizer
# ============================================================

optimizer = torch.optim.AdamW(

    list(
        encoder.parameters()
    )
    +
    list(
        projector.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


scheduler = (
    torch.optim.lr_scheduler
    .CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )
)


# ============================================================
# 9. Training
# ============================================================

history = []

best_train_loss = float(
    "inf"
)


for epoch in range(
    1,
    EPOCHS + 1
):

    encoder.train()
    projector.train()

    sampler.set_epoch(
        epoch
    )

    epoch_loss = 0.0

    num_batches = 0

    epoch_graphs = 0

    epoch_positive_pairs = 0


    for step, batch in enumerate(
        sampler,
        start=1
    ):

        graph = (
            batch["graph"]
            .to(device)
        )

        positive_mask = (
            batch[
                "positive_mask"
            ]
            .to(device)
        )

        positive_weights = (
            batch[
                "positive_weights"
            ]
            .to(device)
        )

        graph_mol_ids = batch["graph_mol_ids"].to(device)

        if USE_SCAFFOLD_ANCHOR_WEIGHT:
            anchor_weights = molecule_scaffold_weights[graph_mol_ids]
        else:
            anchor_weights = torch.ones(
                graph_mol_ids.numel(),
                dtype=torch.float32,
                device=device
            )


        optimizer.zero_grad(
            set_to_none=True
        )


        # ====================================
        # Molecular encoder
        # ====================================

        h = encoder(
            graph.x,
            graph.edge_index,
            graph.batch
        )

        # h = Hmol
        #
        # [number of molecular graphs,
        #  256]


        # ====================================
        # Projection head
        # ====================================

        z = projector(
            h
        )

        # ====================================
        # Weighted SupCon
        # ====================================

        loss = weighted_supcon_loss(
            features=z,

            positive_mask=
                positive_mask,

            positive_weights=
                positive_weights,

            anchor_weights=
                anchor_weights,

            temperature=
                TEMPERATURE
        )


        if torch.isnan(loss):

            raise RuntimeError(
                "NaN loss detected."
            )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(

            list(
                encoder.parameters()
            )
            +
            list(
                projector.parameters()
            ),

            max_norm=GRAD_CLIP
        )


        optimizer.step()


        epoch_loss += (
            loss.item()
        )

        num_batches += 1


        epoch_graphs += (
            batch["num_graphs"]
        )


        # symmetric matrix,
        # 因此除2得到pair数
        epoch_positive_pairs += int(
            positive_mask.sum().item()
            //
            2
        )


        if (
            step == 1
            and
            epoch == 1
        ):

            print(
                "\n===== First batch diagnostic ====="
            )

            print(
                f"Anchors: "
                f"{batch['num_anchor']}"
            )

            print(
                f"Unique original molecules: "
                f"{batch['num_original']}"
            )

            print(
                f"Total graph views: "
                f"{batch['num_graphs']}"
            )

            print(
                f"Hmol shape: "
                f"{tuple(h.shape)}"
            )

            print(
                f"Projection shape: "
                f"{tuple(z.shape)}"
            )

            print(
                "Positive relation pairs: "
                f"{positive_mask.sum().item() // 2}"
            )

            total_possible = (
                positive_mask.size(0)
                *
                (
                    positive_mask.size(0)
                    - 1
                )
                //
                2
            )

            positive_pairs = (
                positive_mask.sum().item()
                //
                2
            )

            negative_pairs = (
                total_possible
                -
                positive_pairs
            )

            print(
                f"In-batch negative pairs "
                f"(approx): "
                f"{negative_pairs}"
            )

            print(
                "Scaffold anchor weights: "
                f"min={anchor_weights.min().item():.6f}, "
                f"mean={anchor_weights.mean().item():.6f}, "
                f"max={anchor_weights.max().item():.6f}"
            )

            print(
                "==================================\n"
            )


    mean_loss = (
        epoch_loss
        /
        max(num_batches, 1)
    )


    scheduler.step()


    lr = optimizer.param_groups[
        0
    ]["lr"]


    history.append({
        "epoch": epoch,
        "loss": mean_loss,
        "lr": lr,
        "graphs": epoch_graphs,
        "positive_pairs":
            epoch_positive_pairs
    })


    print(
        f"Epoch "
        f"{epoch:03d}/{EPOCHS} | "
        f"Loss={mean_loss:.6f} | "
        f"LR={lr:.2e}"
    )


    # ========================================
    # Save lowest training-loss checkpoint
    # 仅用于工程备份，不作为正式模型选择依据
    # ========================================

    if mean_loss < best_train_loss:

        best_train_loss = mean_loss

        torch.save(
            {
                "epoch": epoch,

                "encoder_state_dict":
                    encoder.state_dict(),

                "projector_state_dict":
                    projector.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    mean_loss,

                "config": {
                    "input_dim":
                        INPUT_DIM,

                    "hidden_dim":
                        HIDDEN_DIM,

                    "hmol_dim":
                        HMOL_DIM,

                    "projection_dim":
                        PROJECTION_DIM,

                    "temperature":
                        TEMPERATURE,

                    "scaffold_anchor_weighting":
                        USE_SCAFFOLD_ANCHOR_WEIGHT,

                    "scaffold_weight_power":
                        SCAFFOLD_WEIGHT_POWER,

                    "empty_scaffold_weight":
                        EMPTY_SCAFFOLD_WEIGHT
                }
            },

            OUTPUT_DIR
            /
            "best_train_checkpoint.pt"
        )


    # ========================================
    # Periodic checkpoints
    # ========================================

    if epoch % 20 == 0:

        torch.save(
            encoder.state_dict(),

            OUTPUT_DIR
            /
            f"gin_encoder_epoch_{epoch}.pt"
        )


# ============================================================
# 10. Save final models
# ============================================================

torch.save(
    encoder.state_dict(),

    OUTPUT_DIR
    /
    "gin_encoder_last.pt"
)


torch.save(
    projector.state_dict(),

    OUTPUT_DIR
    /
    "projector_last.pt"
)


torch.save(
    {
        "encoder_state_dict":
            encoder.state_dict(),

        "projector_state_dict":
            projector.state_dict(),

        "config": {
            "input_dim":
                INPUT_DIM,

            "hidden_dim":
                HIDDEN_DIM,

            "hmol_dim":
                HMOL_DIM,

            "projection_dim":
                PROJECTION_DIM,

            "temperature":
                TEMPERATURE,

            "scaffold_anchor_weighting":
                USE_SCAFFOLD_ANCHOR_WEIGHT,

            "scaffold_weight_power":
                SCAFFOLD_WEIGHT_POWER,

            "empty_scaffold_weight":
                EMPTY_SCAFFOLD_WEIGHT
        }
    },

    OUTPUT_DIR
    /
    "contrastive_checkpoint.pt"
)


# ============================================================
# 11. Save training history
# ============================================================

history_df = pd.DataFrame(
    history
)

history_df.to_csv(
    OUTPUT_DIR
    /
    "training_log.csv",

    index=False
)


# ============================================================
# 12. Loss curve
# ============================================================

plt.figure(
    figsize=(7, 5)
)

plt.plot(
    history_df["epoch"],
    history_df["loss"]
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Scaffold-weighted SupCon Loss"
)

plt.title(
    "Contrastive pretraining loss"
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR
    /
    "training_loss.png",

    dpi=300
)

plt.close()


# ============================================================
# 13. Generate final Hmol
#
# IMPORTANT:
# downstream RT prediction uses Hmol
# NOT projection z.
# ============================================================

print(
    "\nGenerating final Hmol embeddings..."
)


embedding_loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=False
)


encoder.eval()

all_h = []


with torch.no_grad():

    for graph_batch in embedding_loader:

        graph_batch = (
            graph_batch.to(
                device
            )
        )

        h = encoder(
            graph_batch.x,
            graph_batch.edge_index,
            graph_batch.batch
        )

        all_h.append(
            h.cpu()
        )


Hmol = torch.cat(
    all_h,
    dim=0
).numpy()


print(
    f"Hmol matrix shape: "
    f"{Hmol.shape}"
)


# ============================================================
# 14. Save NPY
# ============================================================

np.save(
    OUTPUT_DIR
    /
    "hmol_embeddings.npy",

    Hmol
)


# ============================================================
# 15. Save CSV
# ============================================================

hmol_columns = [
    f"Hmol_{i:03d}"
    for i in range(
        Hmol.shape[1]
    )
]


hmol_df = pd.DataFrame(
    Hmol,
    columns=hmol_columns
)


molecule_meta = pd.read_csv(
    MOLECULE_CSV
)


hmol_df.insert(
    0,
    "canonical_smiles",
    molecule_meta[
        "canonical_smiles"
    ].values
)


hmol_df.insert(
    0,
    "mol_id",
    molecule_meta[
        "mol_id"
    ].values
)


hmol_df.to_csv(
    OUTPUT_DIR
    /
    "hmol_embeddings.csv",

    index=False
)


print("\n" + "=" * 70)
print("TRAINING FINISHED")
print("=" * 70)

print(
    f"Best training loss: "
    f"{best_train_loss:.6f}"
)

print(
    f"Hmol shape: "
    f"{Hmol.shape}"
)

print(
    "\nOutputs:"
)

print(
    OUTPUT_DIR
    /
    "gin_encoder_last.pt"
)

print(
    OUTPUT_DIR
    /
    "contrastive_checkpoint.pt"
)

print(
    OUTPUT_DIR
    /
    "hmol_embeddings.npy"
)

print(
    OUTPUT_DIR
    /
    "hmol_embeddings.csv"
)

print(
    OUTPUT_DIR
    /
    "training_log.csv"
)

print(
    OUTPUT_DIR
    /
    "training_loss.png"
)

print(
    "Scaffold-weight diagnostics:\n",
    OUTPUT_DIR / "scaffold_anchor_weights.csv",
    "\n",
    OUTPUT_DIR / "top_scaffold_weight_diagnostics.csv"
)
