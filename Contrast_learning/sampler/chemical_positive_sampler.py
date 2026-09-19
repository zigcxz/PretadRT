import math
import random
import pandas as pd
import torch

from torch_geometric.data import Batch


# ============================================================
# Graph augmentation
# ============================================================

def graph_augmentation(
    data,
    mask_ratio=0.10
):
    """
    Atom-feature masking.

    保持图拓扑不变，只mask部分atom feature。
    """

    aug = data.clone()

    num_nodes = aug.x.size(0)

    if num_nodes <= 1:
        return aug

    num_mask = max(
        1,
        int(round(num_nodes * mask_ratio))
    )

    num_mask = min(
        num_mask,
        num_nodes - 1
    )

    perm = torch.randperm(
        num_nodes
    )[:num_mask]

    aug.x[perm] = 0.0

    return aug


# ============================================================
# Chemical-aware batch sampler
# ============================================================

class ChemicalPositiveBatchSampler:

    def __init__(
        self,
        dataset,
        positive_csv,
        anchor_batch_size=64,
        max_positive=2,
        augmentation_ratio=0.10,
        seed=42
    ):

        self.dataset = dataset

        self.anchor_batch_size = (
            anchor_batch_size
        )

        self.max_positive = max_positive

        self.augmentation_ratio = (
            augmentation_ratio
        )

        self.seed = seed
        self.epoch = 0

        self.indices = list(
            range(len(dataset))
        )

        # ================================================
        # Load chemical relation graph
        # ================================================

        pos_df = pd.read_csv(
            positive_csv
        )

        required = {
            "anchor_idx",
            "positive_idx",
            "contrastive_weight"
        }

        missing = (
            required
            - set(pos_df.columns)
        )

        if missing:
            raise ValueError(
                f"positive csv missing: {missing}"
            )

        self.pos_dict = {}

        # 用于判断任意batch内两分子
        # 是否存在已知chemical relation
        self.relation_weight = {}

        for _, row in pos_df.iterrows():

            i = int(row["anchor_idx"])
            j = int(row["positive_idx"])

            w = float(
                row["contrastive_weight"]
            )

            pair_type = (
                row["type"]
                if "type" in pos_df.columns
                else "chemical"
            )

            self.pos_dict.setdefault(
                i,
                []
            ).append(
                {
                    "idx": j,
                    "weight": w,
                    "type": pair_type
                }
            )

            # 对contrastive loss而言关系对称处理
            key = (
                min(i, j),
                max(i, j)
            )

            # 如果两个方向都有，
            # 保留较大的confidence
            if key not in self.relation_weight:

                self.relation_weight[key] = w

            else:

                self.relation_weight[key] = max(
                    self.relation_weight[key],
                    w
                )

        print(
            f"Chemical relation anchors: "
            f"{len(self.pos_dict)}"
        )

        print(
            f"Unique relation pairs: "
            f"{len(self.relation_weight)}"
        )

    # ========================================================
    # 为每个epoch改变随机顺序
    # ========================================================

    def set_epoch(self, epoch):
        self.epoch = epoch

    # ========================================================
    # Iterator
    # ========================================================

    def __iter__(self):

        rng = random.Random(
            self.seed + self.epoch
        )

        indices = self.indices.copy()

        rng.shuffle(indices)

        for start in range(
            0,
            len(indices),
            self.anchor_batch_size
        ):

            anchors = indices[
                start:
                start + self.anchor_batch_size
            ]

            yield self.make_batch(
                anchors,
                rng
            )

    def __len__(self):

        return math.ceil(
            len(self.indices)
            /
            self.anchor_batch_size
        )

    # ========================================================
    # Sample chemical positives
    # ========================================================

    def sample_positives(
        self,
        anchor_idx,
        rng
    ):

        candidates = (
            self.pos_dict.get(
                anchor_idx,
                []
            )
        )

        if len(candidates) <= self.max_positive:
            return candidates.copy()

        # ----------------------------------------
        # 根据contrastive_weight进行加权抽样
        # without replacement
        # ----------------------------------------

        remaining = candidates.copy()

        selected = []

        while (
            len(selected) < self.max_positive
            and len(remaining) > 0
        ):

            weights = [
                max(
                    float(x["weight"]),
                    1e-8
                )
                for x in remaining
            ]

            chosen = rng.choices(
                remaining,
                weights=weights,
                k=1
            )[0]

            selected.append(chosen)

            remaining.remove(chosen)

        return selected

    # ========================================================
    # Build batch
    # ========================================================

    def make_batch(
        self,
        anchor_indices,
        rng
    ):

        # ----------------------------------------
        # Step 1
        # 每个anchor抽chemical positives
        # ----------------------------------------

        sampled_chemical = {}

        for i in anchor_indices:

            sampled_chemical[i] = (
                self.sample_positives(
                    i,
                    rng
                )
            )

        # ----------------------------------------
        # Step 2
        # 建立 original molecule集合
        #
        # anchor + chemical positive
        #
        # 去重！！！
        # ----------------------------------------

        original_molecule_ids = []

        seen = set()

        # Anchor先进入
        for idx in anchor_indices:

            if idx not in seen:

                original_molecule_ids.append(idx)

                seen.add(idx)

        # Chemical positives再进入
        for anchor_idx in anchor_indices:

            for p in sampled_chemical[
                anchor_idx
            ]:

                p_idx = int(p["idx"])

                if p_idx not in seen:

                    original_molecule_ids.append(
                        p_idx
                    )

                    seen.add(
                        p_idx
                    )

        # ----------------------------------------
        # Step 3
        # original graph
        # ----------------------------------------

        graphs = []

        graph_mol_ids = []

        view_types = []

        # molecule idx -> graph位置
        original_position = {}

        for mol_idx in original_molecule_ids:

            pos = len(graphs)

            original_position[
                mol_idx
            ] = pos

            graphs.append(
                self.dataset[mol_idx]
            )

            graph_mol_ids.append(
                mol_idx
            )

            view_types.append(
                "original"
            )

        # ----------------------------------------
        # Step 4
        # 每个anchor生成self augmentation
        # ----------------------------------------

        augmentation_position = {}

        for anchor_idx in anchor_indices:

            aug = graph_augmentation(
                self.dataset[
                    anchor_idx
                ],
                mask_ratio=
                self.augmentation_ratio
            )

            pos = len(graphs)

            augmentation_position[
                anchor_idx
            ] = pos

            graphs.append(
                aug
            )

            # underlying molecule仍然是anchor
            graph_mol_ids.append(
                anchor_idx
            )

            view_types.append(
                "augmentation"
            )

        # ----------------------------------------
        # Step 5
        # PyG Batch
        # ----------------------------------------

        pyg_batch = (
            Batch.from_data_list(
                graphs
            )
        )

        num_graphs = len(graphs)

        # ----------------------------------------
        # Step 6
        # positive mask / weight matrix
        # ----------------------------------------

        positive_mask = torch.zeros(
            (
                num_graphs,
                num_graphs
            ),
            dtype=torch.bool
        )

        positive_weights = torch.zeros(
            (
                num_graphs,
                num_graphs
            ),
            dtype=torch.float32
        )

        # ========================================
        # P0:
        # anchor-original
        # +
        # self-augmentation
        # ========================================

        for anchor_idx in anchor_indices:

            p1 = original_position[
                anchor_idx
            ]

            p2 = augmentation_position[
                anchor_idx
            ]

            positive_mask[
                p1,
                p2
            ] = True

            positive_mask[
                p2,
                p1
            ] = True

            positive_weights[
                p1,
                p2
            ] = 1.0

            positive_weights[
                p2,
                p1
            ] = 1.0

        # ========================================
        # Chemical relations
        #
        # 对batch中的任意两个graph：
        #
        # 如果underlying molecule之间存在
        # chemical relation，则设为positive
        #
        # 这样还能避免known-positive被错误
        # 当成negative。
        # ========================================

        for a in range(num_graphs):

            mol_a = graph_mol_ids[a]

            for b in range(
                a + 1,
                num_graphs
            ):

                mol_b = graph_mol_ids[b]

                # --------------------------------
                # 同一分子的不同view
                # --------------------------------

                if mol_a == mol_b:

                    positive_mask[a, b] = True
                    positive_mask[b, a] = True

                    positive_weights[a, b] = 1.0
                    positive_weights[b, a] = 1.0

                    continue

                # --------------------------------
                # chemical relation
                # --------------------------------

                key = (
                    min(mol_a, mol_b),
                    max(mol_a, mol_b)
                )

                if key in self.relation_weight:

                    w = float(
                        self.relation_weight[key]
                    )

                    positive_mask[a, b] = True
                    positive_mask[b, a] = True

                    positive_weights[a, b] = w
                    positive_weights[b, a] = w

        # ----------------------------------------
        # diagonal永远不是positive
        # ----------------------------------------

        positive_mask.fill_diagonal_(
            False
        )

        positive_weights.fill_diagonal_(
            0.0
        )

        return {
            "graph": pyg_batch,

            "positive_mask":
                positive_mask,

            "positive_weights":
                positive_weights,

            "graph_mol_ids":
                torch.tensor(
                    graph_mol_ids,
                    dtype=torch.long
                ),

            "anchor_indices":
                torch.tensor(
                    anchor_indices,
                    dtype=torch.long
                ),

            "num_anchor":
                len(anchor_indices),

            "num_original":
                len(
                    original_molecule_ids
                ),

            "num_graphs":
                num_graphs
        }