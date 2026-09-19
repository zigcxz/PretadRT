import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from rdkit import Chem
import pandas as pd


ATOM_TYPES = [
    "C", "N", "O", "S", "F",
    "Cl", "Br", "I", "P"
]


def atom_features(atom):
    """
    9 atom-type one-hot
    + degree
    + formal charge
    + aromatic
    = 12 dimensions
    """

    symbol = atom.GetSymbol()

    feat = [
        int(symbol == a)
        for a in ATOM_TYPES
    ]

    feat.append(atom.GetDegree())
    feat.append(atom.GetFormalCharge())
    feat.append(int(atom.GetIsAromatic()))

    return feat


def smiles_to_graph(smiles):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            f"Invalid SMILES: {smiles}"
        )

    # -------------------------
    # Node features
    # -------------------------

    x = torch.tensor(
        [
            atom_features(atom)
            for atom in mol.GetAtoms()
        ],
        dtype=torch.float32
    )

    # -------------------------
    # Edges
    # -------------------------

    edges = []

    for bond in mol.GetBonds():

        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        edges.append([i, j])
        edges.append([j, i])

    if len(edges) > 0:

        edge_index = torch.tensor(
            edges,
            dtype=torch.long
        ).t().contiguous()

    else:

        # 单原子分子等情况
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long
        )

    return Data(
        x=x,
        edge_index=edge_index
    )


class MoleculeGraphDataset(Dataset):

    def __init__(self, molecule_csv):

        self.df = pd.read_csv(
            molecule_csv
        ).reset_index(drop=True)

        required_cols = {
            "mol_id",
            "canonical_smiles"
        }

        missing = (
            required_cols
            - set(self.df.columns)
        )

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        self.smiles = (
            self.df["canonical_smiles"]
            .tolist()
        )

        self.mol_ids = (
            self.df["mol_id"]
            .tolist()
        )

        print(
            f"Building molecular graphs "
            f"for {len(self.smiles)} molecules..."
        )

        self.graphs = []

        for i, smi in enumerate(self.smiles):

            graph = smiles_to_graph(smi)

            # 保存分子索引
            graph.mol_idx = i

            self.graphs.append(graph)

            if (i + 1) % 1000 == 0:
                print(
                    f"Graphs built: "
                    f"{i + 1}/{len(self.smiles)}"
                )

        print("Graph construction complete.")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx].clone()