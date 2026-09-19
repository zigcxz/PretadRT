import os
import numpy as np
import pandas as pd
from model import build_DNN
from config import *
from sklearn.preprocessing import StandardScaler
import joblib
import json


# ============================================================
# Global molecule-split integrity check
# ============================================================

def check_molecule_split_integrity():

    if not os.path.exists(TRAIN_SMILES_PATH):
        raise FileNotFoundError(
            f"Missing train molecule manifest:\n{TRAIN_SMILES_PATH}"
        )

    if not os.path.exists(VAL_SMILES_PATH):
        raise FileNotFoundError(
            f"Missing validation molecule manifest:\n{VAL_SMILES_PATH}"
        )

    train_smiles = set(
        pd.read_csv(TRAIN_SMILES_PATH)["canonical_smiles"]
        .dropna()
        .astype(str)
    )

    val_smiles = set(
        pd.read_csv(VAL_SMILES_PATH)["canonical_smiles"]
        .dropna()
        .astype(str)
    )

    overlap = train_smiles & val_smiles

    if overlap:
        raise RuntimeError(
            "GLOBAL train/validation molecule overlap detected.\n"
            f"Overlap count = {len(overlap)}\n"
            f"Examples = {list(overlap)[:20]}"
        )

    print("\nGlobal molecule-split check: PASS")
    print(f"Train unique molecules: {len(train_smiles)}")
    print(f"Val unique molecules  : {len(val_smiles)}")
    print("Train/val molecule overlap: 0")
def load_data():


    X_train=pd.read_csv(
        X_PATH,
        header=None
    ).values



    USP_train=pd.read_csv(
        USP_PATH,
        header=None
    ).values.astype(int)



    y_train=pd.read_csv(
        Y_PATH,
        header=None
    ).values[:,0]




    X_val=pd.read_csv(
        VAL_X_PATH,
        header=None
    ).values



    USP_val=pd.read_csv(
        VAL_USP_PATH,
        header=None
    ).values.astype(int)




    y_val=pd.read_csv(
        VAL_Y_PATH,
        header=None
    ).values[:,0]



    return (
        X_train,
        USP_train,
        y_train,
        X_val,
        USP_val,
        y_val
    )



def train():

    check_molecule_split_integrity()

    (
        X_train,
        USP_train,
        y_train,
        X_val,
        USP_val,
        y_val
    ) = load_data()

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    print("train:",X_train.shape)
    print("val:",X_val.shape)

    import torch

    checkpoint = torch.load(
        r"F:\RepoRT-master (2)\Pre_data\processed\usp_embedding.pth",
        map_location="cpu"
    )

    usp_num = len(
        checkpoint["word2id"]
    )

    model = build_DNN(
        X_train.shape[1],
        usp_num,
        lr=0.001
    )

    history = model.fit(
        [
            X_train,
            USP_train
        ],
        y_train,

        validation_data=
        (
            [
                X_val,
                USP_val
            ],
            y_val
        ),

        epochs=320,

        batch_size=64,

        verbose=2
    )

    model.save(
        os.path.join(
            MODEL_DIR,
            "base_model.keras"
        )
    )

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

    pd.DataFrame(
        history.history
    ).to_csv(
        os.path.join(
            MODEL_DIR,
            "loss_history.csv"
        ),
        index=False
    )


    print("MACCS + Mordred molecule-disjoint baseline training finished")


    with open(
            os.path.join(
                MODEL_DIR,
                "usp_config.json"
            ),
            "w"
    ) as f:
        json.dump(
            {
                "usp_num": int(usp_num),
                "rt_split": "global_canonical_molecule_disjoint",
                "molecular_representation": "MACCS + Mordred",
                "condition_representation": "column + gradient + meta + USP embedding",
                "scaler_fit": "train_only",
                "epochs": 320,
                "batch_size": 64,
                "learning_rate": 0.001,
                "loss": "mae"
            },
            f,
            indent=2
        )

if __name__=="__main__":
    train()
