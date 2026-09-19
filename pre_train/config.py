import os


ROOT = r"F:\RepoRT-master (2)\Pre_data"


DATA_DIR = os.path.join(
    ROOT,
    "gen_molecule_split",
    "np_seed29"
)


# 连续变量
X_PATH = os.path.join(
    DATA_DIR,
    "X_train.csv"
)


VAL_X_PATH = os.path.join(
    DATA_DIR,
    "X_val.csv"
)



# USP类别index

USP_PATH = os.path.join(
    DATA_DIR,
    "usp_train.csv"
)


VAL_USP_PATH = os.path.join(
    DATA_DIR,
    "usp_val.csv"
)



# RT

Y_PATH = os.path.join(
    DATA_DIR,
    "y_train.csv"
)


VAL_Y_PATH = os.path.join(
    DATA_DIR,
    "y_val.csv"
)

TRAIN_SMILES_PATH = os.path.join(
    DATA_DIR,
    "smiles_train.csv"
)

VAL_SMILES_PATH = os.path.join(
    DATA_DIR,
    "smiles_val.csv"
)

MODEL_DIR=os.path.join(
    ROOT,
    "base_model_maccs_mordred_molecule_split"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)