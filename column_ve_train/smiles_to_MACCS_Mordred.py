import os
import numpy as np
try:
    np.float = np.float64
except AttributeError:
    pass

try:
    np.int = np.int_
except AttributeError:
    pass
import pandas as pd
import csv
import math

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from mordred import Calculator, descriptors


# ===============================
# 路径
# ===============================

INPUT_DIR = r"F:\RepoRT-master (2)\Tune_231\smiles_rt_data"

OUTPUT_DIR = r"F:\RepoRT-master (2)\Tune_231\processed\molecular_vector"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IDX_MORDRED_PATH = r"F:\RepoRT-master (2)\Pre_data\idx_mordred.csv"

# 读取需要保留的 Mordred 列索引
idx_mordred = np.loadtxt(IDX_MORDRED_PATH, delimiter=",").astype(int)
print("Mordred 保留列数:", len(idx_mordred))
# ===============================
# MACCS
# ===============================

def cal_MACCS(smiles_list, rt_list, mols):

    result=[]

    for i,mol in enumerate(mols):

        fps = MACCSkeys.GenMACCSKeys(mol)

        arr=np.zeros((1,))

        DataStructs.ConvertToNumpyArray(
            fps,
            arr
        )

        arr=[int(x) for x in arr]

        # 去掉MACCS第0位占位符
        result.append(
            [smiles_list[i], rt_list[i]]
            +
            arr[1:]
        )

    return np.array(result, dtype=object)



# ===============================
# Mordred
# ===============================

def cal_Mordred(smiles_list, rt_list, mols):

    calc = Calculator(
        descriptors,
        ignore_3D=True
    )

    result = []

    for i, mol in enumerate(mols):

        values = []

        desc = calc(mol)

        for x in desc.values():

            try:
                x = float(x)
                if math.isnan(x):
                    values.append(0)
                else:
                    values.append(x)
            except:
                values.append(0)

        # ============ 关键修改 ============
        # 只保留 idx_mordred 指定位置的描述符
        values = np.array(values)[idx_mordred].tolist()
        # ==================================

        result.append(
            [smiles_list[i], rt_list[i]]
            + values
        )

    return np.array(result, dtype=object)



def standardize_mordred(data):
    smiles = data[:, 0]
    rt = data[:, 1]
    feature = data[:, 2:].astype(float)
    mean = np.mean(feature, axis=0)
    std = np.std(feature, axis=0)
    std[std == 0] = 1
    feature = (feature - mean) / std
    return np.concatenate(
        [
            smiles.reshape(-1, 1),
            rt.reshape(-1, 1),
            feature
        ],
        axis=1
    )



# ===============================
# 主程序
# ===============================


if __name__=="__main__":


    files=os.listdir(INPUT_DIR)


    for file in files:


        if not file.endswith(".txt"):
            continue


        print("="*40)
        print("processing:",file)


        path=os.path.join(
            INPUT_DIR,
            file
        )


        df = pd.read_csv(path, sep=None, engine='python')


        # 你的数据列名:
        # smiles
        # rt

        smiles=df["smiles"].astype(str).tolist()

        rt=df["rt"].astype(float).tolist()



        mols=[]

        valid_smiles=[]
        valid_rt=[]


        for s,r in zip(smiles,rt):

            mol=Chem.MolFromSmiles(s)

            if mol is not None:

                mols.append(mol)
                valid_smiles.append(s)
                valid_rt.append(r)

            else:

                print("invalid smiles:",s)



        print(
            "valid molecules:",
            len(mols)
        )


        # MACCS

        maccs=cal_MACCS(
            valid_smiles,
            valid_rt,
            mols
        )


        pd.DataFrame(maccs).to_csv(
            os.path.join(
                OUTPUT_DIR,
                file.replace(
                    ".txt",
                    "_MACCS.csv"
                )
            ),
            index=False,
            header=False
        )


        print(
            "MACCS:",
            maccs.shape
        )


        # Mordred

        mordred=cal_Mordred(
            valid_smiles,
            valid_rt,
            mols
        )


        mordred_std=standardize_mordred(
            mordred
        )


        pd.DataFrame(mordred_std).to_csv(
            os.path.join(
                OUTPUT_DIR,
                file.replace(
                    ".txt",
                    "_Mordred_std.csv"
                )
            ),
            index=False,
            header=False
        )


        print(
            "Mordred:",
            mordred_std.shape
        )


    print("All finished")
