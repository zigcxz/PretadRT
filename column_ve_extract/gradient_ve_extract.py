import os
import pandas as pd
import torch
import numpy as np

from GradientAE import GradientAE



SEED = 29



def set_seed(seed=SEED):

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic=True

    torch.backends.cudnn.benchmark=False



# =========================
# Tune 数据
# =========================

GRADIENT_DIR = (
    r"F:\RepoRT-master (2)\Tune\origin\original_gradient"
)


SAVE_DIR = (
    r"F:\RepoRT-master (2)\Tune\processed\gradient_vector"
)



# =========================
# 预训练 GradientAE
# =========================

MODEL_PATH = (
    r"F:\RepoRT-master (2)\Pre_data\processed\models\gradient_AE.pth"
)



os.makedirs(
    SAVE_DIR,
    exist_ok=True
)



FEATURES = [
    "t [min]",
    "A [%]",
    "B [%]",
    "flow rate [ml/min]"
]



# =========================
# 提取
# =========================

def extract_vector():


    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )



    # 创建同结构AE

    model=GradientAE(device)



    # 加载Pre_data参数

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )


    # 非训练模式

    model.eval()



    print(
        "Loaded pretrained GradientAE:"
        ,
        MODEL_PATH
    )



    for file in os.listdir(GRADIENT_DIR):


        if not(
            file.endswith(".tsv")
            or
            file.endswith(".txt")
        ):
            continue



        path=os.path.join(
            GRADIENT_DIR,
            file
        )



        try:

            df=pd.read_csv(
                path,
                sep="\t"
            )

        except:

            df=pd.read_csv(
                path
            )



        df=df[FEATURES]

        df=df.fillna(0)



        x=torch.tensor(
            df.values,
            dtype=torch.float32
        ).to(device)



        with torch.no_grad():


            # 只使用encoder

            z=model.encoder(x)



            # gradient sequence
            # 压缩为一个vector

            z=torch.mean(
                z,
                dim=0
            )



        z=z.cpu().numpy()



        name=file.split(".")[0]



        pd.DataFrame(
            [z]
        ).to_csv(
            os.path.join(
                SAVE_DIR,
                name+".csv"
            ),
            index=False,
            header=False
        )



        print(
            name,
            "saved",
            z.shape
        )



    print(
        "Tune gradient vector extraction finished"
    )



if __name__=="__main__":


    set_seed()


    extract_vector()