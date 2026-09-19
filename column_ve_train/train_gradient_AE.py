import os
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import random
from GradientAE import GradientAE
import numpy as np
SEED = 29

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
# =========================
# 修改路径
# =========================

GRADIENT_DIR = r"F:\RepoRT-master (2)\Tune_231\origin\original_gradient"

SAVE_DIR = r"F:\RepoRT-master (2)\Tune_231\processed\gradient_vector"

MODEL_DIR = r"F:\RepoRT-master (2)\Tune_231\processed\models"


os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# 只读取这四列
FEATURES = [
    "t [min]",
    "A [%]",
    "B [%]",
    "flow rate [ml/min]"
]


def read_gradient_data():

    data=[]

    files=os.listdir(GRADIENT_DIR)

    for file in files:

        if file.endswith(".tsv") or file.endswith(".txt"):

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


            data.append(
                torch.tensor(
                    df.values,
                    dtype=torch.float32
                )
            )


    return data



def train():

    set_seed(SEED)
    gradient_data=read_gradient_data()

    print("gradient数量:",len(gradient_data))


    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    model=GradientAE(device)


    criterion=nn.MSELoss()

    optimizer=optim.Adam(
        model.parameters(),
        lr=1e-3
    )


    best_loss=999999


    for epoch in range(200):

        total_loss=0


        for x in gradient_data:

            x=x.to(device)


            pred=model(x)


            loss=criterion(
                pred,
                x
            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            total_loss+=loss.item()


        loss=total_loss/len(gradient_data)


        if epoch%20==0:

            print(
                "epoch:",
                epoch,
                "loss:",
                loss
            )


        if loss<best_loss:

            best_loss=loss

            torch.save(
                model.state_dict(),
                os.path.join(
                    MODEL_DIR,
                    "gradient_AE.pth"
                )
            )


    print("AE训练完成")
    print("best loss:",best_loss)



def extract_vector():

    device=torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    model=GradientAE(device)


    model.load_state_dict(
        torch.load(
            os.path.join(
                MODEL_DIR,
                "gradient_AE.pth"
            ),
            map_location=device
        )
    )


    model.eval()


    for file in os.listdir(GRADIENT_DIR):

        if file.endswith(".tsv") or file.endswith(".txt"):

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
                df=pd.read_csv(path)


            df=df[FEATURES].fillna(0)


            x=torch.tensor(
                df.values,
                dtype=torch.float32
            ).to(device)



            with torch.no_grad():

                z=model.encoder(x)


                # 一个gradient程序压缩成一个向量
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


    print("gradient vector提取完成")



if __name__=="__main__":
    set_seed(SEED)
    train()

    extract_vector()
