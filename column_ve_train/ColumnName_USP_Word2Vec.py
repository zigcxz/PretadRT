import os
import csv
import numpy as np
import torch
from torch import nn
from gensim.models import Word2Vec
import random
SEED = 29

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)   # 如果使用 GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ==========================
# 路径
# ==========================

COLUMN_NAME_DIR = r"F:\RepoRT-master (2)\Tune_231\origin\original_columns_name"

USP_DIR = r"F:\RepoRT-master (2)\Tune_231\origin\original_columns_usp_code"

OUT = r"F:\RepoRT-master (2)\Tune_231\processed"

NAME_VECTOR_DIR = os.path.join(OUT,"column_name_vector")

USP_VECTOR_DIR = os.path.join(OUT,"usp_vector")

FINAL_VECTOR_DIR = os.path.join(OUT,"column_vector")


# ==========================
# 1. Column name Word2Vec
# ==========================

def train_column_name_word2vec():

    sentences=[]

    for f in os.listdir(COLUMN_NAME_DIR):

        if f.endswith(".txt"):

            with open(
                os.path.join(COLUMN_NAME_DIR,f),
                "r",
                encoding="utf-8"
            ) as file:

                text=file.readline().strip()

            sentences.append(text.split())


    model=Word2Vec(
        sentences,
        vector_size=10,
        window=500,
        min_count=1,
        sg=0,
        negative=5,
        workers=4,
        seed=SEED,
    )

    print(os.path.exists(OUT))
    print(os.path.abspath(OUT))

    model.save(
        os.path.join(
            OUT,
            "column_name.model"
        )
    )

    return model



def extract_column_name_vector(model):

    os.makedirs(
        NAME_VECTOR_DIR,
        exist_ok=True
    )


    for f in os.listdir(COLUMN_NAME_DIR):

        if f.endswith(".txt"):

            with open(
                os.path.join(COLUMN_NAME_DIR,f),
                "r",
                encoding="utf-8"
            ) as file:

                words=file.readline().strip().split()


            vec=np.zeros(10)


            for w in words:

                if w in model.wv:

                    vec += model.wv[w]


            np.savetxt(
                os.path.join(
                    NAME_VECTOR_DIR,
                    f.replace(".txt",".csv")
                ),
                vec.reshape(1,-1),
                delimiter=","
            )


# ==========================
# 2. USP code Embedding
# ==========================

class USPEmbedding(nn.Module):

    def __init__(self, num_classes, embedding_dim=5):

        super().__init__()

        self.embedding=nn.Embedding(
            num_classes,
            embedding_dim
        )


    def forward(self,x):

        return self.embedding(x)



def build_usp_dictionary():

    usp_words=[]

    for f in os.listdir(USP_DIR):

        if f.endswith(".txt"):

            with open(
                os.path.join(USP_DIR,f),
                "r",
                encoding="utf-8"
            ) as file:

                usp=file.readline().strip()

            usp_words.append(usp)


    usp_words=list(set(usp_words))


    word2id={
        w:i for i,w in enumerate(usp_words)
    }

    return word2id



def extract_usp_embedding():
    os.makedirs(OUT, exist_ok=True)

    os.makedirs(
        USP_VECTOR_DIR,
        exist_ok=True
    )


    word2id=build_usp_dictionary()


    model=USPEmbedding(
        len(word2id),
        5
    )

    print(os.path.exists(OUT))
    print(os.path.abspath(OUT))

    # 保存USP embedding模型
    torch.save(
        {
        "state_dict":model.state_dict(),
        "word2id":word2id
        },
        os.path.join(
            OUT,
            "usp_embedding.pth"
        )
    )


    for f in os.listdir(USP_DIR):

        if f.endswith(".txt"):

            with open(
                os.path.join(USP_DIR,f),
                "r",
                encoding="utf-8"
            ) as file:

                usp=file.readline().strip()


            idx=torch.tensor(
                [word2id[usp]]
            )


            with torch.no_grad():

                vec=model(idx).numpy()[0]


            np.savetxt(
                os.path.join(
                    USP_VECTOR_DIR,
                    f.replace(".txt",".csv")
                ),
                vec.reshape(1,-1),
                delimiter=","
            )



# ==========================
# 3. concat
# ==========================

def merge():

    os.makedirs(
        FINAL_VECTOR_DIR,
        exist_ok=True
    )


    for f in os.listdir(NAME_VECTOR_DIR):

        name=np.loadtxt(
            os.path.join(NAME_VECTOR_DIR,f),
            delimiter=","
        )


        usp=np.loadtxt(
            os.path.join(USP_VECTOR_DIR,f),
            delimiter=","
        )


        column_vector=np.concatenate(
            [
                name,
                usp
            ]
        )


        np.savetxt(
            os.path.join(
                FINAL_VECTOR_DIR,f
            ),
            column_vector.reshape(1,-1),
            delimiter=","
        )



if __name__=="__main__":

    os.makedirs(OUT,exist_ok=True)


    # column_name:
    # Word2Vec -> 10维

    model=train_column_name_word2vec()

    extract_column_name_vector(model)


    # USP:
    # Embedding -> 5维

    extract_usp_embedding()


    # concat:
    # 10+5=15

    merge()


    print("Finished")
    print("column_name Word2Vec = 10")
    print("USP Embedding = 5")
    print("column vector = 15")
