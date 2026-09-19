import os
import numpy as np
from gensim.models import Word2Vec
import random


SEED = 29


random.seed(SEED)
np.random.seed(SEED)



# ==========================
# Pre_data Word2Vec
# ==========================

MODEL_PATH = (
    r"F:\RepoRT-master (2)\Pre_data"
    r"\processed\column_name.model"
)



# ==========================
# Tune column name
# ==========================

COLUMN_NAME_DIR = (
    r"F:\RepoRT-master (2)\Tune"
    r"\origin\original_columns_name"
)



SAVE_DIR = (
    r"F:\RepoRT-master (2)\Tune"
    r"\processed\column_name_vector"
)



os.makedirs(
    SAVE_DIR,
    exist_ok=True
)



# ==========================
# extract
# ==========================


def extract_column_vector():


    print(
        "Loading Word2Vec:"
    )

    print(
        MODEL_PATH
    )


    model=Word2Vec.load(
        MODEL_PATH
    )


    print(
        "Word2Vec loaded"
    )


    for file in os.listdir(
        COLUMN_NAME_DIR
    ):


        if not file.endswith(".txt"):
            continue



        path=os.path.join(
            COLUMN_NAME_DIR,
            file
        )


        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            words=f.readline().strip().split()



        vec=np.zeros(
            10,
            dtype=float
        )


        for w in words:


            if w in model.wv:

                vec += model.wv[w]



        save_path=os.path.join(
            SAVE_DIR,
            file.replace(
                ".txt",
                ".csv"
            )
        )


        np.savetxt(
            save_path,
            vec.reshape(
                1,-1
            ),
            delimiter=","
        )


        print(
            file,
            vec.shape
        )



    print(
        "Column vector extraction finished"
    )



if __name__=="__main__":


    extract_column_vector()