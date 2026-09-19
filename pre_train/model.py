# Baseline DNN for global molecule-disjoint RT prediction
# Continuous X = MACCS + Mordred + Column + Gradient + Meta
# USP remains a separate categorical embedding input.

import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Flatten,
    Concatenate
)
from tensorflow.keras.models import Model



def build_DNN(
        feature_dim,
        usp_num,
        lr=0.001
):


    # =====================
    # 连续特征
    # =====================

    input_x = Input(
        shape=(feature_dim,),
        name="feature_input"
    )


    # =====================
    # USP index
    # =====================

    input_usp = Input(
        shape=(1,),
        dtype="int32",
        name="usp_input"
    )


    # USP embedding

    usp_emb = Embedding(
        input_dim=usp_num,
        output_dim=5,
        name="USP_embedding"
    )(input_usp)


    usp_emb = Flatten()(usp_emb)



    # =====================
    # concat
    # =====================

    x = Concatenate()(
        [
            input_x,
            usp_emb
        ]
    )



    # =====================
    # DNN
    # =====================

    x = Dense(
        1024,
        activation="relu"
    )(x)


    x = Dense(
        512,
        activation="relu"
    )(x)


    x = Dense(
        256,
        activation="relu"
    )(x)


    x = Dense(
        64,
        activation="relu"
    )(x)


    out=Dense(
        1,
        name="RT"
    )(x)



    model=Model(
        [
            input_x,
            input_usp
        ],
        out
    )


    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="mae"
    )


    return model