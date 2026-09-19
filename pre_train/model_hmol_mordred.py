import tensorflow as tf

from tensorflow.keras.layers import (
    Input,
    Dense,
    Embedding,
    Flatten,
    Concatenate
)

from tensorflow.keras.models import Model


# ============================================================
# Hmol + Mordred + Hcond
# MDL-TL downstream DNN
# Split-independent architecture: works with global molecule split
# ============================================================

def build_DNN(
    feature_dim,
    usp_num,
    lr=0.001
):

    # ========================================================
    # Continuous feature input
    #
    # X =
    #
    # Hmol
    # +
    # Mordred
    # +
    # Column
    # +
    # Gradient
    # +
    # Meta
    #
    # feature_dim is automatically determined from X_train
    # ========================================================

    input_x = Input(
        shape=(
            feature_dim,
        ),
        name="feature_input"
    )


    # ========================================================
    # USP categorical index input
    # ========================================================

    input_usp = Input(
        shape=(1,),
        dtype="int32",
        name="usp_input"
    )


    # ========================================================
    # USP embedding
    #
    # Same as Original MDL-TL
    # ========================================================

    usp_emb = Embedding(
        input_dim=usp_num,
        output_dim=5,
        name="USP_embedding"
    )(
        input_usp
    )


    usp_emb = Flatten(
        name="USP_flatten"
    )(
        usp_emb
    )


    # ========================================================
    # Fusion
    #
    # continuous molecular/condition features
    # +
    # USP embedding
    # ========================================================

    x = Concatenate(
        name="feature_fusion"
    )(
        [
            input_x,
            usp_emb
        ]
    )


    # ========================================================
    # SAME MDL-TL prediction head
    #
    # Do not modify here yet.
    #
    # This keeps the comparison controlled.
    # ========================================================

    x = Dense(
        1024,
        activation="relu",
        name="dense_1024"
    )(
        x
    )


    x = Dense(
        512,
        activation="relu",
        name="dense_512"
    )(
        x
    )


    x = Dense(
        256,
        activation="relu",
        name="dense_256"
    )(
        x
    )


    x = Dense(
        64,
        activation="relu",
        name="dense_64"
    )(
        x
    )


    # ========================================================
    # RT output
    # ========================================================

    out = Dense(
        1,
        name="RT"
    )(
        x
    )


    # ========================================================
    # Build model
    # ========================================================

    model = Model(

        inputs=[
            input_x,
            input_usp
        ],

        outputs=out,

        name="Hmol_Mordred_MDLTL"

    )


    # ========================================================
    # Same optimizer / loss as Original
    # ========================================================

    model.compile(

        optimizer=
            tf.keras.optimizers.Adam(
                learning_rate=lr
            ),

        loss="mae"

    )


    return model