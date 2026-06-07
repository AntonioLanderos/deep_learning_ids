from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def build_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(100, activation="relu"),
        Dense(100, activation="relu"),
        Dense(100, activation="relu"),

        Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer = Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"]
    )

    return model
