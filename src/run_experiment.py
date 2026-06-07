import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    TRAIN_PATH,
    TEST_PATH,
    MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH,
    TRAIN_COLUMNS_PATH,
)

from src.preprocess import (
    load_data,
    preprocess,
    create_validation_split,
    save_artifacts
)

from src.model import (
    build_model
    )

from src.train import calculate_class_weights, save_model, train_model
from src.evaluate import evaluate_model, evaluate_predictions, plot_training_curves


def run_experiment(build_model, experiment_name, epochs, batch_size):
    print("=" * 60)
    print(f"Running experiment: {experiment_name}")
    print("=" * 60)

    train_df, test_df = load_data(TRAIN_PATH, TEST_PATH)

    (
        train_features,
        train_labels,
        test_features,
        test_labels,
        label_encoder,
        scaler,
        train_columns
    ) = preprocess(
        train_df,
        test_df,
    )

    (
        train_features_final,
        validation_features,
        train_labels_final,
        validation_labels
    ) = create_validation_split(
        train_features,
        train_labels,
    )

    class_weights = calculate_class_weights(train_labels_final)

    input_dim = train_features.shape[1]
    num_classes = len(label_encoder.classes_)

    model = build_model(input_dim, num_classes)

    checkpoint = save_model(experiment_name)

    history = train_model(
        model,
        train_features_final,
        train_labels_final,
        validation_features,
        validation_labels,
        class_weights,
        epochs,
        batch_size,
        checkpoint
    )

    plot_training_curves(history)

    evaluate_model(model, test_features, test_labels)

    evaluate_predictions(
        model,
        test_features,
        test_labels,
        label_encoder
    )

    model.save(MODEL_PATH)

    save_artifacts(
        scaler,
        label_encoder,
        train_columns,
        SCALER_PATH,
        LABEL_ENCODER_PATH,
        TRAIN_COLUMNS_PATH
    )

    print("Experiment finished successfully.")


if __name__ == "__main__":
    run_experiment(
        build_model=build_model,
        experiment_name="DNN-LR-capWeights",
        epochs=100,
        batch_size=100
    )
