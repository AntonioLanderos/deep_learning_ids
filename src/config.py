import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "raw", "UNSW_NB15_training-set.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "raw", "UNSW_NB15_testing-set.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

MODEL_PATH = os.path.join(MODEL_DIR, "paper_dnn_model.keras")

SCALER_PATH = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "label_encoder.pkl")
TRAIN_COLUMNS_PATH = os.path.join(ARTIFACTS_DIR, "train_columns.pkl")
