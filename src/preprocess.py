import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def load_data(train_path, test_path):
  train_df = pd.read_csv(train_path)
  test_df = pd.read_csv(test_path)

  return train_df, test_df

def preprocess(train_df, test_df):
  excluded_cols = ["id", "label", "attack_cat"]

  train_features = train_df.drop(columns=excluded_cols)
  test_features = test_df.drop(columns=excluded_cols)

  categorical_cols = train_features.select_dtypes(include=["object"]).columns.tolist()

  train_labels_text = train_df["attack_cat"]
  test_labels_text = test_df["attack_cat"]

  label_encoder = LabelEncoder()
  train_labels = label_encoder.fit_transform(train_labels_text)
  test_labels = label_encoder.transform(test_labels_text)

  train_features_encoded = pd.get_dummies(train_features, columns=categorical_cols)
  test_features_encoded = pd.get_dummies(test_features, columns=categorical_cols)

  train_features_encoded, test_features_encoded = train_features_encoded.align(
      test_features_encoded,
      join="left",
      axis=1,
      fill_value=0
  )

  train_features_encoded = train_features_encoded.astype(float)
  test_features_encoded = test_features_encoded.astype(float)

  scaler = StandardScaler()
  train_features_scaled = scaler.fit_transform(train_features_encoded)
  test_features_scaled = scaler.transform(test_features_encoded)

  train_columns = train_features_encoded.columns.tolist()

  return train_features_scaled, train_labels, test_features_scaled, test_labels, label_encoder, scaler, train_columns

def create_validation_split(train_features, train_labels):
  return train_test_split(
      train_features,
      train_labels,
      test_size=0.2,
      random_state=42,
      stratify=train_labels
  )

def save_artifacts(scaler, label_encoder, train_columns, scaler_path, label_encoder_path, train_columns_path):
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, label_encoder_path)
    joblib.dump(train_columns, train_columns_path)


def load_artifacts(scaler_path, label_encoder_path, train_columns_path):
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    train_columns = joblib.load(train_columns_path)

    return scaler, label_encoder, train_columns
