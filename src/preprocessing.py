from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Resolve project root dynamically
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load diabetes dataset from CSV file.
    """
    file_path = DATA_DIR / filename
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace invalid zero values with NaN and
    impute missing values using the median.
    """
    cols_with_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    df = df.copy()
    df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
    df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].median())

    return df


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    val_size: float = 0.5,
    random_state: int = 42
):
    """
    Split data into train, validation and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessing_pipeline(features: list):
    """
    Build preprocessing pipeline for numeric features.
    """
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, features)
        ]
    )

    return preprocessor


def preprocess_data(filename: str = "diabetes.csv"):
    """
    Full preprocessing pipeline:
    - load data
    - clean data
    - split dataset
    - scale features
    """
    df = load_data(filename)
    df = clean_data(df)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = build_preprocessing_pipeline(X.columns.tolist())

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        preprocessor
    )


if __name__ == "__main__":
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _
    ) = preprocess_data()

    print("Preprocessing completed successfully.")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
