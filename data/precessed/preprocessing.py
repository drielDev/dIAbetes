import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def load_data(path: str) -> pd.DataFrame:
    """
    Load diabetes dataset from CSV file.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path_obj} (resolved: {path_obj.resolve()})"
        )

    return pd.read_csv(path_obj)


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


def preprocess_data(csv_path: str):
    """
    Full preprocessing pipeline:
    - load data
    - clean data
    - split dataset
    - scale features
    """
    df = load_data(csv_path)
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
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_PATH = PROJECT_ROOT / "data" / "raw" / "diabetes.csv"

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
    ) = preprocess_data(DATA_PATH)

    print("Preprocessing completed successfully.")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
