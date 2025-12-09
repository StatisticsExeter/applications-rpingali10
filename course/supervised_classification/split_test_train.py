import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def test_and_train():
    base_data_path = Path("data_cache") / "energy.csv"

    X_train_path = Path("data_cache") / "energy_X_train.csv"
    y_train_path = Path("data_cache") / "energy_y_train.csv"
    X_test_path  = Path("data_cache") / "energy_X_test.csv"
    y_test_path  = Path("data_cache") / "energy_y_test.csv"

    split_data(base_data_path, X_train_path, y_train_path, X_test_path, y_test_path)

    return None   


def split_data(base_data_path, X_train_path, y_train_path, X_test_path, y_test_path):
    df = pd.read_csv(base_data_path)

    df = df.dropna(subset=['built_age'])

    y = df['built_age']
    X = df.drop(columns=['built_age'])

    X = X.fillna(X.mean(numeric_only=True))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_train.to_csv(X_train_path, index=False)
    y_train.to_csv(y_train_path, index=False)
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
