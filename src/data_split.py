
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocessing import prepare_property_features, TARGET_COLUMN


def load_and_split_data(
    train_path: str = "data/raw/train.csv",
    test_size: float = 0.20,
    random_state: int = 42,
):
    raw_train = pd.read_csv(train_path)

    if TARGET_COLUMN not in raw_train.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    y = pd.to_numeric(raw_train[TARGET_COLUMN], errors="coerce")

    if y.isna().any():
        raise ValueError("Target contains missing or non-numeric values.")

    if (y <= 0).any():
        raise ValueError("Target contains non-positive prices.")

    X = prepare_property_features(raw_train)

    # Keep original row indices so validation predictions
    # can be matched to their source property records.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    return X_train, X_valid, y_train, y_valid
