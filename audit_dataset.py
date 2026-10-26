
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

FILES = {
    "train": DATA_DIR / "train.csv",
    "test": DATA_DIR / "test.csv",
    "sample_submission": DATA_DIR / "sample_submission.csv",
}


def inspect_dataset(name: str, file_path: Path) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print(f"DATASET: {name.upper()}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes.to_string()}")
    print(f"\nFirst 5 rows:\n{df.head().to_string()}")
    print(f"\nMissing values:\n{df.isnull().sum().to_string()}")
    print(f"\nDuplicate rows: {df.duplicated().sum()}")

    return df


def main() -> None:
    datasets = {}

    for name, file_path in FILES.items():
        if not file_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        datasets[name] = inspect_dataset(name, file_path)

    train = datasets["train"]
    test = datasets["test"]
    submission = datasets["sample_submission"]

    target = "TARGET(PRICE_IN_LACS)"

    print("\n" + "=" * 70)
    print("TRAINING TARGET ANALYSIS")
    print("=" * 70)

    if target in train.columns:
        prices = pd.to_numeric(train[target], errors="coerce")

        print(f"Valid numeric prices: {prices.notna().sum()}")
        print(f"Invalid or missing prices: {prices.isna().sum()}")

        if prices.notna().any():
            print("\nPrice statistics (in dataset units):")
            print(prices.describe(percentiles=[0.01, 0.25, 0.50, 0.75, 0.99]))
            print(f"\nNon-positive prices: {(prices.dropna() <= 0).sum()}")
    else:
        print(f"WARNING: Expected target column {target!r} was not found.")

    print("\n" + "=" * 70)
    print("TRAIN / TEST SCHEMA COMPARISON")
    print("=" * 70)

    print("Columns only in train:", sorted(set(train.columns) - set(test.columns)))
    print("Columns only in test:", sorted(set(test.columns) - set(train.columns)))

    print("\nSubmission columns:", submission.columns.tolist())
    print("\nDataset audit completed.")


if __name__ == "__main__":
    main()
