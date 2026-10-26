
import numpy as np
import pandas as pd


TARGET_COLUMN = "TARGET(PRICE_IN_LACS)"

INDIA_LAT_MIN = 6.0
INDIA_LAT_MAX = 38.0
INDIA_LON_MIN = 68.0
INDIA_LON_MAX = 98.0

AREA_THRESHOLD_SQFT = 10_000


def prepare_property_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create model-ready raw features without modifying the input DataFrame.

    Coordinate interpretation:
        LATITUDE column  -> corrected longitude
        LONGITUDE column -> corrected latitude

    Suspicious coordinates become missing values for later pipeline
    imputation. Original coordinate columns remain unchanged.

    Area handling:
        SQUARE_FT: original reported area, preserved.
        LOG_SQUARE_FT: log-transformed original area.
        SQUARE_FT_CLEANED: areas above the threshold become missing.
        LOG_SQUARE_FT_CLEANED: log-transformed cleaned area.

    The cleaned-area features support a separate sensitivity experiment.
    No property rows are deleted.
    """
    data = df.copy()

    required_columns = {
        "POSTED_BY",
        "UNDER_CONSTRUCTION",
        "RERA",
        "BHK_NO.",
        "BHK_OR_RK",
        "SQUARE_FT",
        "READY_TO_MOVE",
        "RESALE",
        "ADDRESS",
        "LONGITUDE",
        "LATITUDE",
    }

    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Parse address candidates while retaining the original address.
    address_parts = (
        data["ADDRESS"]
        .astype("string")
        .str.rsplit(",", n=1, expand=True)
    )

    data["LOCALITY_CANDIDATE"] = address_parts[0].str.strip()

    if address_parts.shape[1] > 1:
        data["CITY_CANDIDATE"] = address_parts[1].str.strip()
    else:
        data["CITY_CANDIDATE"] = pd.NA

    # Correct the coordinate interpretation into named features.
    data["GEO_LATITUDE"] = pd.to_numeric(
        data["LONGITUDE"], errors="coerce"
    )
    data["GEO_LONGITUDE"] = pd.to_numeric(
        data["LATITUDE"], errors="coerce"
    )

    # Keep coordinates within broad India bounds; do not delete listings.
    inside_india = (
        data["GEO_LATITUDE"].between(
            INDIA_LAT_MIN, INDIA_LAT_MAX
        )
        & data["GEO_LONGITUDE"].between(
            INDIA_LON_MIN, INDIA_LON_MAX
        )
    )

    data.loc[
        ~inside_india,
        ["GEO_LATITUDE", "GEO_LONGITUDE"],
    ] = np.nan

    # Convert area to numeric without changing the original column.
    area = pd.to_numeric(data["SQUARE_FT"], errors="coerce")

    # Flag unusually large reported areas without deleting records.
    data["AREA_OVER_10000_SQFT"] = (
        area > AREA_THRESHOLD_SQFT
    ).astype("int8")

    # Original-area feature engineering.
    valid_area = area.where(area >= 0)
    data["LOG_SQUARE_FT"] = np.log1p(valid_area)

    # Sensitivity experiment: treat very large reported areas as unreliable.
    cleaned_area = area.where(
        area.between(0, AREA_THRESHOLD_SQFT)
    )
    data["SQUARE_FT_CLEANED"] = cleaned_area

    data["LOG_SQUARE_FT_CLEANED"] = np.log1p(cleaned_area)

    # Return features only; never include the target in model inputs.
    return data.drop(columns=[TARGET_COLUMN], errors="ignore")
