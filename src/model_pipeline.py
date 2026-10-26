
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler



NUMERIC_FEATURES = [
    "UNDER_CONSTRUCTION",
    "RERA",
    "BHK_NO.",
    "SQUARE_FT",
    "LOG_SQUARE_FT",
    "SQUARE_FT_CLEANED",
    "LOG_SQUARE_FT_CLEANED",
    "READY_TO_MOVE",
    "RESALE",
    "GEO_LATITUDE",
    "GEO_LONGITUDE",
    "AREA_OVER_10000_SQFT",
]


CATEGORICAL_FEATURES = [
    "POSTED_BY",
    "BHK_OR_RK",
    "LOCALITY_CANDIDATE",
    "CITY_CANDIDATE",
]


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build preprocessing fitted only on the data passed to model.fit()."""

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
    ]

    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
