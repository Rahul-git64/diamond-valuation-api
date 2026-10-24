"""Physical geometry feature engineering and strict GIA ordinal encoding pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from diamond_valuation.core.config import AppConfig


class DiamondGeometryEngineer(BaseEstimator, TransformerMixin):
    """Engineer physical stone geometry features: volume, table-to-depth ratio, and depth deviation."""

    ENGINEERED_COLUMNS: tuple[str, ...] = (
        "volume_mm3",
        "table_to_depth_ratio",
        "length_to_width_ratio",
        "depth_deviation_pct",
        "log_carat",
    )

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> DiamondGeometryEngineer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        x = np.maximum(df["x"].astype(float), 0.5)
        y = np.maximum(df["y"].astype(float), 0.5)
        z = np.maximum(df["z"].astype(float), 0.3)
        depth = np.maximum(df["depth"].astype(float), 40.0)
        table = np.maximum(df["table"].astype(float), 40.0)
        carat = np.maximum(df["carat"].astype(float), 0.05)

        df["volume_mm3"] = np.clip(x * y * z, 0.5, 4500.0)
        df["table_to_depth_ratio"] = table / depth
        df["length_to_width_ratio"] = x / y
        expected_depth_pct = (2.0 * z / (x + y)) * 100.0
        df["depth_deviation_pct"] = np.abs(depth - expected_depth_pct)
        df["log_carat"] = np.log1p(carat)
        return df


def build_preprocessing_pipeline(config: AppConfig) -> Pipeline:
    """Build scikit-learn preprocessing pipeline with strict GIA ordinal hierarchy and scaling."""
    engineered_num_cols = list(config.features.numerical_columns) + list(
        DiamondGeometryEngineer.ENGINEERED_COLUMNS
    )
    cat_cols = list(config.features.categorical_columns)

    # Enforce strict worst-to-best GIA ordinal ranking:
    # cut: Fair < Good < Very Good < Premium < Ideal
    # color: J < I < H < G < F < E < D
    # clarity: I1 < SI2 < SI1 < VS2 < VS1 < VVS2 < VVS1 < IF
    gia_categories = [config.features.categories[col] for col in cat_cols]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal_encoder",
                OrdinalEncoder(
                    categories=gia_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, engineered_num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("geometry_engineer", DiamondGeometryEngineer()),
            ("column_transformer", column_transformer),
        ]
    )
