"""
Dublin housing price and purchase prediction
Single file script you can drop into your repo as: housing_price_models.py

What it does
1 Loads the raw CSV
2 Cleans text fields and converts types
3 Creates total_price
4 Removes high outliers using upper IQR bound
5 Trains regression models to predict total_price
6 Trains classification models to predict buying_or_not_buying

Notes
This script assumes your raw CSV has these columns
property_scope
availability
location
bedrooms
total_sqft
bath
balcony
buying or not buying
BER
Renovation needed
price-per-sqft-$

If your CSV uses slightly different names, update COLUMN_MAP below.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

warnings.filterwarnings("ignore")


COLUMN_MAP = {
    "size": "bedrooms",
    "Renovation needed": "renovation_needed",
    "buying or not buying": "buying_or_not_buying",
}


BER_MAP = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7}


@dataclass
class RegressionResults:
    mae: float
    rmse: float
    r2: float


@dataclass
class ClassificationResults:
    accuracy: float
    report: str
    confusion: np.ndarray


def _safe_strip_lower(x):
    if isinstance(x, str):
        return x.strip().lower()
    return x


def _split_range(value: str) -> Tuple[str, str]:
    if isinstance(value, str) and " - " in value:
        parts = value.split(" - ")
        if len(parts) == 2:
            return parts[0], parts[1]
    return str(value), str(value)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.rename(columns=COLUMN_MAP)

    df = df.applymap(_safe_strip_lower)

    if "availability" in df.columns:
        df["availability"] = df["availability"].apply(
            lambda x: re.sub(r"ready to move", "immediate possession", x, flags=re.IGNORECASE)
            if isinstance(x, str)
            else x
        )

    if "bedrooms" in df.columns:
        df["bedrooms"] = df["bedrooms"].apply(
            lambda x: re.sub(r"\bbed\b|\bbeds\b|\bbedrooms\b|\bbedroom\b|\bbed\b", "", x, flags=re.IGNORECASE)
            if isinstance(x, str)
            else x
        )
        df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")

    if "total_sqft" in df.columns:
        df["total_sqft"] = df["total_sqft"].apply(
            lambda x: re.sub(r"[a-zA-Z]", "", x) if isinstance(x, str) else x
        )

        start_end = df["total_sqft"].apply(_split_range).apply(pd.Series)
        start_end.columns = ["start", "end"]
        start_end["start"] = pd.to_numeric(start_end["start"], errors="coerce")
        start_end["end"] = pd.to_numeric(start_end["end"], errors="coerce")

        df["total_sqft"] = pd.to_numeric(df["total_sqft"], errors="coerce")

        df["total_sqft"] = np.where(
            start_end["start"].notna() & start_end["end"].notna(),
            (start_end["start"] + start_end["end"]) / 2,
            df["total_sqft"],
        )

    if "bath" in df.columns:
        df["bath"] = pd.to_numeric(df["bath"], errors="coerce")

    if "balcony" in df.columns:
        df["balcony"] = pd.to_numeric(df["balcony"], errors="coerce")

    if "price-per-sqft-$" in df.columns:
        df["price-per-sqft-$"] = pd.to_numeric(df["price-per-sqft-$"], errors="coerce").round(2)

    if "ber" in df.columns:
        df["ber"] = df["ber"].map(BER_MAP)

    if "BER" in df.columns:
        df["BER"] = df["BER"].map(BER_MAP)

    if "BER" in df.columns and "ber" not in df.columns:
        df = df.rename(columns={"BER": "ber"})

    if "renovation_needed" in df.columns:
        df["renovation_needed"] = df["renovation_needed"].astype("category")

    if "property_scope" in df.columns:
        df["property_scope"] = df["property_scope"].astype("category")

    if "location" in df.columns:
        df["location"] = df["location"].astype("category")

    df = df.drop_duplicates()

    df = df.dropna()

    if "total_price" not in df.columns:
        if "price-per-sqft-$" in df.columns and "total_sqft" in df.columns:
            df["total_price"] = (df["price-per-sqft-$"] * df["total_sqft"]).round(2)

    return df


def remove_upper_outliers_iqr(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df2 = df.copy()
    for col in cols:
        if col not in df2.columns:
            continue
        x = pd.to_numeric(df2[col], errors="coerce")
        x = x.dropna()
        if x.empty:
            continue
        q25, q75 = np.percentile(x, [25, 75])
        iqr = q75 - q25
        upper_lim = 1.5 * iqr + q75
        df2 = df2[pd.to_numeric(df2[col], errors="coerce") < upper_lim]
    return df2


def build_preprocessor(
    numeric_cols: List[str],
    categorical_cols: List[str],
    poly_degree: int = 2,
) -> ColumnTransformer:
    numeric_transform = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=poly_degree, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transform = OneHotEncoder(handle_unknown="ignore")

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_transform, numeric_cols),
            ("cat", categorical_transform, categorical_cols),
        ],
        remainder="drop",
    )
    return pre


def regression_models(df: pd.DataFrame) -> Dict[str, RegressionResults]:
    df = df.copy()

    if "availability" in df.columns:
        df = df.drop(columns=["availability"])

    y = df["total_price"].astype(float)

    X = df.drop(columns=["total_price"])

    numeric_cols = [c for c in ["total_sqft", "bath", "bedrooms", "balcony", "ber", "price-per-sqft-$"] if c in X.columns]
    categorical_cols = [c for c in ["property_scope", "location", "renovation_needed"] if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, poly_degree=2)

    results: Dict[str, RegressionResults] = {}

    lin = Pipeline(steps=[("pre", pre), ("model", LinearRegression())])
    lin.fit(X_train, y_train)
    pred = lin.predict(X_test)
    results["linear_regression"] = RegressionResults(
        mae=float(mean_absolute_error(y_test, pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        r2=float(r2_score(y_test, pred)),
    )

    rf = Pipeline(
        steps=[("pre", pre), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))]
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    results["random_forest_baseline"] = RegressionResults(
        mae=float(mean_absolute_error(y_test, pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        r2=float(r2_score(y_test, pred)),
    )

    param_dist = {
        "model__n_estimators": [300, 500],
        "model__max_depth": [10, 15],
        "model__max_features": ["sqrt"],
        "model__min_samples_split": [5, 10],
        "model__min_samples_leaf": [2],
    }

    tuner = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=5,
        cv=3,
        scoring="neg_mean_squared_error",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    tuner.fit(X_train, y_train)

    best = tuner.best_estimator_
    pred = best.predict(X_test)
    results["random_forest_tuned"] = RegressionResults(
        mae=float(mean_absolute_error(y_test, pred)),
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        r2=float(r2_score(y_test, pred)),
    )

    return results


def classification_models(df: pd.DataFrame) -> Dict[str, ClassificationResults]:
    df = df.copy()

    if "availability" in df.columns:
        df = df.drop(columns=["availability"])

    if "buying_or_not_buying" not in df.columns:
        raise ValueError("Missing column buying_or_not_buying")

    y_raw = df["buying_or_not_buying"].astype(str).str.lower().str.strip()
    y = y_raw.map({"no": 0, "yes": 1})

    X = df.drop(columns=["buying_or_not_buying"])

    numeric_cols = [c for c in ["total_sqft", "bath", "bedrooms", "balcony", "ber", "price-per-sqft-$"] if c in X.columns]
    categorical_cols = [c for c in ["property_scope", "location", "renovation_needed"] if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pre = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols, poly_degree=2)

    results: Dict[str, ClassificationResults] = {}

    logit = Pipeline(
        steps=[
            ("pre", pre),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    logit.fit(X_train, y_train)
    pred = logit.predict(X_test)
    results["logistic_regression_balanced"] = ClassificationResults(
        accuracy=float(accuracy_score(y_test, pred)),
        report=classification_report(y_test, pred, digits=3),
        confusion=confusion_matrix(y_test, pred),
    )

    rf_cls = Pipeline(
        steps=[
            ("pre", pre),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)),
        ]
    )
    rf_cls.fit(X_train, y_train)
    pred = rf_cls.predict(X_test)
    results["random_forest_classifier_balanced"] = ClassificationResults(
        accuracy=float(accuracy_score(y_test, pred)),
        report=classification_report(y_test, pred, digits=3),
        confusion=confusion_matrix(y_test, pred),
    )

    try:
        from xgboost import XGBClassifier

        pos = float((y_train == 0).sum()) / max(1.0, float((y_train == 1).sum()))
        xgb = Pipeline(
            steps=[
                ("pre", pre),
                ("model", XGBClassifier(
                    n_estimators=50,
                    max_depth=3,
                    subsample=0.8,
                    learning_rate=0.1,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=42,
                    n_jobs=-1,
                    scale_pos_weight=pos,
                    eval_metric="logloss",
                )),
            ]
        )
        xgb.fit(X_train, y_train)
        pred = xgb.predict(X_test)
        results["xgboost_classifier_weighted"] = ClassificationResults(
            accuracy=float(accuracy_score(y_test, pred)),
            report=classification_report(y_test, pred, digits=3),
            confusion=confusion_matrix(y_test, pred),
        )
    except Exception:
        pass

    return results


def main(csv_path: str) -> None:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(path)

    df = clean_df(df)

    df = remove_upper_outliers_iqr(df, cols=["bedrooms", "bath", "total_sqft", "total_price"])

    print("\nRows and columns after cleaning")
    print(df.shape)

    print("\nRegression results predicting total_price")
    reg = regression_models(df)
    for name, r in reg.items():
        print(f"\n{name}")
        print(f"MAE: {r.mae:,.2f}")
        print(f"RMSE: {r.rmse:,.2f}")
        print(f"R2: {r.r2:.3f}")

    print("\nClassification results predicting buying_or_not_buying")
    cls = classification_models(df)
    for name, r in cls.items():
        print(f"\n{name}")
        print(f"Accuracy: {r.accuracy:.3f}")
        print(r.report)
        print("Confusion matrix")
        print(r.confusion)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the raw Dublin housing CSV file",
    )
    args = parser.parse_args()
    main(args.csv)
