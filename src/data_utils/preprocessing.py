from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


@dataclass
class PreprocessingConfig:
    """
    Configuration for preprocessing store sales data.
    """
    date_col: str = "Date"
    store_col: str = "Store"
    target_col: str = "Sales"
    lags: Sequence[int] = (1, 7, 14, 28, 365)
    corr_threshold: float = 0.15


def load_train(path: str | Path) -> pd.DataFrame:
    """Load the training dataset from CSV."""
    return pd.read_csv(Path(path))


def load_store(path: str | Path) -> pd.DataFrame:
    """Load the store metadata dataset from CSV."""
    return pd.read_csv(Path(path))


def clean_train(train: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """
    Basic cleaning of the train dataframe:
    - Parse dates
    - Convert StateHoliday to binary
    - Ensure numeric types for Sales and Customers
    """
    df = train.copy()

    df[config.date_col] = pd.to_datetime(df[config.date_col], format="%Y-%m-%d")

    if "StateHoliday" in df.columns:
        df["StateHoliday"] = df["StateHoliday"].isin(["a", "b", "c"]).astype("int64")

    if config.target_col in df.columns:
        df[config.target_col] = df[config.target_col].astype("float64")

    if "Customers" in df.columns:
        df["Customers"] = df["Customers"].astype("float64")

    return df


def clean_store(store: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning and feature engineering for the store dataframe:
    - Fill/convert Promo2SinceWeek, Promo2SinceYear, CompetitionOpenSinceYear
    - Create dummies for CompetitionOpenSinceMonth (month name)
    - Create dummies for PromoInterval (comma-separated string)
    """
    df = store.copy()

    for col in ["Promo2SinceWeek", "Promo2SinceYear", "CompetitionOpenSinceYear"]:
        if col in df.columns:
            df[col] = df[col].fillna(-1).astype("int64")

    if "CompetitionOpenSinceMonth" in df.columns:
        comp_month = pd.to_datetime(
            df["CompetitionOpenSinceMonth"], format="%m", errors="coerce"
        ).dt.month_name()

        comp_dummies = pd.get_dummies(
            comp_month,
            prefix="Competition_open_since",
            drop_first=True,
            dtype="int64",
        )
        df = df.join(comp_dummies)

    if "PromoInterval" in df.columns:
        dummies = (
            df["PromoInterval"]
            .fillna("")
            .str.get_dummies(sep=",")
            .add_prefix("Promo2_Interval_in_")
        )
        if dummies.shape[1] > 0:
            dummies = dummies.iloc[:, 1:]
        df = df.join(dummies)

    drop_cols: List[str] = []
    for col in ["CompetitionOpenSinceMonth", "PromoInterval"]:
        if col in df.columns:
            drop_cols.append(col)

    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def merge_train_store(
    train: pd.DataFrame,
    store: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """
    Merge cleaned train and store dataframes on the store column.
    Sets the date column as the index and sorts by date.
    """
    df = train.join(
        store.set_index(config.store_col),
        on=config.store_col,
        how="left",
    )

    df[config.store_col] = pd.Categorical(df[config.store_col])
    df = df.set_index(config.date_col).sort_index()

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features (Year, Month, Day) from the DatetimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex to add calendar features.")

    out = df.copy()
    out["Month"] = out.index.month
    out["Year"] = out.index.year
    out["Day"] = out.index.day
    return out


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    lagged_cols: Sequence[str],
    lags: Sequence[int],
) -> pd.DataFrame:
    """
    Add lagged features for the given numeric columns, grouped by `group_col`.
    """
    out = df.copy()
    grouped = out.groupby(group_col, observed=True)

    for lag in lags:
        for col in lagged_cols:
            if col not in out.columns:
                continue
            out[f"{col}_lag{lag}"] = grouped[col].shift(lag)

    return out


def summarize_numeric(
    df: pd.DataFrame,
    cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Compute mean and median for each numeric feature.

    Returns a dataframe with columns: ["Feature", "mean_val", "median_val"].
    """
    if cols is None:
        cols = df.select_dtypes(include="float64").columns.tolist()

    means = df[cols].mean()
    medians = df[cols].median()

    features_summ = pd.DataFrame(
        {
            "Feature": cols,
            "mean_val": means.values,
            "median_val": medians.values,
        }
    )
    return features_summ


def transform_by_mean_median(
    df: pd.DataFrame,
    features_summ: pd.DataFrame,
    exclude: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Apply log or sqrt transforms based on mean/median comparison:

    - If mean < median: x -> log(x + 1)
    - If mean > median: x -> sqrt(x)
    - Else: no transform
    """
    out = df.copy()
    exclude = set(exclude or [])

    for _, row in features_summ.iterrows():
        feature = row["Feature"]
        if feature in exclude:
            continue
        if feature not in out.columns:
            continue

        mean_val = row["mean_val"]
        median_val = row["median_val"]

        if mean_val < median_val:
            out[feature] = np.log(out[feature] + 1.0)
            out = out.rename(columns={feature: f"log_{feature}"})
        elif mean_val > median_val:
            out[feature] = np.sqrt(out[feature])
            out = out.rename(columns={feature: f"sqrt_{feature}"})

    return out


def fourth_root_column(df: pd.DataFrame, col: str, new_name: Optional[str] = None) -> pd.DataFrame:
    """
    Apply a 4th-root transform using sqrt(sqrt(x)).
    """
    if new_name is None:
        new_name = f"{col}_4th_root"

    out = df.copy()
    values = out[col]
    out[new_name] = np.sqrt(np.sqrt(values.clip(lower=0)))
    return out


def get_correlated_features(
    df: pd.DataFrame,
    target_col: str,
    threshold: float,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Get numeric features whose absolute correlation with `target_col`
    is above the given threshold.

    Returns (high_corr_cols, corr_table).
    """
    float_cols = df.select_dtypes(include="float64").columns
    if target_col not in float_cols:
        raise ValueError(f"{target_col!r} must be a float64 column to compute correlations.")

    corr = df[float_cols].corr()
    target_corr = corr[target_col].drop(target_col)

    mask = target_corr.abs() >= threshold
    high_corr_cols = target_corr.index[mask].tolist()

    corr_table = (
        target_corr[mask]
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature", target_col: "Correlation"})
    )

    return high_corr_cols, corr_table


def scale_features(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    target_col: str,
) -> Tuple[pd.DataFrame, StandardScaler, StandardScaler]:
    """
    Standardize numeric features and target using sklearn's StandardScaler.
    """
    df_scaled = df.copy()

    x_scaler = StandardScaler()
    df_scaled[numeric_cols] = x_scaler.fit_transform(df_scaled[numeric_cols])

    y_scaler = StandardScaler()
    df_scaled[target_col] = y_scaler.fit_transform(df_scaled[[target_col]]).ravel()

    return df_scaled, x_scaler, y_scaler


def plot_hist_grid(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> None:
    """
    Plot a grid of histograms for the given columns.
    """
    cols = list(cols)
    n = len(cols)
    if n == 0:
        return

    y = int(np.sqrt(n))
    x = int(np.ceil(n / y))

    fig, axes = plt.subplots(
        nrows=y,
        ncols=x,
        figsize=(16, 10),
        sharex=False,
        sharey=False,
    )

    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()

        mean_val = data.mean()
        median_val = data.median()

        ax.hist(data, bins=30, alpha=0.7)
        ax.axvline(mean_val, linestyle="--", linewidth=1.5, label="Mean")
        ax.axvline(median_val, linestyle=":", linewidth=1.5, label="Median")
        ax.set_title(f"Histogram of {col}")
        ax.legend()

    for j in range(n, x * y):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


__all__ = [
    "PreprocessingConfig",
    "load_train",
    "load_store",
    "clean_train",
    "clean_store",
    "merge_train_store",
    "add_calendar_features",
    "add_lag_features",
    "summarize_numeric",
    "transform_by_mean_median",
    "fourth_root_column",
    "get_correlated_features",
    "scale_features",
    "plot_hist_grid",
]
