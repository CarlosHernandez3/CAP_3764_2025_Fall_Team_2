
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

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


# ============================================================
# Deterministic, TRAIN-derived transformations (HARD-CODED)
# ------------------------------------------------------------
# These must be set AFTER analyzing the TRAINING dataset.
# They MUST NOT depend on the test data.
# ============================================================

LOG_TRANSFORM_COLS: Tuple[str, ...] = ()
SQRT_TRANSFORM_COLS: Tuple[str, ...] = ()
FOURTH_ROOT_COLS: Tuple[str, ...] = ()


# ============================================================
# Loading
# ============================================================

def load_train(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))

def load_test(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))

def load_store(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


# ============================================================
# Cleaning (shared by train and test)
# ============================================================

def clean_sales_dataframe(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """
    Cleaning logic used for BOTH train and test.
    """
    out = df.copy()

    out[config.date_col] = pd.to_datetime(out[config.date_col], format="%Y-%m-%d")

    if "StateHoliday" in out.columns:
        out["StateHoliday"] = out["StateHoliday"].isin(["a", "b", "c"]).astype("int64")

    if config.target_col in out.columns:
        out[config.target_col] = out[config.target_col].astype("float64")

    if "Customers" in out.columns:
        out["Customers"] = out["Customers"].astype("float64")

    return out


def clean_store(store: pd.DataFrame) -> pd.DataFrame:
    """
    Same as before; no changes required.
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
            comp_month, prefix="Competition_open_since", drop_first=True, dtype="int64"
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

    drop_cols = [c for c in ["CompetitionOpenSinceMonth", "PromoInterval"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


# ============================================================
# Merging
# ============================================================

def merge_sales_store(
    sales_df: pd.DataFrame,
    store_df: pd.DataFrame,
    config: PreprocessingConfig,
) -> pd.DataFrame:

    df = sales_df.join(
        store_df.set_index(config.store_col),
        on=config.store_col,
        how="left",
    )

    df[config.store_col] = pd.Categorical(df[config.store_col])
    df = df.set_index(config.date_col).sort_index()
    return df


# ============================================================
# Calendar & Lag Features
# ============================================================

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    out = df.copy()
    out["Month"] = out.index.month
    out["Year"] = out.index.year
    out["Day"] = out.index.day
    return out


def add_lag_features(df: pd.DataFrame, group_col: str, lagged_cols: Sequence[str], lags: Sequence[int]):
    out = df.copy()
    grouped = out.groupby(group_col, observed=True)

    for lag in lags:
        for col in lagged_cols:
            if col in out.columns:
                out[f"{col}_lag{lag}"] = grouped[col].shift(lag)
    return out


# ============================================================
# Hardcoded Transformations (Train + Test)
# ============================================================

def apply_hardcoded_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic transformations based ONLY on rules defined
    from the TRAIN dataset.
    """
    out = df.copy()

    # log(x+1)
    for col in LOG_TRANSFORM_COLS:
        if col in out.columns:
            out[f"log_{col}"] = np.log(out[col].clip(lower=0) + 1)

    # sqrt(x)
    for col in SQRT_TRANSFORM_COLS:
        if col in out.columns:
            out[f"sqrt_{col}"] = np.sqrt(out[col].clip(lower=0))

    # fourth-root
    for col in FOURTH_ROOT_COLS:
        if col in out.columns:
            out[f"{col}_4th_root"] = np.sqrt(np.sqrt(out[col].clip(lower=0)))

    return out


# ============================================================
# Correlation (Train only)
# ============================================================

def get_correlated_features(df: pd.DataFrame, target_col: str, threshold: float):
    float_cols = df.select_dtypes(include="float64").columns
    if target_col not in float_cols:
        raise ValueError(f"{target_col} must be float64.")

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


# ============================================================
# Scaling: Train vs Test
# ============================================================

def scale_train(df: pd.DataFrame, numeric_cols: Sequence[str], target_col: Optional[str] = None):
    out = df.copy()

    x_scaler = StandardScaler()
    out[numeric_cols] = x_scaler.fit_transform(out[numeric_cols])

    y_scaler = None
    if target_col is not None and target_col in out.columns:
        y_scaler = StandardScaler()
        out[target_col] = y_scaler.fit_transform(out[[target_col]]).ravel()

    return out, x_scaler, y_scaler


def scale_test(
    df: pd.DataFrame,
    numeric_cols: Sequence[str],
    x_scaler: StandardScaler,
    target_col: Optional[str] = None,
    y_scaler: Optional[StandardScaler] = None,
):
    out = df.copy()
    out[numeric_cols] = x_scaler.transform(out[numeric_cols])

    if target_col and y_scaler and target_col in out.columns:
        out[target_col] = y_scaler.transform(out[[target_col]]).ravel()

    return out


# ============================================================
# Optional Plotting
# ============================================================

def plot_hist_grid(df: pd.DataFrame, cols: Sequence[str]):
    cols = list(cols)
    n = len(cols)
    if n == 0:
        return

    y = int(np.sqrt(n))
    x = int(np.ceil(n / y))

    fig, axes = plt.subplots(y, x, figsize=(16, 10), sharex=False, sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=30, alpha=0.7)
        ax.set_title(col)

    for j in range(n, x*y):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


__all__ = [
    "PreprocessingConfig",
    "load_train",
    "load_test",
    "load_store",
    "clean_sales_dataframe",
    "clean_store",
    "merge_sales_store",
    "add_calendar_features",
    "add_lag_features",
    "apply_hardcoded_transforms",
    "LOG_TRANSFORM_COLS",
    "SQRT_TRANSFORM_COLS",
    "FOURTH_ROOT_COLS",
    "get_correlated_features",
    "scale_train",
    "scale_test",
    "plot_hist_grid",
]
