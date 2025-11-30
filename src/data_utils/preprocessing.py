
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
    Configuration container for common column names and preprocessing defaults.

    Attributes
    ----------
    date_col : str
        Name of the timestamp column used as the index.
    store_col : str
        Identifier for the store; also used for grouping and merging.
    target_col : str
        Name of the sales target column.
    lags : Sequence[int]
        Default lag steps to create for lagged features.
    corr_threshold : float
        Minimum absolute correlation required to keep a feature.
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
SQRT_TRANSFORM_COLS: Tuple[str, ...] = (
                                            'Sales' ,  'Sales_lag1' ,
                                            'Customers_lag1', 'Sales_lag7', 'Customers_lag7',
                                            'Sales_lag14', 'Customers_lag14', 'Sales_lag28',
                                            'Customers_lag28', 'Sales_lag365', 'Customers_lag365'
                                             )
FOURTH_ROOT_COLS: Tuple[str, ...] = ('CompetitionDistance')


# ============================================================
# Loading
# ============================================================

def load_train(path: str | Path) -> pd.DataFrame:
    """
    Load the training split from a CSV file.

    Parameters
    ----------
    path : str | Path
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Raw training data.
    """
    return pd.read_csv(Path(path))



def load_store(path: str | Path) -> pd.DataFrame:
    """
    Load store metadata from a CSV file.

    Parameters
    ----------
    path : str | Path
        Location of the CSV file.

    Returns
    -------
    pd.DataFrame
        Store metadata keyed by store identifier.
    """
    return pd.read_csv(Path(path))


# ============================================================
# Cleaning (shared by train and test)
# ============================================================

def clean_sales_dataframe(df: pd.DataFrame, config: PreprocessingConfig) -> pd.DataFrame:
    """
    Normalize schema and basic types for sales data used in train and test.

    - Parses the date column to datetime.
    - Converts `StateHoliday` to a binary indicator.
    - Casts sales and customer counts to floats for downstream math.

    Parameters
    ----------
    df : pd.DataFrame
        Input sales data.
    config : PreprocessingConfig
        Column configuration.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of the input frame.
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
    Standardize store metadata to numeric and dummy-encoded columns.

    Fills missing numeric fields with -1, expands month/year and promo interval
    strings into categorical dummies, and drops unused raw columns.

    Parameters
    ----------
    store : pd.DataFrame
        Raw store metadata.

    Returns
    -------
    pd.DataFrame
        Cleaned store metadata ready for joining to sales data.
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
    """
    Join sales records with store metadata and sort by date.

    Parameters
    ----------
    sales_df : pd.DataFrame
        Cleaned sales records.
    store_df : pd.DataFrame
        Cleaned store metadata.
    config : PreprocessingConfig
        Column configuration with store and date column names.

    Returns
    -------
    pd.DataFrame
        Merged frame indexed by date with store as a categorical column.
    """

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
    """
    Add calendar features (month, year, day) from a DatetimeIndex.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by datetime.

    Returns
    -------
    pd.DataFrame
        Copy of the frame with `Month`, `Year`, and `Day` columns added.

    Raises
    ------
    ValueError
        If the index is not a `pd.DatetimeIndex`.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    out = df.copy()
    out["Month"] = out.index.month
    out["Year"] = out.index.year
    out["Day"] = out.index.day
    return out


def add_lag_features(df: pd.DataFrame, group_col: str, lagged_cols: Sequence[str], lags: Sequence[int]):
    """
    Create lagged versions of numeric columns within grouped time series.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with a grouping column.
    group_col : str
        Column used to group the series (e.g., store).
    lagged_cols : Sequence[str]
        Columns to lag.
    lags : Sequence[int]
        Lag offsets to apply to each column.

    Returns
    -------
    pd.DataFrame
        Copy of the frame containing additional lag columns.
    """
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
    Apply deterministic transforms (log, sqrt, fourth-root) defined on train data.

    The columns to transform are controlled by the module-level tuples
    `LOG_TRANSFORM_COLS`, `SQRT_TRANSFORM_COLS`, and `FOURTH_ROOT_COLS`.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with numeric columns to transform.

    Returns
    -------
    pd.DataFrame
        Copy of the frame containing the new transformed feature columns.
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
    """
    Compute features with absolute correlation above a threshold to the target.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing numeric features and the target.
    target_col : str
        Name of the target column; must be float64.
    threshold : float
        Minimum absolute correlation value to select.

    Returns
    -------
    Tuple[List[str], pd.DataFrame]
        List of feature names meeting the threshold and a summary correlation table.

    Raises
    ------
    ValueError
        If `target_col` is not present as float64.
    """
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
    """
    Fit scalers on training data and transform numeric features (and optionally target).

    Parameters
    ----------
    df : pd.DataFrame
        Training data.
    numeric_cols : Sequence[str]
        Columns to standardize.
    target_col : Optional[str], default None
        Target column to scale; when provided a separate scaler is fit.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler, Optional[StandardScaler]]
        Scaled training data, feature scaler, and optional target scaler.
    """
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
    """
    Apply pre-fit scalers to a test/validation dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Data to transform.
    numeric_cols : Sequence[str]
        Numeric columns expected by the feature scaler.
    x_scaler : StandardScaler
        Scaler fitted on training features.
    target_col : Optional[str], default None
        Target column to scale if present.
    y_scaler : Optional[StandardScaler], default None
        Target scaler fitted on the training set.

    Returns
    -------
    pd.DataFrame
        Scaled copy of the input.
    """
    out = df.copy()
    out[numeric_cols] = x_scaler.transform(out[numeric_cols])

    if target_col and y_scaler and target_col in out.columns:
        out[target_col] = y_scaler.transform(out[[target_col]]).ravel()

    return out


# ============================================================
# Optional Plotting
# ============================================================

def plot_hist_grid(df: pd.DataFrame, cols: Sequence[str]):
    """
    Plot a grid of histograms for selected columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    cols : Sequence[str]
        Column names to plot; does nothing if empty.

    Returns
    -------
    None
    """
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


def prepare_train_test_split(
    train_path: str | Path,
    store_path: str | Path,
    config: PreprocessingConfig,
    lagged_cols: Optional[Sequence[str]] = None,
    numeric_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Optional[StandardScaler]]:
    """
    Load train/store data, merge, split by latest dates, transform, and scale.

    The most recent 25% of dates become the test set. Lag features and
    hardcoded transforms are computed on the full merged frame before the split
    so the test set can use history from the training period.

    Parameters
    ----------
    train_path : str | Path
        Path to ``train.csv`` (includes sales/target).
    store_path : str | Path
        Path to ``store.csv`` metadata.
    config : PreprocessingConfig
        Column configuration and default lags.
    lagged_cols : Optional[Sequence[str]], default None
        Columns to lag per store; defaults to target plus ``Customers`` if
        present.
    numeric_cols : Optional[Sequence[str]], default None
        Feature columns to scale. If None, all float columns except the target
        are used.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, StandardScaler, Optional[StandardScaler]]
        Scaled training set, scaled holdout set, fitted feature scaler, and
        optional target scaler.
    """
    # Load and clean
    train_raw = load_train(train_path)
    store_raw = load_store(store_path)

    sales_df = clean_sales_dataframe(train_raw, config)
    store_df = clean_store(store_raw)

    merged = merge_sales_store(sales_df, store_df, config)
    merged = add_calendar_features(merged)

    # Decide lagged columns
    default_lagged = [config.target_col]
    if "Customers" in merged.columns:
        default_lagged.append("Customers")
    lagged_cols = list(lagged_cols) if lagged_cols is not None else default_lagged
    merged = add_lag_features(
        merged, group_col=config.store_col, lagged_cols=lagged_cols, lags=config.lags
    )

    merged = apply_hardcoded_transforms(merged)

    # Split by most recent 25% of dates
    unique_dates = merged.index.unique().sort_values()
    if len(unique_dates) < 2:
        raise ValueError("Not enough unique dates to perform a train/test split.")

    split_idx = int(np.floor(0.75 * len(unique_dates)))
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    threshold_date = unique_dates[split_idx]

    train_df = merged.loc[merged.index < threshold_date]
    test_df = merged.loc[merged.index >= threshold_date]

    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split resulted in an empty subset.")

    # Choose numeric columns to scale
    target_col = config.target_col if config.target_col in train_df.columns else None
    if numeric_cols is None:
        numeric_cols = [
            col
            for col in train_df.select_dtypes(include=["float64", "float32"]).columns
            if col != target_col
        ]

    train_scaled, x_scaler, y_scaler = scale_train(
        train_df, numeric_cols=numeric_cols, target_col=target_col
    )
    test_scaled = scale_test(
        test_df,
        numeric_cols=numeric_cols,
        x_scaler=x_scaler,
        target_col=target_col,
        y_scaler=y_scaler,
    )

    return train_scaled, test_scaled, x_scaler, y_scaler


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
    "prepare_train_test_split",
]
