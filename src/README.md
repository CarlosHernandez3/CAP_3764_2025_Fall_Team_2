# Preprocessing Module

This module provides a repeatable pipeline for preparing store-level sales data for modeling. It keeps train and test processing aligned, avoids leakage, and makes the train-time choices (transforms, scalers, feature list) explicit and reusable.

## Layout

- data/train.csv, data/test.csv: sales records (Date, Store, Sales, Customers, etc.)
- data/store.csv: store metadata used for joins and feature expansion
- src/data_utils/preprocessing.py: preprocessing functions and defaults

## Key Ideas

- Train and test are cleaned separately but use the same rules.
- Transform choices (log/sqrt/fourth-root) are hardcoded tuples at the top of `preprocessing.py` and must be decided using training data only.
- Scaling is fit on train once via `scale_train`; reuse the same scalers with `scale_test`.
- Store and date handling is standardized through `PreprocessingConfig` (store column, date column, target name, lags, correlation threshold).

## Setup (conda)

Update the existing environment named `adv_ds` (creates it if missing) from `store_sale_prediction_env.yml`:

```
conda env update -n adv_ds -f store_sale_prediction_env.yml --prune
```

Then activate it:

```
conda activate adv_ds
```

## Manual workflow (train -> test)

```python
from src.data_utils.preprocessing import (
    PreprocessingConfig,
    load_train,
    load_store,
    clean_sales_dataframe,
    clean_store,
    merge_sales_store,
    add_calendar_features,
    add_lag_features,
    apply_hardcoded_transforms,
    scale_train,
    scale_test,
)

config = PreprocessingConfig()

# Train prep
train = load_train("data/train.csv")
store = load_store("data/store.csv")
train = clean_sales_dataframe(train, config)
store = clean_store(store)
train = merge_sales_store(train, store, config)
train = add_calendar_features(train)
train = add_lag_features(train, group_col=config.store_col, lagged_cols=[config.target_col, "Customers"], lags=config.lags)
train = apply_hardcoded_transforms(train)

# Choose numeric columns to scale (exclude the target if present)
numeric_cols = [c for c in train.select_dtypes(include=["float64", "float32"]).columns if c != config.target_col]
train_scaled, x_scaler, y_scaler = scale_train(train, numeric_cols=numeric_cols, target_col=config.target_col)

# Save these for inference
# numeric_cols, x_scaler, y_scaler

# Test/holdout prep using the same decisions
# test = load_train("data/test.csv")  # use load_train for any CSV with the same schema
# test = clean_sales_dataframe(test, config)
# test = merge_sales_store(test, store, config)
# test = add_calendar_features(test)
# test = add_lag_features(test, group_col=config.store_col, lagged_cols=[config.target_col, "Customers"], lags=config.lags)
# test = apply_hardcoded_transforms(test)
# test_scaled = scale_test(test, numeric_cols=numeric_cols, x_scaler=x_scaler, target_col=None, y_scaler=None)
```

## Quick split helper

`prepare_train_test_split(train_path, store_path, config, lagged_cols=None, numeric_cols=None)`

- Loads train and store, cleans, merges, adds calendar/lag features, applies hardcoded transforms.
- Splits by date: latest 25% of unique dates become the holdout set.
- Scales both subsets; returns `(train_scaled, test_scaled, x_scaler, y_scaler)`.
- Override `lagged_cols` or `numeric_cols` to control which columns are lagged or scaled.

### Example: single call to get train, test, and scalers

```python
from pathlib import Path
from joblib import dump, load
from src.data_utils.preprocessing import PreprocessingConfig, prepare_train_test_split

config = PreprocessingConfig()

# Build ready-to-model sets plus scalers in one step
train_scaled, test_scaled, x_scaler, y_scaler = prepare_train_test_split(
    train_path=Path("data/train.csv"),
    store_path=Path("data/store.csv"),
    config=config,
    # optional overrides:
    # lagged_cols=["Sales", "Customers"],
    # numeric_cols=["sqrt_Sales", "sqrt_Customers", ...],
)

# Persist the scalers for reuse in inference/production
dump(x_scaler, "artifacts/x_scaler.joblib")
if y_scaler is not None:
    dump(y_scaler, "artifacts/y_scaler.joblib")

# Later (e.g., in an inference script) load and reuse them
x_scaler = load("artifacts/x_scaler.joblib")
y_scaler = load("artifacts/y_scaler.joblib")
```

## Hardcoded transforms

Edit `LOG_TRANSFORM_COLS`, `SQRT_TRANSFORM_COLS`, and `FOURTH_ROOT_COLS` in `preprocessing.py` after exploring the training data. These tuples drive `apply_hardcoded_transforms` for both train and test, ensuring the same feature math everywhere.

## Extra utilities

- `get_correlated_features(df, target_col, threshold)`: pick features above an absolute correlation cutoff with the target.
- `plot_hist_grid(df, cols)`: quick histogram grid for exploratory checks.

## Good practices

- Keep `PreprocessingConfig` consistent between training and inference.
- Persist `numeric_cols`, `x_scaler`, `y_scaler`, and any feature list your model expects.
- Never refit scalers on test or production data; reuse the training-fitted objects.
