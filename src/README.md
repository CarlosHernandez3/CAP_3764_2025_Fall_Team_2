# Data Preprocessing Helper Module

This mini-package contains a reusable preprocessing module for a store-sales style
time series project (e.g. Rossmann). It is designed to live inside your project
as a helper module and be imported from notebooks or training scripts.


```bash
touch src/__init__.py
touch src/data_utils/__init__.py
```

## Installation / setup

Make sure your environment has the required Python packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

If you are not sure you have the packages, use `store_sale_prediction_env.yml`, located in the repo directory to install them

```bash
conda env update -n adv_ds -f store_sale_prediction_env.yml --prune
```

## How to import in a notebook

From a notebook whose working directory is the **project root**:

```python
from src.data_utils.preprocessing import (
    PreprocessingConfig,
    load_train,
    load_store,
    clean_train,
    clean_store,
    merge_train_store,
    add_calendar_features,
    add_lag_features,
    summarize_numeric,
    transform_by_mean_median,
    fourth_root_column,
    get_correlated_features,
    scale_features,
)
```


## End-to-end example

Below is a typical end-to-end usage for a store-sales forecasting project.

```python
from pathlib import Path
from src.data_utils.preprocessing import (
    PreprocessingConfig,
    load_train,
    load_store,
    clean_train,
    clean_store,
    merge_train_store,
    add_calendar_features,
    add_lag_features,
    summarize_numeric,
    transform_by_mean_median,
    fourth_root_column,
    get_correlated_features,
    scale_features,
)

# 1. Configuration
config = PreprocessingConfig(
    date_col="Date",
    store_col="Store",
    target_col="Sales",
    lags=(1, 7, 14, 28, 365),
    corr_threshold=0.15,
)

# 2. Load data
data_dir = Path("data")
train = load_train(data_dir / "train.csv")
store = load_store(data_dir / "store.csv")

# 3. Basic cleaning
train_clean = clean_train(train, config)
store_clean = clean_store(store)

# 4. Merge and add calendar features
df = merge_train_store(train_clean, store_clean, config)
df = add_calendar_features(df)

# 5. Add lagged features per store for Sales and Customers
df = add_lag_features(
    df,
    group_col=config.store_col,
    lagged_cols=["Sales", "Customers"],
    lags=config.lags,
)

# (Optional) Drop Customers if you only want to use lagged Customers
df = df.drop(columns=["Customers"])

# 6. Summarize numeric columns
float_cols = df.select_dtypes(include="float64").columns.tolist()
features_summ = summarize_numeric(df, float_cols)

# 7. Automatic log/sqrt transforms based on mean vs median
df = transform_by_mean_median(
    df,
    features_summ,
    exclude=[config.target_col],  # don't transform the raw target
)

# 8. 4th-root transformation for CompetitionDistance (if present)
if "sqrt_CompetitionDistance" in df.columns:
    df = fourth_root_column(
        df,
        col="sqrt_CompetitionDistance",
        new_name="CompetitionDistance_4th_root",
    )

# 9. Get numeric features highly correlated with transformed target
target_name = "sqrt_Sales"  # after transform_by_mean_median this is typical
high_corr_cols, corr_table = get_correlated_features(
    df,
    target_col=target_name,
    threshold=config.corr_threshold,
)

print("Numeric features with |corr| ≥ threshold:")
print(corr_table)

# 10. Build modeling dataframe
int_cols = df.select_dtypes(include="int64").columns.tolist()

modeling_cols = (
    [config.store_col]      # categorical store ID
    + int_cols              # integer / dummy variables
    + high_corr_cols        # selected numeric features
    + [target_name]         # transformed target
)

df_model = df[modeling_cols].dropna()

# 11. Scale numeric features and target for modeling
numeric_cols = df_model.select_dtypes(include="float64").columns.tolist()
df_scaled, x_scaler, y_scaler = scale_features(
    df_model,
    numeric_cols=numeric_cols,
    target_col=target_name,
)
```

## Optional: plotting helper

The module also provides a simple histogram grid helper for quick EDA:

```python
from src.data_utils.preprocessing import plot_hist_grid

float_cols = df.select_dtypes(include="float64").columns
plot_hist_grid(df, float_cols)
```

This will draw a grid of histograms with vertical lines for mean and median
for each feature.

## Adapting to your project

You can safely edit `PreprocessingConfig` and the helper functions to match your
project's column names and logic. The goal of this module is to group common
preprocessing steps in a clean, testable, and reusable way instead of having
all logic inside notebooks.
