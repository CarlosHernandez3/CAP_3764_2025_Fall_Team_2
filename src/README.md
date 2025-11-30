
# Preprocessing Module

This module provides a clean and consistent pipeline for preparing **train and test**
datasets for a store‑sales forecasting project. It ensures:

- identical transformations for train and test  
- no leakage from test into train  
- reproducible lag creation and feature engineering  
- deterministic transformations derived ONLY from training  

## Directory Structure

```
CAP_3764_2025_FALL_TEAM_2/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── store.csv
│
└── src/
    └── data_utils/
        └── preprocessing.py
```

## Key Design Principles

### ✔ Train and Test Are Preprocessed Separately  
You load and clean `train.csv` and `test.csv` independently.  
Both are merged with `store.csv`.

### ✔ Transformations for Test Are **Hardcoded**
Transforms like log/sqrt/fourth-root MUST be chosen during **training** and stored as:

```python
LOG_TRANSFORM_COLS = (...)
SQRT_TRANSFORM_COLS = (...)
FOURTH_ROOT_COLS = (...)
```

These are defined at the top of `preprocessing.py` and applied equally to **train and test**.

### ✔ Scaling Uses Train-Fitted Scalers ONLY  
- During training → `scale_train()` fits scalers  
- During test → `scale_test()` reuses those fitted scalers  

---

## Common Workflow

### 1. Preprocess Train

```python
from src.data_utils.preprocessing import *

config = PreprocessingConfig()

train = load_train("data/train.csv")
store = load_store("data/store.csv")

train = clean_sales_dataframe(train, config)
store = clean_store(store)

train = merge_sales_store(train, store, config)
train = add_calendar_features(train)

train = add_lag_features(train, "Store", ["Sales", "Customers"], config.lags)

# Apply hardcoded transforms decided from training EDA
train = apply_hardcoded_transforms(train)

# Select numeric columns and scale
numeric_cols = train.select_dtypes(include="float64").columns
train_scaled, x_scaler, y_scaler = scale_train(train, numeric_cols, target_col="sqrt_Sales")
```

Save:

```
x_scaler, y_scaler, numeric_cols, feature_list
```

---

## 2. Preprocess Test (NO refitting!)

```python
test = load_test("data/test.csv")

test = clean_sales_dataframe(test, config)
test = merge_sales_store(test, store, config)
test = add_calendar_features(test)
test = add_lag_features(test, "Store", ["Sales", "Customers"], config.lags)

# Use SAME transforms chosen in train
test = apply_hardcoded_transforms(test)

# Apply SAME numerical columns and scalers
test_scaled = scale_test(
    test,
    numeric_cols=numeric_cols,     # from training
    x_scaler=x_scaler,             # fitted in training
    target_col=None,               # no target in test
    y_scaler=None
)
```

---

## Notes

- This module **ensures reproducibility** and prevents test leakage.
- It is meant for **real ML workflows**, not only Kaggle-style notebooks.
- You can extend `preprocessing.py` with more domain features as needed.

