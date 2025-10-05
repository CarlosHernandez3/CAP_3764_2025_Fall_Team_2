"""
scripts package

This package contains reusable modules for data preprocessing, cleaning,
feature engineering, and exploratory data analysis.

Modules:
--------
preprocessing : Includes functions for data loading, cleaning, 
                feature extraction, encoding, and transformation.
"""

from .processing import (
    load_data,
    clean_data,
    process_date,
    reclassify_columns,
    encode_categoricals,
    log_transform,
    calculate_statistics,
    full_preprocessing,
)
