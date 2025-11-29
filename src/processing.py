import pandas as pd
import numpy as np

def load_data(train_path):
    """
    Load training (and optional store) data.

    Parameters:
    train_path (str): Path to train.csv file.

    Returns:
    pd.DataFrame: Merged DataFrame if both paths are provided, else training data only.
    """
    train_df = pd.read_csv(train_path)
    return train_df


def clean_data(df):
    """
    Clean the dataset by:
    - Dropping closed days
    - Dropping rows with missing target values
    - Handling missing values and duplicates
    """
    df = df.copy()

    # Drop closed stores
    if 'Open' in df.columns:
        df = df[df['Open'] == 1]

    # Drop rows with missing Sales or Customers
    df = df.dropna(subset=['Sales', 'Customers'])

    # Remove duplicates
    df = df.drop_duplicates()

    return df


def process_date(df, date_col='Date'):
    """
    Convert date column to datetime and extract features Year, Month, Week, Day, Quarter, DayOfYear

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    date_col (str): Column name containing date values.

    Returns:
    pd.DataFrame: DataFrame with new date-based columns.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    df['Year'] = df[date_col].dt.year
    df['Month'] = df[date_col].dt.month
    df['Week'] = df[date_col].dt.isocalendar().week.astype(int)
    df['Day'] = df[date_col].dt.day
    df['Quarter'] = df[date_col].dt.quarter
    df['DayOfYear'] = df[date_col].dt.dayofyear

    df.drop(columns=[date_col], inplace=True)

    return df



def reclassify_columns(df):
    """
    Move appropriate numeric columns to categorical based on domain knowledge.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    tuple: (num_cols, cat_cols) lists of numerical and categorical column names.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    to_categorical = ['Store', 'DayOfWeek', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
    for col in to_categorical:
        if col in num_cols:
            num_cols.remove(col)
        if col not in cat_cols and col in df.columns:
            cat_cols.append(col)

    return num_cols, cat_cols


def encode_categoricals(df, cat_cols):
    """
    Perform one-hot encoding on categorical columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    cat_cols (list): List of categorical columns.

    Returns:
    pd.DataFrame: DataFrame with one-hot encoded columns.
    """
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def log_transform(df, cols):
    """
    Apply log1p transformation to reduce skewness for numeric columns.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    cols (list): Columns to log-transform.

    Returns:
    pd.DataFrame: DataFrame with transformed columns.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col])
    return df


def calculate_statistics(data):
    """
    Calculate basic statistics for numeric columns.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    stats = {}
    for column in data.select_dtypes(include=[np.number]).columns:
        stats[column] = {
            'mean': data[column].mean(),
            'median': data[column].median(),
            'std': data[column].std(),
            'min': data[column].min(),
            'max': data[column].max()
        }
    
    return stats


def full_preprocessing(train_path, log_cols=['Sales', 'Customers']):
    """
    Run the full preprocessing pipeline.

    Parameters:
    train_path (str): Path to training data.
    log_cols (list): Columns to apply log transformation.

    Returns:
    pd.DataFrame: Fully processed DataFrame ready for modeling.
    """
    df = load_data(train_path)
    df = clean_data(df)
    df = process_date(df)
    num_cols, cat_cols = reclassify_columns(df)
    df = log_transform(df, log_cols)
    df = encode_categoricals(df, cat_cols)
    return df