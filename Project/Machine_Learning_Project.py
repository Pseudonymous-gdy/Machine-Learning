from ucimlrepo import fetch_ucirepo

# fetch dataset
support2 = fetch_ucirepo(id=880)

# data (as pandas dataframes)
X = support2.data.features
y = support2.data.targets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.concat([X,y],axis=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OrdinalEncoder
# OrdinalEncoder is used for encoding categorical features

missing_value_table = data.isnull().sum()
missing_value_proportion = missing_value_table[missing_value_table>0].sort_values(ascending=False) / len(data)

def fill_missing_values(data: pd.DataFrame, missing_col: str) -> pd.DataFrame:
    """
    Fill missing values in a specified column of a DataFrame.
    For categorical columns (dtype 'object'), use DecisionTreeClassifier.
    For numerical columns, use Polynomial Regression (degree = 2).

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing missing values.
    missing_col : str
        The name of the column with missing values to be filled.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with missing values in the specified column filled.
    """
    # Create a boolean mask to identify rows with missing values in the target column
    missing_mask = data[missing_col].isnull()

    # If there are no missing values, return the original data
    if not missing_mask.any():
        return data

    # Separate the feature matrix (X) and the target column (y)
    # X_full: DataFrame with the missing column dropped
    # y_full: The target column with missing values
    X_full = data.drop(missing_col, axis = 1)
    y_full = data[missing_col]

    # Split the data into training set (non - missing values) and prediction set (missing values)
    # X_train: Features for training (rows without missing values in the target column)
    # y_train: Target values for training (non - missing values in the target column)
    # X_missing: Features for prediction (rows with missing values in the target column)
    X_train = X_full[~missing_mask]
    y_train = y_full[~missing_mask]
    X_missing = X_full[missing_mask]

    # Handle categorical columns (dtype 'object')
    if data[missing_col].dtype == 'object' or 'O':
        # Use OrdinalEncoder to encode categorical features.
        # handle_unknown='use_encoded_value' and unknown_value=-1: Deal with unknown categories in the prediction set
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

        # Encode the training features
        X_train_enc = enc.fit_transform(X_train)
        # Encode the prediction features if there are missing values to predict
        X_missing_enc = enc.transform(X_missing) if not X_missing.empty else None

        # Encode the training target (convert categorical target to numerical)
        y_train_enc = enc.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Create and train a DecisionTreeClassifier
        model = DecisionTreeClassifier(random_state = 42)
        model.fit(X_train_enc, y_train_enc)

        # Predict the missing values if there are rows to predict
        if not X_missing.empty:
            y_pred_enc = model.predict(X_missing_enc)
            # Decode the predicted values back to the original categorical values
            y_pred = enc.inverse_transform(y_pred_enc.reshape(-1, 1)).ravel()
            # Fill the missing values in the original DataFrame
            data.loc[missing_mask, missing_col] = y_pred

    # Handle numerical columns
    else:
        # Create polynomial features (degree = 2)
        poly = PolynomialFeatures(degree = 2)
        # Transform the training features to polynomial features
        X_train_poly = poly.fit_transform(X_train)

        # Create and train a LinearRegression model
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # Predict the missing values if there are rows to predict
        if not X_missing.empty:
            # Transform the prediction features to polynomial features
            X_missing_poly = poly.transform(X_missing)
            y_pred = model.predict(X_missing_poly)
            # Fill the missing values in the original DataFrame
            data.loc[missing_mask, missing_col] = y_pred

    return data[missing_col]


for i in missing_value_proportion.index:
  data[i] = fill_missing_values(data, i)

def outlier_detection(data):
    """
    Detect outliers for numerical and categorical features.
    Returns a DataFrame with outlier flags (1=outlier, 0=normal) for each method.
    """
    results = data.copy()
    outlier_flags = pd.DataFrame(index=results.index)

    # Numerical: IQR method (threshold=3, typical for moderate outlier frequency)
    num_cols = results.select_dtypes(include=["number"]).columns
    def iqr_detector(col, threshold=3):
        q1 = col.quantile(0.25)
        q3 = col.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return ((col < lower_bound) | (col > upper_bound)).astype(int)
    for col in num_cols:
        outlier_flags[f'iqr_{col}'] = iqr_detector(results[col])

    # Categorical: rare category (threshold=0.0005, i.e., <.1% frequency)
    cat_cols = results.select_dtypes(include=["object", "category"]).columns
    def category_outlier_detector(col, threshold=0.0005):
        freq = col.value_counts(normalize=True)
        rare_categories = freq[freq < threshold].index
        return col.isin(rare_categories).astype(int)
    for col in cat_cols:
        outlier_flags[f'cat_outlier_{col}'] = category_outlier_detector(results[col])

    return outlier_flags

def remove_outliers(data):
    """
    Remove outliers from the DataFrame.
    Returns a DataFrame with outliers removed.
    """
    crit = outlier_detection(data)
    for row in crit.index:
        if crit.loc[row].sum() > 5:
            data = data.drop(row)
    return data

data_no_outliers = remove_outliers(data.copy())
data_no_outliers.reset_index(drop=True, inplace=True)
print(data_no_outliers)
def encoding(data, max_categories=50):
    '''
    Encode categorical features using:
    - Binary indicator (1/0) for dichotomous variables
    - One-hot encoding for variables with 3-10 categories
    - Ordinal encoding for variables with more than 10 categories
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame
    max_categories : int
        Maximum number of categories for one-hot encoding
    '''
    results = data.copy()
    cat_cols = results.select_dtypes(include=["object", "category"]).columns.tolist()
    
    for col in cat_cols:
        n_unique = results[col].nunique(dropna=False)
        
        if n_unique == 2:
            # Binary indicator: map the two categories to 1 and 0
            categories = results[col].dropna().unique()
            mapping = {categories[0]: 1, categories[1]: 0}
            results[f'binary_{col}'] = results[col].map(mapping)
            results = results.drop(col, axis=1)
            
        elif 2 < n_unique <= max_categories:
            # One-hot encode columns with moderate number of categories
            dummies = pd.get_dummies(results[col], prefix=col, drop_first=True, dtype=float)
            results = pd.concat([results.drop(col, axis=1), dummies], axis=1)
            
        else:
            # Ordinal encode columns with too many categories
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            results[f'ordinal_{col}'] = enc.fit_transform(results[[col]])
            results = results.drop(col, axis=1)
    
    return results

data_encoded = encoding(data_no_outliers)
print(data_encoded)