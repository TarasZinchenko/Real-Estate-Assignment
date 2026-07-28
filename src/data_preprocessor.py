"""
Data Preprocessing Module for Real Estate Price Prediction
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def preprocess_real_estate_data(df: pd.DataFrame, target_column: str = 'price'):
    """
    Cleans, imputes, and encodes real estate features.
    
    Parameters:
        df (pd.DataFrame): Raw housing dataframe.
        target_column (str): Target column to predict (e.g., 'price').
        
    Returns:
        X_processed (pd.DataFrame): Transformed feature matrix.
        y (pd.Series or None): Target variable if specified.
    """
    # Drop irrelevant or redundant text columns
    columns_to_drop = ['full_address', 'street', 'sold_date', 'status']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        print(f"Dropped non-predictive columns: {columns_to_drop}")

    feature_columns = [col for col in df.columns if col != target_column] if target_column else df.columns.tolist()

    numerical_features = []
    categorical_features = []

    for col in feature_columns:
        if df[col].dtype in ['int64', 'float64']:
            if col in ['bed', 'bath', 'acre_lot', 'house_size']:
                numerical_features.append(col)
            elif col == 'zip_code':
                df[col] = df[col].astype(str)
                categorical_features.append(col)
            elif col != target_column:
                numerical_features.append(col)
        else:
            categorical_features.append(col)
            df[col] = df[col].astype(str)

    y = None
    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df[feature_columns].copy()
        print(f"Target column '{target_column}' separated successfully.")
    elif target_column:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    else:
        X = df.copy()

    # Numerical pipeline: Median imputation + Standard Scaling
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute with 'missing' + One-Hot Encoding
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    final_numerical_features = [f for f in numerical_features if f in X.columns]
    final_categorical_features = [f for f in categorical_features if f in X.columns]

    transformers_list = []
    if final_numerical_features:
        transformers_list.append(('num', numerical_pipeline, final_numerical_features))
    if final_categorical_features:
        transformers_list.append(('cat', categorical_pipeline, final_categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers_list, remainder='passthrough')

    X_processed_np = preprocessor.fit_transform(X)

    # Reconstruct feature names
    processed_feature_names = []
    if 'num' in preprocessor.named_transformers_ and final_numerical_features:
        processed_feature_names.extend(final_numerical_features)
    
    if 'cat' in preprocessor.named_transformers_ and final_categorical_features:
        try:
            onehot_cols = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(final_categorical_features)
            processed_feature_names.extend(onehot_cols)
        except Exception:
            pass

    X_processed = pd.DataFrame(X_processed_np, columns=processed_feature_names, index=X.index)
    print(f"Preprocessing completed. Shape: {X_processed.shape}")

    if target_column:
        return X_processed, y
    return X_processed


if __name__ == '__main__':
    # Determine base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file_path = os.path.join(base_dir, 'data', '2 realtor-data.csv')
    
    if not os.path.exists(data_file_path):
        data_file_path = os.path.join(base_dir, '2 realtor-data.csv')

    print(f"Loading raw dataset from: {data_file_path}")
    try:
        df_loaded = pd.read_csv(data_file_path)
        print(f"Dataset loaded. Total rows: {len(df_loaded)}")

        X_processed_price, y_price = preprocess_real_estate_data(df_loaded, target_column='price')

        processed_df_price = X_processed_price.copy()
        processed_df_price['price'] = y_price.values

        output_dir = os.path.join(base_dir, 'data')
        os.makedirs(output_dir, exist_ok=True)
        output_price_path = os.path.join(output_dir, "prep-realtor-data_price.csv")
        
        processed_df_price.to_csv(output_price_path, index=False)
        print(f"Preprocessed data saved to: {output_price_path}")

    except FileNotFoundError:
        print(f"Error: Dataset not found at '{data_file_path}'. Please check file path.")
    except Exception as e:
        print(f"Error during preprocessing: {e}")
