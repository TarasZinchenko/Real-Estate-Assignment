"""
Machine Learning Model Training & Evaluation Script for Real Estate Price Prediction
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_preprocessor import preprocess_real_estate_data

TARGET_VARIABLE_NAME = 'price'


def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_data_path = os.path.join(base_dir, 'data', 'prep-realtor-data_price.csv')
    raw_data_path = os.path.join(base_dir, 'data', '2 realtor-data.csv')

    df = None
    if os.path.exists(prep_data_path):
        print(f"Loading preprocessed dataset from: {prep_data_path}")
        df = pd.read_csv(prep_data_path)
    elif os.path.exists(raw_data_path):
        print(f"Preprocessed data not found. Processing raw dataset from: {raw_data_path}")
        raw_df = pd.read_csv(raw_data_path)
        X_proc, y_proc = preprocess_real_estate_data(raw_df, target_column=TARGET_VARIABLE_NAME)
        df = X_proc.copy()
        df[TARGET_VARIABLE_NAME] = y_proc.values
    else:
        # Check root directory fallbacks
        root_raw_path = os.path.join(base_dir, '2 realtor-data.csv')
        if os.path.exists(root_raw_path):
            print(f"Processing raw dataset from root directory: {root_raw_path}")
            raw_df = pd.read_csv(root_raw_path)
            X_proc, y_proc = preprocess_real_estate_data(raw_df, target_column=TARGET_VARIABLE_NAME)
            df = X_proc.copy()
            df[TARGET_VARIABLE_NAME] = y_proc.values

    if df is None:
        raise FileNotFoundError("Could not find dataset. Please place '2 realtor-data.csv' in the data/ directory.")

    if TARGET_VARIABLE_NAME not in df.columns:
        raise ValueError(f"Target column '{TARGET_VARIABLE_NAME}' not found in dataset.")

    X = df.drop(columns=[TARGET_VARIABLE_NAME])
    y = df[TARGET_VARIABLE_NAME]

    print(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train split: {X_train.shape}, Test split: {X_test.shape}")

    # Model initialization & training
    print("\nTraining Random Forest Regressor (n_jobs=-1)...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Prediction & evaluation
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Model Evaluation Results ---")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R2 Score):          {r2:.4f}")

    # Output directory for plots
    output_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Actual vs. Predicted Prices
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=2, label='Ideal Fit Line')
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Actual vs. Predicted Real Estate Prices")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    actual_vs_pred_path = os.path.join(output_dir, 'actual_over_predicted.png')
    plt.savefig(actual_vs_pred_path, bbox_inches='tight')
    plt.close()

    # Plot 2: Residuals Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    plt.scatter(y_pred, residuals, alpha=0.3, color='darkorange')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Price ($)")
    plt.ylabel("Residual Error ($)")
    plt.title("Residual Error Analysis")
    plt.grid(True, linestyle=':', alpha=0.6)
    residuals_path = os.path.join(output_dir, 'residuals_plot.png')
    plt.savefig(residuals_path, bbox_inches='tight')
    plt.close()

    # Plot 3: Feature Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': X_train.columns, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(15), palette='viridis')
    plt.title('Top 15 Most Important Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    importance_path = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(importance_path, bbox_inches='tight')
    plt.close()

    print(f"\nEvaluation plots successfully saved to: {output_dir}")


if __name__ == '__main__':
    train_and_evaluate()
