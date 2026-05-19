import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def load_and_prepare_data(train_path, test_path):
    """
    Load training and test data with path fallbacks.
    Real datasets often exist in multiple possible locations during development.
    """
    train_candidates = [train_path, "/mnt/data/buenos-aires-real-estate-1-Copy1.csv", "/mnt/data/buenos-aires-real-estate-1.csv"]
    test_candidates = [test_path, "/mnt/data/buenos-aires-test-features.csv"]
    
    train_file = next((p for p in train_candidates if os.path.exists(p)), None)
    test_file = next((p for p in test_candidates if os.path.exists(p)), None)
    
    if train_file is None or test_file is None:
        raise FileNotFoundError("Could not locate training or test files.")
    
    return pd.read_csv(train_file), pd.read_csv(test_file)


def filter_buenos_aires_apartments(df, max_price=400000):
    """
    Filter dataset for Capital Federal region apartments under specified price.
    This is a common preprocessing step for real estate analysis.
    """
    df["price_aprox_usd"] = pd.to_numeric(df.get("price_aprox_usd"), errors="coerce")
    
    mask = (
        (df.get("operation") == "sell") &
        df.get("property_type", "").astype(str).str.lower().str.contains("apartment", na=False) &
        df.get("place_with_parent_names", "").astype(str).str.contains("Capital Federal", na=False)
    )
    
    return df.loc[mask, ["surface_covered_in_m2", "price_aprox_usd"]].copy()


def remove_outliers_and_prepare(df, price_col="price_aprox_usd", area_col="surface_covered_in_m2"):
    """
    Remove missing values and outliers using quantile-based bounds.
    Quantile-based outlier removal is preferable to z-score for non-normal distributions.
    """
    df = df.dropna()
    df = df[df[price_col] < 400000]
    df[area_col] = pd.to_numeric(df[area_col], errors="coerce")
    df = df.dropna(subset=[area_col])
    
    low, high = df[area_col].quantile([0.10, 0.90])
    df = df[df[area_col].between(low, high)].reset_index(drop=True)
    
    return df


def train_and_evaluate(X_train, y_train, X_test):
    """
    Train linear regression model and generate predictions.
    Returns model object and predictions for analysis.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_test = model.predict(X_test)
    
    # Training set evaluation
    y_pred_train = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    
    print(f"Training MAE: ${train_mae:,.2f}")
    print(f"Training R2: {train_r2:.4f}")
    print(f"Model intercept: ${model.intercept_:,.2f}")
    print(f"Price per m2: ${model.coef_[0]:,.2f}")
    
    return model, y_pred_test


if __name__ == "__main__":
    # Main execution
    print("Buenos Aires Real Estate Analysis - Task 1\n")
    print("="*50)
    
    try:
        df_train, df_test = load_and_prepare_data(
            "data/buenos-aires-real-estate-1.csv",
            "data/buenos-aires-test-features.csv"
        )
        
        df_filtered = filter_buenos_aires_apartments(df_train)
        df_clean = remove_outliers_and_prepare(df_filtered)
        
        print(f"Dataset size after filtering: {len(df_clean)} apartments")
        print(f"Price range: ${df_clean['price_aprox_usd'].min():,.0f} - ${df_clean['price_aprox_usd'].max():,.0f}")
        print(f"Area range: {df_clean['surface_covered_in_m2'].min():.1f} - {df_clean['surface_covered_in_m2'].max():.1f} m2\n")
        
        X_train = df_clean[["surface_covered_in_m2"]]
        y_train = df_clean["price_aprox_usd"]
        
        X_test = df_test[["surface_covered_in_m2"]].copy()
        X_test["surface_covered_in_m2"] = X_test["surface_covered_in_m2"].fillna(X_train["surface_covered_in_m2"].median())
        
        model, y_pred = train_and_evaluate(X_train, y_train, X_test)
        
        predictions = pd.Series(y_pred, dtype="float64").round(2)
        print(f"\nTest predictions (first 5):\n{predictions.head()}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data files are in the 'data/' directory.")