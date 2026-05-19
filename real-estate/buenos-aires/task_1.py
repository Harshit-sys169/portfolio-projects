import os
import pandas as pd
from sklearn.linear_model import LinearRegression

train_candidates = ["data/buenos-aires-real-estate-1.csv", "/mnt/data/buenos-aires-real-estate-1-Copy1.csv", "/mnt/data/buenos-aires-real-estate-1.csv"]
test_candidates = ["data/buenos-aires-test-features.csv", "/mnt/data/buenos-aires-test-features.csv"]

train_path = next((p for p in train_candidates if os.path.exists(p)), None)
test_path = next((p for p in test_candidates if os.path.exists(p)), None)

df = pd.read_csv(train_path)

df["price_aprox_usd"] = pd.to_numeric(df.get("price_aprox_usd"), errors="coerce")
mask = (
    (df.get("operation") == "sell") &
    df.get("property_type", "").astype(str).str.lower().str.contains("apartment", na=False) &
    df.get("place_with_parent_names", "").astype(str).str.contains("Capital Federal", na=False)
)
df = df.loc[mask, ["surface_covered_in_m2", "price_aprox_usd"]].copy()
df = df.dropna()
df = df[df["price_aprox_usd"] < 400000]
df["surface_covered_in_m2"] = pd.to_numeric(df["surface_covered_in_m2"], errors="coerce")
df = df.dropna(subset=["surface_covered_in_m2"])
low, high = df["surface_covered_in_m2"].quantile([0.10, 0.90])
df = df[df["surface_covered_in_m2"].between(low, high)].reset_index(drop=True)

X_train = df[["surface_covered_in_m2"]]
y_train = df["price_aprox_usd"]

model = LinearRegression()
model.fit(X_train, y_train)

X_test = pd.read_csv(test_path)
if "surface_covered_in_m2" not in X_test.columns and "surface_total_in_m2" in X_test.columns:
    X_test["surface_covered_in_m2"] = X_test["surface_total_in_m2"]
X_test["surface_covered_in_m2"] = pd.to_numeric(X_test["surface_covered_in_m2"], errors="coerce")
X_test_feat = X_test[["surface_covered_in_m2"]].copy()
X_test_feat["surface_covered_in_m2"] = X_test_feat["surface_covered_in_m2"].fillna(X_train["surface_covered_in_m2"].median())

y_pred_test = pd.Series(model.predict(X_test_feat), index=X_test.index, dtype="float64").round(6)
y_pred_test.head(5)