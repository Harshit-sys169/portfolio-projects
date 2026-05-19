# =========================
# IMPORTS & SETUP
# =========================
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)


# =========================
# WRANGLE FUNCTION
# =========================
def wrangle(filepath):
    df = pd.read_csv(filepath)

    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    df = df[df["surface_covered_in_m2"].between(low, high)]

    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True)
    df["lat"] = pd.to_numeric(df["lat"].str.strip(), errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"].str.strip(), errors="coerce")

    df = df.drop(columns="lat-lon")

    return df


# =========================
# LOAD & CONCAT DATA
# =========================
frame1 = wrangle("data/buenos-aires-real-estate-1.csv")
frame2 = wrangle("data/buenos-aires-real-estate-2.csv")

df = pd.concat([frame1, frame2], ignore_index=True)


# =========================
# FEATURES & TARGET
# =========================
X_train = df[["lon", "lat"]]
y_train = df["price_aprox_usd"]


# =========================
# BASELINE MODEL
# =========================
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)

mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print("Mean apt price:", round(y_mean, 2))
print("Baseline MAE:", round(mae_baseline, 2))


# =========================
# PIPELINE MODEL
# =========================
model = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)
check_is_fitted(model["regressor"])


# =========================
# TRAINING PREDICTIONS
# =========================
y_pred_training = model.predict(X_train)
mae_training = mean_absolute_error(y_train, y_pred_training)
print("Training MAE:", round(mae_training, 2))


# =========================
# TEST PREDICTIONS
# =========================
X_test = pd.read_csv("data/buenos-aires-test-features.csv")
X_test = X_test[["lon", "lat"]]

y_pred_test = model.predict(X_test)
y_pred_test = pd.Series(y_pred_test)
y_pred_test.head()


# =========================
# MODEL PARAMETERS
# =========================
intercept = model["regressor"].intercept_
coefficients = model["regressor"].coef_

print(
    f"price = {intercept:.2f} + ({coefficients[0]:.2f} * longitude) + ({coefficients[1]:.2f} * latitude)"
)