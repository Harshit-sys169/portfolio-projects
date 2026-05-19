# =========================
# Section 2.3: Predicting Price with Neighborhood
# =========================

# Imports
import warnings
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from category_encoders import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)


# =========================
# Wrangle Function
# =========================
def wrangle(filepath):
    df = pd.read_csv(filepath)

    # Filter Buenos Aires apartments under $400k
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    # Remove area outliers
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    df = df[df["surface_covered_in_m2"].between(low, high)]

    # Extract neighborhood
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]

    # Drop unused column
    df.drop(columns="place_with_parent_names", inplace=True)

    return df


# =========================
# Load & Combine Data
# =========================
files = glob("data/buenos-aires-real-estate-*.csv")

frames = []
for file in files:
    frames.append(wrangle(file))

df = pd.concat(frames, ignore_index=True)


# =========================
# Features & Target
# =========================
target = "price_aprox_usd"
features = ["neighborhood"]

X_train = df[features]
y_train = df[target]


# =========================
# Baseline MAE
# =========================
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)

print("Mean apt price:", round(y_mean, 2))
print("Baseline MAE:", round(mean_absolute_error(y_train, y_pred_baseline), 2))


# =========================
# Model (OneHot + Ridge)
# =========================
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    Ridge(alpha=1.0)
)

model.fit(X_train, y_train)
check_is_fitted(model[-1])


# =========================
# Training MAE
# =========================
y_pred_training = model.predict(X_train)
mae_training = mean_absolute_error(y_train, y_pred_training)

print("Training MAE:", round(mae_training, 2))


# =========================
# Test Predictions
# =========================
X_test = pd.read_csv("data/buenos-aires-test-features.csv")[features]
y_pred_test = pd.Series(model.predict(X_test)).astype(float)

pd.options.display.float_format = "{:.6f}".format
y_pred_test.head()


# =========================
# Extract Coefficients
# =========================
intercept = model[-1].intercept_
coefficients = model[-1].coef_

print("coefficients len:", len(coefficients))
print(coefficients[:5])


# =========================
# Feature Names
# =========================
feature_names = model[0].get_feature_names_out()

print("features len:", len(feature_names))
print(feature_names[:5])


# =========================
# Feature Importance Series
# =========================
feat_imp = pd.Series(coefficients, index=feature_names)
feat_imp.head()


# =========================
# Print Model Equation
# =========================
print(f"price = {intercept.round(2)}")
for f, c in feat_imp.items():
    print(f"+ ({round(c, 2)} * {f})")


# =========================
# Top 15 Feature Importance Plot
# =========================
top_15 = feat_imp.reindex(
    feat_imp.abs().sort_values(ascending=False).head(15).index
)

plt.figure(figsize=(8, 6))
top_15.sort_values().plot(kind="barh")

plt.xlabel("Coefficient Value")
plt.ylabel("Neighborhood")
plt.title("Top 15 Neighborhood Coefficients by Absolute Value")

plt.tight_layout()
plt.show()