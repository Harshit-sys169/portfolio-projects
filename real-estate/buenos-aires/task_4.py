import warnings
from glob import glob

import pandas as pd
import seaborn as sns
from category_encoders import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted

warnings.simplefilter(action="ignore", category=FutureWarning)


def wrangle(filepath):
    df = pd.read_csv(filepath)

    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    df = df[mask_ba & mask_apt & mask_price]

    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    df = df[df["surface_covered_in_m2"].between(low, high)]

    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)

    return df


files = glob("data/buenos-aires-real-estate-*.csv")
assert len(files) == 5

frames = [wrangle(file) for file in files]
df = pd.concat(frames, ignore_index=True)

X = df.drop(columns="price_aprox_usd")
y = df["price_aprox_usd"]

y_mean = y.mean()
y_pred_baseline = [y_mean] * len(y)
baseline_mae = mean_absolute_error(y, y_pred_baseline)

print("Mean apartment price:", y_mean)
print("Baseline MAE:", baseline_mae)

lin_reg = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    LinearRegression()
)

lin_reg.fit(X, y)
check_is_fitted(lin_reg)

y_pred_lr = lin_reg.predict(X)
mae_lr = mean_absolute_error(y, y_pred_lr)

print("Linear Regression MAE:", mae_lr)

ridge = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge(alpha=1.0)
)

ridge.fit(X, y)
check_is_fitted(ridge)

y_pred_ridge = ridge.predict(X)
mae_ridge = mean_absolute_error(y, y_pred_ridge)

print("Ridge Regression MAE:", mae_ridge)

pd.Series(
    {
        "Baseline": baseline_mae,
        "Linear Regression": mae_lr,
        "Ridge Regression": mae_ridge,
    }
).sort_values()

features = ridge.named_steps["onehotencoder"].get_feature_names()
coefficients = ridge.named_steps["ridge"].coef_

feat_imp = pd.Series(coefficients, index=features).sort_values(key=abs)
feat_imp.tail(10)

feat_imp.tail(10).plot(kind="barh")