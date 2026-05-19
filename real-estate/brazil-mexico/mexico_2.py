import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


df = pd.read_csv("data/mexico-city-real-estate.csv")

df = df[
    (df["operation"] == "sale") &
    (df["property_type"] == "apartment")
]

df = df[df["price_aprox_usd"].notna()]

df = df[df["surface_covered_in_m2"].between(20, 500)]

df = df.loc[:, df.isna().mean() < 0.5]

df["lat"] = df["lat-lon"].str.split(",", expand=True)[0].astype(float)
df["lon"] = df["lat-lon"].str.split(",", expand=True)[1].astype(float)

df["borough"] = (
    df["place_with_parent_names"]
    .str.split("|", regex=False)
    .str[1]
)

df = df[
    ["price_aprox_usd", "surface_covered_in_m2", "lat", "lon", "borough"]
]

df = df.dropna(subset=["price_aprox_usd", "surface_covered_in_m2"])

df = df.reset_index(drop=True)

print("FINAL ROW COUNT:", len(df))
print(df.info())


# Task 2.5.4 – Histogram
fig, ax = plt.subplots()
ax.hist(df["price_aprox_usd"], bins=50)
ax.set_xlabel("Price [$]")
ax.set_ylabel("Count")
ax.set_title("Distribution of Apartment Prices")
plt.show()


# Task 2.5.5 – Scatter plot
fig, ax = plt.subplots()
ax.scatter(df["surface_covered_in_m2"], df["price_aprox_usd"])
ax.set_xlabel("Area [sq meters]")
ax.set_ylabel("Price [USD]")
ax.set_title("Mexico City: Price vs. Area")
plt.show()


# Task 2.5.7 – Features and target
X_train = df.drop(columns=["price_aprox_usd"])
y_train = df["price_aprox_usd"]


# Task 2.5.8 – Baseline MAE
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)

print("Mean apt price:", y_mean)
print("Baseline MAE:", baseline_mae)


# Task 2.5.9 – Ridge regression pipeline
model = make_pipeline(
    ColumnTransformer(
        transformers=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
                make_column_selector(dtype_include=object)
            ),
            (
                "num",
                make_pipeline(SimpleImputer(), StandardScaler()),
                make_column_selector(dtype_include="number")
            )
        ]
    ),
    Ridge()
)

model.fit(X_train, y_train)


# Task 2.5.10 – Load test data
X_test = pd.read_csv("data/mexico-city-test-features.csv")
print(X_test.info())


# Task 2.5.11 – Predictions
y_test_pred = pd.Series(model.predict(X_test))
print(y_test_pred.head())


# Task 2.5.12 – Feature importances
coefficients = model.named_steps["ridge"].coef_
features = model.named_steps["columntransformer"].get_feature_names_out()

features_clean = (
    pd.Series(features)
    .str.replace("onehot__", "", regex=False)
    .str.replace("num__", "", regex=False)
)

feat_imp = (
    pd.Series(coefficients, index=features_clean)
    .sort_values(key=abs)
)

print(feat_imp.tail(10))


# Task 2.5.13 – Feature importance plot
fig, ax = plt.subplots()
feat_imp.tail(10).plot(kind="barh", ax=ax)
ax.set_xlabel("Importance [USD]")
ax.set_ylabel("Feature")
ax.set_title("Feature Importances for Apartment Price")
plt.show()