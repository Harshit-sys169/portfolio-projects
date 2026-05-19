# =========================
# TASK 5.5 – COMPLETE SOLUTION (ALL TASKS)
# =========================

# -------------------------
# Imports
# -------------------------
import gzip
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from imblearn.over_sampling import RandomOverSampler


# =========================
# Task 5.5.1 – Load data
# =========================
with gzip.open("data/taiwan-bankruptcy-data.json.gz", "rt", encoding="utf-8") as f:
    taiwan_data = json.load(f)


# =========================
# Task 5.5.2 – Extract keys
# =========================
taiwan_data_keys = taiwan_data.keys()


# =========================
# Task 5.5.3 – Number of companies
# =========================
n_companies = len(taiwan_data["observations"])


# =========================
# Task 5.5.4 – Number of features
# =========================
n_features = len(taiwan_data["observations"][0])


# =========================
# Task 5.5.5 – Wrangle function
# =========================
def wrangle(filepath):
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["observations"])
    df = df.set_index("id")
    return df


df = wrangle("data/taiwan-bankruptcy-data.json.gz")


# =========================
# Task 5.5.6 – Missing data
# =========================
nans_by_col = df.isna().sum()


# =========================
# Task 5.5.7 – Class balance plot
# =========================
fig, ax = plt.subplots()
df["bankrupt"].value_counts(normalize=True).plot(kind="bar", ax=ax)
ax.set_xlabel("Bankrupt")
ax.set_ylabel("Frequency")
ax.set_title("Class Balance")
plt.show()


# =========================
# Task 5.5.8 – Feature matrix & target
# =========================
X = df.drop(columns="bankrupt")
y = df["bankrupt"]


# =========================
# Task 5.5.9 – Train-test split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# Task 5.5.10 – Oversampling
# =========================
over_sampler = RandomOverSampler(random_state=42)
X_train_over, y_train_over = over_sampler.fit_resample(X_train, y_train)


# =========================
# Task 5.5.11 – Classifier
# =========================
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_over, y_train_over)


# =========================
# Task 5.5.12 – Cross-validation
# =========================
cv_scores = cross_val_score(
    clf, X_train_over, y_train_over, cv=5, n_jobs=-1
)


# =========================
# Task 5.5.13 – GridSearchCV
# =========================
params = {
    "max_depth": range(30, 50, 10),
    "n_estimators": range(25, 51, 25)
}

model = GridSearchCV(
    clf,
    param_grid=params,
    cv=5,
    n_jobs=-1,
    verbose=1
)

model.fit(X_train_over, y_train_over)


# =========================
# Task 5.5.14 – CV results
# =========================
cv_results = pd.DataFrame(model.cv_results_)


# =========================
# Task 5.5.15 – Best parameters
# =========================
best_params = model.best_params_


# =========================
# Task 5.5.16 – Confusion matrix
# =========================
fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_estimator(
    model.best_estimator_,
    X_test,
    y_test,
    ax=ax
)
plt.show()


# =========================
# Task 5.5.17 – Classification report
# =========================
class_report = classification_report(
    y_test,
    model.best_estimator_.predict(X_test)
)
print(class_report)


# =========================
# Task 5.5.18 – Feature importance
# =========================
importances = model.best_estimator_.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values().tail(10)

fig, ax = plt.subplots()
feat_imp.plot(kind="barh", ax=ax)
ax.set_xlabel("Gini Importance")
ax.set_ylabel("Feature")
ax.set_title("Feature Importance")
plt.show()


# =========================
# Task 5.5.19 – Save model (GridSearchCV)
# =========================
with open("model-5-5.pkl", "wb") as f:
    pickle.dump(model, f)


# =========================
# Task 5.5.20 – Predictor module
# =========================
# my_predictor_assignment.py

import gzip
import json
import pickle
import pandas as pd


def wrangle(filepath):
    with gzip.open(filepath, "rt", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data["observations"])
    df = df.set_index("id")
    return df


def make_predictions(data_filepath, model_filepath):
    X = wrangle(data_filepath)

    with open(model_filepath, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    return pd.Series(y_pred, index=X.index)


# -------------------------
# Test predictor
# -------------------------
y_test_pred = make_predictions(
    data_filepath="data/taiwan-bankruptcy-data-test-features.json.gz",
    model_filepath="model-5-5.pkl",
)

print("predictions shape:", y_test_pred.shape)
y_test_pred.head()