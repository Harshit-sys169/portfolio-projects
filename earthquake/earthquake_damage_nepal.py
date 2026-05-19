import sqlite3
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def wrangle(sqlite_path):
    conn = sqlite3.connect(sqlite_path)

    query = """
    SELECT im.b_id,
           bs.age_building,
           bs.plinth_area_sq_ft,
           bs.height_ft_pre_eq,
           bs.land_surface_condition,
           bs.foundation_type,
           bs.roof_type,
           bs.ground_floor_type,
           bs.other_floor_type,
           bs.position,
           bs.plan_configuration,
           bs.superstructure,
           bd.damage_grade
    FROM (
        SELECT DISTINCT building_id AS b_id
        FROM id_map
        WHERE district_id = 3
    ) im
    JOIN building_structure bs ON im.b_id = bs.building_id
    JOIN building_damage bd ON im.b_id = bd.building_id;
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["severe_damage"] = (df["damage_grade"] > "Grade 3").astype(int)

    df = df.drop(columns=["damage_grade", "b_id"])

    return df

df = wrangle("../nepal.sqlite")
fig, ax = plt.subplots()

(df["severe_damage"]
 .value_counts(normalize=True)
 .sort_index()
 .plot(kind="bar", ax=ax))

ax.set_xlabel("Severe Damage")
ax.set_ylabel("Relative Frequency")
ax.set_title("Kavrepalanchok, Class Balance")

plt.show()
fig, ax = plt.subplots()

sns.boxplot(
    x="severe_damage",
    y="plinth_area_sq_ft",
    data=df,
    ax=ax
)

ax.set_xlabel("Severe Damage")
ax.set_ylabel("Plinth Area [sq. ft.]")
ax.set_title("Kavrepalanchok, Plinth Area vs Building Damage")

plt.show()

roof_pivot = (
    df.pivot_table(
        index="roof_type",
        values="severe_damage",
        aggfunc="mean"
    )
    .loc[
        ["RCC/RB/RBC",
         "Bamboo/Timber-Heavy roof",
         "Bamboo/Timber-Light roof"]
    ]
)

roof_pivot
X = df.drop(columns="severe_damage")
y = df["severe_damage"]

print("X shape:", X.shape)
print("y shape:", y.shape)
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
baseline = y_train.value_counts(normalize=True).max()
round(baseline, 2)
model_lr = Pipeline(
    steps=[
        ("onehotencoder", OneHotEncoder(handle_unknown="ignore")),
        ("logisticregression", LogisticRegression(max_iter=1000))
    ]
)
model_lr.fit(X_train, y_train)

train_acc = model_lr.score(X_train, y_train)
val_acc = model_lr.score(X_val, y_val)

train_acc, val_acc
training_acc = {}
validation_acc = {}

for depth in range(1, 16):
    model_dt = Pipeline(
        steps=[
            ("ordinalencoder",
             OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ("decisiontreeclassifier",
             DecisionTreeClassifier(max_depth=depth, random_state=42))
        ]
    )

    model_dt.fit(X_train, y_train)

    training_acc[depth] = model_dt.score(X_train, y_train)
    validation_acc[depth] = model_dt.score(X_val, y_val)

submission = pd.Series(validation_acc)
submission
fig, ax = plt.subplots()

ax.plot(training_acc.keys(), training_acc.values(), label="training")
ax.plot(validation_acc.keys(), validation_acc.values(), label="validation")

ax.set_xlabel("Max Depth")
ax.set_ylabel("Accuracy Score")
ax.set_title("Validation Curve, Decision Tree Model")
ax.legend()

plt.show()
final_model_dt = Pipeline(
    steps=[
        ("ordinalencoder",
         OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("decisiontreeclassifier",
         DecisionTreeClassifier(max_depth=10, random_state=42))
    ]
)

final_model_dt.fit(X_train, y_train)
X_test = pd.read_csv("data/kavrepalanchok-test-features.csv")

X_test = X_test[X_train.columns]

y_test_pred = final_model_dt.predict(X_test)
feat_imp = pd.Series(
    final_model_dt.named_steps["decisiontreeclassifier"].feature_importances_,
    index=X.columns
).sort_values()

feat_imp
fig, ax = plt.subplots()

feat_imp.plot(kind="barh", ax=ax)

ax.set_xlabel("Gini Importance")
ax.set_ylabel("Feature")
ax.set_title("Kavrepalanchok Decision Tree, Feature Importance")

plt.show()