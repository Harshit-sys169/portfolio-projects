import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------- Task 6.5.2 ----------
prop_biz_owners = df["HBUS"].mean()
print("proportion of business owners in df:", prop_biz_owners)

# ---------- Task 6.5.3 ----------
inccat_dict = {
    1: "0-20",
    2: "21-39.9",
    3: "40-59.9",
    4: "60-79.9",
    5: "80-89.9",
    6: "90-100"
}

df_inccat = (
    df.assign(INCCAT=df["INCCAT"].map(inccat_dict))
      .groupby(["HBUS", "INCCAT"])
      .size()
      .groupby(level=0)
      .apply(lambda x: x / x.sum())
      .reset_index(name="frequency")
)

# ---------- Task 6.5.4 ----------
fig, ax = plt.subplots()
sns.barplot(
    data=df_inccat,
    x="INCCAT",
    y="frequency",
    hue="HBUS",
    order=list(inccat_dict.values()),
    ax=ax
)
ax.set_title("Income Distribution: Business Owners vs. Non-Business Owners")
ax.set_xlabel("Income Category")
ax.set_ylabel("Frequency")
plt.show()

# ---------- Task 6.5.5 ----------
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="DEBT",
    y="HOUSES",
    hue="HBUS",
    palette="deep",
    ax=ax
)
ax.set_title("Home Value vs. Household Debt")
plt.show()

# ---------- Task 6.5.6 ----------
mask = (df["HBUS"] == 1) & (df["INCOME"] < 500000)
df_small_biz = df.loc[mask]
print("df_small_biz shape:", df_small_biz.shape)

# ---------- Task 6.5.7 ----------
fig, ax = plt.subplots()
df_small_biz["AGE"].plot(kind="hist", bins=10, ax=ax)
ax.set_title("Small Business Owners: Age Distribution")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency (count)")
plt.show()

# ---------- Task 6.5.8 ----------
top_ten_var = (
    df_small_biz.select_dtypes(include=np.number)
    .var(ddof=0)
    .sort_values(ascending=True)
    .tail(10)
)
print(top_ten_var)

# ---------- Task 6.5.9 ----------
def trimmed_variance(series, trim=0.1):
    s = series.sort_values()
    n = len(s)
    k = int(n * trim)
    return s.iloc[k:n-k].var(ddof=0)

top_ten_trim_var = (
    df_small_biz.select_dtypes(include=np.number)
    .apply(trimmed_variance)
    .sort_values(ascending=True)
    .tail(10)
)
print(top_ten_trim_var)

# ---------- Task 6.5.10 ----------
fig = px.bar(
    x=top_ten_trim_var.values,
    y=top_ten_trim_var.index,
    orientation="h",
    title="Small Business Owners: High Variance Features"
)
fig.update_layout(
    xaxis_title="Trimmed Variance [$]",
    yaxis_title="Feature"
)
fig.show()

# ---------- Task 6.5.11 ----------
high_var_cols = top_ten_trim_var.sort_values().tail(5).index.tolist()
print(high_var_cols)

# ---------- Task 6.5.12 ----------
X = df_small_biz[high_var_cols]
print("X shape:", X.shape)
X.head()

# ---------- Task 6.5.13 ----------
n_clusters = list(range(2, 13))
inertia_errors = []
silhouette_scores = []

for k in n_clusters:
    pipe = Pipeline([
        ("standardscaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, random_state=42))
    ])
    pipe.fit(X)
    labels_k = pipe.named_steps["kmeans"].labels_
    inertia_errors.append(pipe.named_steps["kmeans"].inertia_)
    silhouette_scores.append(silhouette_score(StandardScaler().fit_transform(X), labels_k))

print("Inertia:", inertia_errors)
print("Silhouette Scores:", silhouette_scores)

# ---------- Task 6.5.14 ----------
fig = px.line(
    x=n_clusters,
    y=inertia_errors,
    title="K-Means Model: Inertia vs Number of Clusters"
)
fig.update_layout(
    xaxis_title="Number of Clusters",
    yaxis_title="Inertia"
)
fig.show()

# ---------- Task 6.5.15 ----------
fig = px.line(
    x=n_clusters,
    y=silhouette_scores,
    title="K-Means Model: Silhouette Score vs Number of Clusters"
)
fig.update_layout(
    xaxis_title="Number of Clusters",
    yaxis_title="Silhouette Score"
)
fig.show()

# ---------- Task 6.5.16 ----------
final_model = Pipeline([
    ("standardscaler", StandardScaler()),
    ("kmeans", KMeans(n_clusters=3, random_state=42))
])
final_model.fit(X)

# ---------- Task 6.5.17 ----------
labels = final_model.named_steps["kmeans"].labels_
xgb = (
    pd.DataFrame(
        final_model.named_steps["standardscaler"].inverse_transform(
            final_model.named_steps["standardscaler"].transform(X)
        ),
        columns=X.columns
    )
    .assign(Cluster=labels)
    .groupby("Cluster")
    .mean()
)
print(xgb)

# ---------- Task 6.5.18 ----------
fig = px.bar(
    xgb,
    barmode="group",
    title="Small Business Owner Finances by Cluster"
)
fig.update_layout(
    xaxis_title="Cluster",
    yaxis_title="Value [$]"
)
fig.show()

# ---------- Task 6.5.19 ----------
pca = PCA(n_components=2, random_state=42)
X_t = pca.fit_transform(X)
X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])
print("X_pca shape:", X_pca.shape)
X_pca.head()

# ---------- Task 6.5.20 ----------
fig = px.scatter(
    data_frame=X_pca,
    x="PC1",
    y="PC2",
    color=labels.astype(str),
    title="PCA Representation of Clusters"
)
fig.update_layout(
    xaxis_title="PC1",
    yaxis_title="PC2"
)
fig.show()