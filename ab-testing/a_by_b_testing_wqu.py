# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import random
from datetime import datetime
from pymongo import MongoClient
import country_converter as coco
import plotly.express as px
from scipy.stats import norm
from statsmodels.stats.power import GofChisquarePower
from statsmodels.stats.contingency_tables import Table2x2
from teaching_tools.ab_test.reset import Reset
from teaching_tools.ab_test.experiment import Experiment

# =========================
# MongoDB Connection
# =========================
host = "192.178.116.2"
client = MongoClient(host=host, port=27017)
db = client["wqu-abtest"]
mscfe_app = db["mscfe-applicants"]

# =========================
# Reset Database
# =========================
r = Reset(host)
r.reset_database()

# =========================
# Task 7.5.2 – Aggregate by nationality
# =========================
pipeline = [
    {"$group": {"_id": "$countryISO2", "count": {"$sum": 1}}},
    {"$sort": {"_id": 1}}
]
df_nationality = (
    pd.DataFrame(list(mscfe_app.aggregate(pipeline)))
    .rename(columns={"_id": "country_iso2"})
    .loc[:, ["country_iso2", "count"]]
    .reset_index(drop=True)
)

# =========================
# Task 7.5.3 – Country names
# =========================
cc = coco.CountryConverter()
df_nationality["country_name"] = cc.convert(df_nationality["country_iso2"], to="short_name")
df_nationality["country_iso3"] = cc.convert(df_nationality["country_iso2"], to="ISO3")

# =========================
# Task 7.5.4 – Choropleth
# =========================
def build_nat_choropleth():
    fig = px.choropleth(
        df_nationality,
        locations="country_iso3",
        color="count",
        hover_name="country_name",
        title="MScFE Applicants: Nationalities",
        color_continuous_scale=px.colors.sequential.Oranges,
        projection="natural earth"
    )
    return fig

nat_fig = build_nat_choropleth()

# =========================
# MongoRepository Class
# =========================
class MongoRepository:
    def __init__(self, collection, db="wqu-abtest", host=host, port=27017):
        client = MongoClient(host=host, port=port)
        self.collection = client[db][collection]

    def find_by_date(self, date):
        start = datetime(date.year, date.month, date.day)
        end = datetime(date.year, date.month, date.day, 23, 59, 59)
        query = {
            "createdAt": {"$gte": start, "$lte": end},
            "admissionsQuiz": "incomplete"
        }
        return list(self.collection.find(query))

    def update_applicants(self, applicants):
        n = 0
        nModified = 0
        for applicant in applicants:
            result = self.collection.update_one(
                {"_id": applicant["_id"]},
                {"$set": {"group": applicant["group"], "inExperiment": True}}
            )
            n += result.matched_count
            nModified += result.modified_count
        return {"n": n, "nModified": nModified}

    def assign_to_groups(self, date):
        applicants = self.find_by_date(date)
        for applicant in applicants:
            applicant["group"] = random.choice(["email (t)", "no email (c)"])
        return self.update_applicants(applicants)

    def find_exp_observations(self):
        return list(self.collection.find({"inExperiment": True}))

repo = MongoRepository("mscfe-applicants")

# =========================
# Task 7.5.8 – Power analysis
# =========================
chi_square_power = GofChisquarePower()
group_size = int(np.ceil(chi_square_power.solve_power(effect_size=0.5, alpha=0.05, power=0.8, n_bins=4)))
total_applicants = 64

# =========================
# Task 7.5.9 – No-quiz per day
# =========================
pipeline = [
    {"$match": {"admissionsQuiz": "incomplete"}},
    {
        "$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$createdAt"}},
            "new_users": {"$sum": 1}
        }
    },
    {"$sort": {"_id": 1}}
]

no_quiz_mscfe = (
    pd.DataFrame(list(mscfe_app.aggregate(pipeline)))
    .rename(columns={"_id": "date"})
    .assign(date=lambda df: pd.to_datetime(df["date"]))
    .set_index("date")["new_users"]
)
no_quiz_mscfe.name = "new_users"
no_quiz_mscfe.index.name = "date"

# =========================
# Task 7.5.10 – Mean & Std
# =========================
mean = no_quiz_mscfe.mean()
std = no_quiz_mscfe.std()

# =========================
# Task 7.5.11 – Experiment duration
# =========================
target = 65
for exp_days in range(1, 31):
    sum_mean = mean * exp_days
    sum_std = std * (exp_days ** 0.5)
    prob = 1 - norm.cdf(target - 1, sum_mean, sum_std)
    if prob >= 0.95:
        break

# =========================
# Task 7.5.12 – Run experiment
# =========================
exp = Experiment(repo=repo, db="wqu-abtest", collection="mscfe-applicants")
exp.reset_experiment()
result = exp.run_experiment(days=exp_days, assignment=True)

# =========================
# Task 7.5.14 – Load experiment data
# =========================
df = pd.DataFrame(repo.find_exp_observations())

# =========================
# Task 7.5.15 – Crosstab
# =========================
data = pd.crosstab(df["group"], df["admissionsQuiz"])

# =========================
# Task 7.5.16 – Bar chart
# =========================
def build_contingency_bar():
    temp = data.reset_index()
    df_plot = temp.melt(
        id_vars="group",
        var_name="Admissions Quiz",
        value_name="Frequency [count]"
    )
    fig = px.bar(
        df_plot,
        x="group",
        y="Frequency [count]",
        color="Admissions Quiz",
        barmode="group",
        title="MScFE: Admissions Quiz Completion by Group",
        labels={"group": "Group"}
    )
    return fig

cb_fig = build_contingency_bar()

# =========================
# Task 7.5.17 – Contingency table
# =========================
contingency_table = Table2x2(data.values)

# =========================
# Task 7.5.18 – Chi-square test
# =========================
chi_square_test = contingency_table.test_nominal_association()

# =========================
# Task 7.5.19 – Odds ratio
# =========================
odds_ratio = round(contingency_table.oddsratio, 1)