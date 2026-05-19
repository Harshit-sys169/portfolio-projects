# =========================
# Imports
# =========================
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import unicodedata


# =========================
# Task 1.5.1
# =========================
df1 = pd.read_csv("data/brasil-real-estate-1.csv")


# =========================
# Task 1.5.2
# =========================
df1 = df1.dropna()
