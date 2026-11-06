
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline


# Load the coffee dataset
url = (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/"
    "refs/heads/main/data/coffee_analysis.csv"
)

df = pd.read_csv(url)

df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["100g_USD"] = pd.to_numeric(df["100g_USD"], errors="coerce")


# Exercise 1: Linear Regression
# Predict rating based on coffee price

df_1 = df.dropna(subset=["100g_USD", "rating"])

X = df_1[["100g_USD"]].values
y = df_1["rating"].values

lr = LinearRegression()
lr.fit(X, y)

with open("model_1.pickle", "wb") as f:
    pickle.dump(lr, f)

print("Exercise 1: model_1.pickle saved")


# Exercise 2: Decision Tree Regressor
# Predict rating using price and roast

roast_mapping = {
    "light": 0,
    "medium-light": 1,
    "medium": 2,
    "medium-dark": 3,
    "dark": 4
}


def normalize_roast(value):
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    return value if value else None


# Include any additional roast labels in the dataset
extra_roasts = sorted(
    set(df["roast"].map(normalize_roast).dropna())
    - set(roast_mapping)
)

for roast in extra_roasts:
    roast_mapping[roast] = max(roast_mapping.values()) + 1


def roast_category(value):
    roast = normalize_roast(value)
    return roast_mapping.get(roast, np.nan)


df_2 = df.dropna(subset=["100g_USD", "rating"]).copy()

df_2["roast_cat"] = df_2["roast"].map(roast_category)

X = df_2[["100g_USD", "roast_cat"]].values
y = df_2["rating"].values

dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X, y)

with open("model_2.pickle", "wb") as f:
    pickle.dump(dtr, f)

with open("roast_categories.pickle", "wb") as f:
    pickle.dump(roast_mapping, f)

print("Exercise 2: model_2.pickle saved")


# Bonus Exercise 4: TF-IDF Text Regression
# Predict rating using coffee review descriptions

df_3 = df.dropna(subset=["rating"]).copy()

df_3["desc_3"] = (
    df_3["desc_3"].fillna("").astype(str).str.strip()
)

df_3 = df_3[df_3["desc_3"] != ""]

X = df_3["desc_3"]
y = df_3["rating"]

text_model = make_pipeline(
    TfidfVectorizer(
        max_features=1000,
        stop_words="english"
    ),
    LinearRegression()
)

text_model.fit(X, y)

with open("model_3.pickle", "wb") as f:
    pickle.dump(text_model, f)

print("Bonus Exercise 4: model_3.pickle saved")
print("All models trained successfully!")
