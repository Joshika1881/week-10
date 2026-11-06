
from pathlib import Path
import pickle

import numpy as np
import pandas as pd


MODEL_DIR = Path(__file__).resolve().parent


def load_model(filename):
    with open(MODEL_DIR / filename, "rb") as f:
        return pickle.load(f)


def normalize_roast(value):
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def predict_rating(df_X, text=False):

    # Bonus Exercise 4: Text predictions
    if text:
        model = load_model("model_3.pickle")

        if isinstance(df_X, pd.DataFrame):
            texts = df_X.iloc[:, 0]
        elif isinstance(df_X, pd.Series):
            texts = df_X
        else:
            texts = pd.Series(df_X)

        texts = texts.fillna("").astype(str)
        return model.predict(texts)

    # Bonus Exercise 3: Price and roast predictions
    model_1 = load_model("model_1.pickle")
    model_2 = load_model("model_2.pickle")
    roast_mapping = load_model("roast_categories.pickle")

    predictions = []

    for _, row in df_X.iterrows():
        price = float(row["100g_USD"])
        roast = normalize_roast(row["roast"])

        if roast in roast_mapping:
            roast_cat = roast_mapping[roast]
            X = np.array([[price, roast_cat]])
            prediction = model_2.predict(X)[0]
        else:
            X = np.array([[price]])
            prediction = model_1.predict(X)[0]

        predictions.append(prediction)

    return np.array(predictions)
