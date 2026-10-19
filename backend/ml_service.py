"""Module 2.2 - Data analysis (EDA) and machine learning.

Dataset 1: Titanic  -> EDA + compare 6 models + final model + predictions on test.csv
Dataset 2: MovieLens 1M -> EDA (ratings by gender, age, genre, top movies)
"""
import json
import threading

import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .config import DATA_DIR, PRECOMPUTED_DIR, PYDATA_BASE

RANDOM_STATE = 42
NUM_COLS = ["Age", "SibSp", "Parch", "Fare", "FamilySize"]
CAT_COLS = ["Pclass", "Sex", "Embarked", "IsAlone"]
RESULTS_FILE = DATA_DIR / "ml_results.json"
_lock = threading.Lock()
_cache = {}          # in-memory cache: final model, results, EDA summaries


# ---------------------------------------------------------------- helpers
def _download(rel_path: str):
    """Download a dataset file once and keep it in the data folder."""
    target = DATA_DIR / rel_path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{PYDATA_BASE}/{rel_path}", timeout=60)
        r.raise_for_status()
        target.write_bytes(r.content)
    return target


def load_titanic():
    train = pd.read_csv(_download("titanic/train.csv"))
    test = pd.read_csv(_download("titanic/test.csv"))
    return train, test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple feature engineering that is easy to explain."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1      # the person + family on board
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Pclass"] = df["Pclass"].astype(str)                 # treat class as a category
    df["IsAlone"] = df["IsAlone"].astype(str)
    return df[NUM_COLS + CAT_COLS]


def make_preprocessor():
    """Fill missing values, scale numbers, one-hot encode categories."""
    num = Pipeline([("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUM_COLS), ("cat", cat, CAT_COLS)],
                             sparse_threshold=0)


def candidate_models():
    """The 6 models from the assignment, with simple, reasonable settings."""
    return {
        "kNN": KNeighborsClassifier(n_neighbors=7),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=7,
                                                random_state=RANDOM_STATE, n_jobs=1),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                                 subsample=0.9, eval_metric="logloss",
                                 random_state=RANDOM_STATE, n_jobs=1),
    }


def _scores(y_true, y_pred, y_prob):
    return {
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
    }


# ---------------------------------------------------------------- training
def train_and_compare():
    """Compare 6 models, choose the best one, and predict test.csv.

    test.csv has NO 'Survived' column, so:
      * 80% of train.csv -> 5-fold cross-validation to compare models
      * 20% of train.csv -> hold-out validation = final performance report
      * test.csv -> predictions only (no score is possible)
    """
    with _lock:
        if "results" in _cache:
            return _cache["results"]

        train, test = load_titanic()
        X, y = add_features(train), train["Survived"]
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scoring = {"Recall": "recall", "F1": "f1",
                   "AUROC": "roc_auc", "PR-AUC": "average_precision"}
        rows = []
        for name, clf in candidate_models().items():
            pipe = Pipeline([("prep", make_preprocessor()), ("model", clf)])
            cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring)
            row = {"Model": name}
            for m in scoring:
                row[m] = round(float(np.mean(cvres[f"test_{m}"])), 4)
                row[f"{m} std"] = round(float(np.std(cvres[f"test_{m}"])), 4)
            # Selection score: simple average of the 4 metrics (all have equal weight)
            row["Mean of 4"] = round(float(np.mean([row[m] for m in scoring])), 4)
            rows.append(row)

        table = sorted(rows, key=lambda r: r["Mean of 4"], reverse=True)
        best_name = table[0]["Model"]

        # Fit the best model on the 80% part, score it on the 20% hold-out
        best = Pipeline([("prep", make_preprocessor()),
                         ("model", candidate_models()[best_name])])
        best.fit(X_tr, y_tr)
        prob = best.predict_proba(X_val)[:, 1]
        holdout = {k: round(float(v), 4)
                   for k, v in _scores(y_val, (prob >= 0.5).astype(int), prob).items()}
        holdout["Accuracy"] = round(float(((prob >= 0.5).astype(int) == y_val).mean()), 4)

        # Refit on ALL of train.csv, then predict test.csv (no labels -> no score)
        final = Pipeline([("prep", make_preprocessor()),
                          ("model", candidate_models()[best_name])])
        final.fit(X, y)
        test_prob = final.predict_proba(add_features(test))[:, 1]
        test_pred = (test_prob >= 0.5).astype(int)
        pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_pred}) \
            .to_csv(DATA_DIR / "titanic_test_predictions.csv", index=False)

        results = {
            "comparison": table,
            "best_model": best_name,
            "selection_rule": "Highest average of Recall, F1, AUROC and PR-AUC "
                              "(5-fold cross-validation on 80% of train.csv)",
            "holdout_metrics": holdout,
            "holdout_size": int(len(y_val)),
            "test_has_label": "Survived" in test.columns,
            "test_rows": int(len(test)),
            "test_predicted_survival_rate": round(float(test_pred.mean()), 4),
            "test_preview": pd.DataFrame({
                "PassengerId": test["PassengerId"], "Name": test["Name"],
                "Sex": test["Sex"], "Pclass": test["Pclass"],
                "Survival probability": np.round(test_prob, 3),
                "Predicted Survived": test_pred}).head(15).to_dict("records"),
        }
        _cache["results"] = results
        _cache["final_model"] = final
        RESULTS_FILE.write_text(json.dumps(results, indent=2))
        return results


def predict_passenger(p: dict) -> dict:
    """Predict survival for one passenger typed in the Streamlit form."""
    train_and_compare()                       # makes sure the final model exists
    model = _cache["final_model"]
    df = add_features(pd.DataFrame([p]))
    prob = float(model.predict_proba(df)[0, 1])
    return {"survived": int(prob >= 0.5), "probability": round(prob, 4),
            "model": _cache["results"]["best_model"]}


def test_predictions_csv() -> str:
    train_and_compare()
    return (DATA_DIR / "titanic_test_predictions.csv").read_text()


# ---------------------------------------------------------------- EDA
def titanic_eda() -> dict:
    if "titanic_eda" in _cache:
        return _cache["titanic_eda"]
    train, test = load_titanic()
    df = train.copy()
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 18, 30, 45, 60, 80],
                            labels=["0-12", "13-18", "19-30", "31-45", "46-60", "61-80"])
    num = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]]
    out = {
        "shape_train": list(train.shape),
        "shape_test": list(test.shape),
        "test_columns": list(test.columns),
        "test_has_survived": "Survived" in test.columns,
        "missing_train": train.isna().sum().to_dict(),
        "missing_test": test.isna().sum().to_dict(),
        "survival_rate": round(float(df["Survived"].mean()), 4),
        "by_sex": df.groupby("Sex")["Survived"].mean().round(4).to_dict(),
        "by_class": {str(k): v for k, v in
                     df.groupby("Pclass")["Survived"].mean().round(4).to_dict().items()},
        "by_age_group": {str(k): v for k, v in
                         df.groupby("AgeGroup", observed=False)["Survived"]
                         .mean().round(4).fillna(0).to_dict().items()},
        "by_sex_class": df.pivot_table(index="Pclass", columns="Sex", values="Survived")
                          .round(4).rename(index=str).to_dict(),
        "ages": df["Age"].dropna().round(1).tolist(),
        "ages_survived": df["Survived"][df["Age"].notna()].tolist(),
        "fares": df["Fare"].round(2).tolist(),
        "corr": num.corr().round(3).to_dict(),
        "head": train.head(10).astype(object).where(train.head(10).notna(), None)
                     .to_dict("records"),
    }
    _cache["titanic_eda"] = out
    return out


def movielens_eda() -> dict:
    """MovieLens 1M (Chapter 13.2): 1,000,209 ratings by 6,040 users on 3,706 movies."""
    if "movielens_eda" in _cache:
        return _cache["movielens_eda"]
    # The full analysis needs a lot of memory (1 million rows), so the summary is computed
    # once on a PC and committed as a small JSON file. The cloud server only reads it.
    # Delete backend/precomputed/movielens_eda.json to compute it again from the raw data.
    for cache_file in (PRECOMPUTED_DIR / "movielens_eda.json", DATA_DIR / "movielens_eda.json"):
        if cache_file.exists():
            _cache["movielens_eda"] = json.loads(cache_file.read_text())
            return _cache["movielens_eda"]
    cache_file = DATA_DIR / "movielens_eda.json"

    opts = dict(sep="::", engine="python", encoding="latin-1", header=None)
    users = pd.read_table(_download("movielens/users.dat"),
                          names=["user_id", "gender", "age", "occupation", "zip"], **opts)
    ratings = pd.read_table(_download("movielens/ratings.dat"),
                            names=["user_id", "movie_id", "rating", "timestamp"], **opts)
    movies = pd.read_table(_download("movielens/movies.dat"),
                           names=["movie_id", "title", "genres"], **opts)
    data = ratings.merge(users[["user_id", "gender", "age"]], on="user_id") \
                  .merge(movies, on="movie_id")

    age_names = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44",
                 45: "45-49", 50: "50-55", 56: "56+"}

    # Average rating per movie and gender (only movies with >= 250 ratings, like the book)
    counts = data.groupby("title").size()
    active = counts.index[counts >= 250]
    by_gender = data.pivot_table("rating", index="title", columns="gender",
                                 aggfunc="mean").loc[active]
    by_gender["diff"] = by_gender["M"] - by_gender["F"]
    top_f = by_gender.sort_values("F", ascending=False).head(10)

    genres = data[["genres", "rating"]].assign(genre=data["genres"].str.split("|")) \
        .explode("genre")
    genre_stats = genres.groupby("genre")["rating"].agg(["count", "mean"]).round(3) \
        .sort_values("count", ascending=False)

    out = {
        "n_ratings": int(len(ratings)), "n_users": int(len(users)),
        "n_movies": int(movies["movie_id"].nunique()),
        "rating_distribution": {str(k): int(v) for k, v in
                                ratings["rating"].value_counts().sort_index().items()},
        "users_by_gender": users["gender"].value_counts().to_dict(),
        "mean_rating_by_gender": data.groupby("gender")["rating"].mean().round(3).to_dict(),
        "mean_rating_by_age": {age_names[k]: v for k, v in
                               data.groupby("age")["rating"].mean().round(3).to_dict().items()},
        "genres": genre_stats.reset_index().to_dict("records"),
        "top_by_women": top_f.round(3).reset_index()[["title", "F", "M"]].to_dict("records"),
        "men_prefer": by_gender.sort_values("diff", ascending=False).head(10).round(3)
                               .reset_index()[["title", "F", "M", "diff"]].to_dict("records"),
        "women_prefer": by_gender.sort_values("diff").head(10).round(3)
                                 .reset_index()[["title", "F", "M", "diff"]].to_dict("records"),
        "most_disagreed": data[data["title"].isin(active)].groupby("title")["rating"]
                          .std().sort_values(ascending=False).head(10).round(3)
                          .reset_index().rename(columns={"rating": "std"}).to_dict("records"),
    }
    cache_file.write_text(json.dumps(out))
    _cache["movielens_eda"] = out
    del data, genres, ratings                    # free memory (important on free servers)
    return out
