"""
Milestone 4, part 2 -- your first ML model, explained step by step.

We use LogisticRegression here (not something fancier yet) on purpose: it's
the simplest possible classifier, which makes it easier to see exactly what
"fitting a model" means before adding any complexity. It predicts a
probability between 0 and 1 (here: "probability price is up over the next
5 days") by learning a weighted combination of the input features.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from features import build_features, build_label_fixed_horizon


def prepare_dataset(panel: pd.DataFrame, ticker: str, horizon: int = 5) -> pd.DataFrame:
    grp = panel.loc[panel["ticker"] == ticker].sort_values("date").set_index("date")
    feats = build_features(grp["close"], grp["volume"])
    label = build_label_fixed_horizon(grp["close"], horizon=horizon)
    data = feats.copy()
    data["label"] = label
    return data.dropna()


def time_based_split(data: pd.DataFrame, split_date: str):
    """
    THE MOST IMPORTANT LINE IN THIS FILE. We split by DATE, not randomly.
    Everything before split_date is "train" (the model gets to see the
    answers and learn from them). Everything from split_date onward is
    "test" (the model NEVER sees these labels during fitting -- we only
    check its guesses against them afterward, once, as a final exam).
    """
    train = data.loc[:split_date]
    test = data.loc[split_date:]
    feature_cols = [c for c in data.columns if c != "label"]
    return (train[feature_cols], train["label"],
           test[feature_cols], test["label"])


def train_and_evaluate(panel: pd.DataFrame, ticker: str, split_date: str = "2022-06-01"):
    data = prepare_dataset(panel, ticker)
    X_train, y_train, X_test, y_test = time_based_split(data, split_date)

    # sklearn models generally work better when features are on comparable
    # scales (a momentum feature might range -0.5 to +0.5, while volume_surge
    # might range -1 to +10). StandardScaler rescales each feature to have
    # mean 0, std 1. CRITICAL: fit the scaler on TRAIN ONLY, then just apply
    # (transform) it to test -- fitting on test data would leak test-set
    # statistics (its mean/std) into the training process.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)   # learns mean/std from train, then rescales train
    X_test_scaled = scaler.transform(X_test)         # applies train's mean/std to test -- never refit here

    # This is the "learning" step: the model looks at every (features, label)
    # pair in the training set and finds weights that best predict the label
    # from the features.
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # predict_proba gives a probability for each class; column 1 is
    # "probability of label=1" (i.e. probability price is up in `horizon` days)
    train_prob = model.predict_proba(X_train_scaled)[:, 1]
    test_prob = model.predict_proba(X_test_scaled)[:, 1]

    train_pred = (train_prob > 0.5).astype(int)
    test_pred = (test_prob > 0.5).astype(int)

    # Baseline: what if we just always guessed the majority class in TRAIN?
    # A real model needs to beat this trivial strategy to be worth anything.
    majority_class = y_train.mode()[0]
    baseline_test_acc = (y_test == majority_class).mean()

    print(f"\n===== {ticker}: Logistic Regression, 5-day-ahead direction =====")
    print(f"Train rows: {len(y_train)}   Test rows: {len(y_test)}")
    print(f"Train accuracy:            {accuracy_score(y_train, train_pred):.1%}")
    print(f"Test accuracy:             {accuracy_score(y_test, test_pred):.1%}")
    print(f"Naive baseline accuracy:   {baseline_test_acc:.1%}  (always guess majority class '{int(majority_class)}')")
    print(f"Test AUC:                  {roc_auc_score(y_test, test_prob):.3f}  "
         f"(0.5 = no better than random, 1.0 = perfect)")

    print("\nFeature weights (higher |weight| = more influence on the prediction):")
    for name, coef in sorted(zip(X_train.columns, model.coef_[0]), key=lambda x: -abs(x[1])):
        print(f"  {name:16s} {coef:+.3f}")

    print("\nInterpretation:")
    print("  If test accuracy is barely above the naive baseline (often true here --")
    print("  this is finance, signal is weak), the model isn't 'bad', it's telling you")
    print("  the truth: 5-day direction is close to a coin flip using these simple features.")
    print("  What matters far more than accuracy is whether turning this into a real")
    print("  trading strategy survives costs -- which we test next.")

    return model, scaler, test_prob, y_test, data


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/real_ohlcv.csv"
    ticker = sys.argv[2] if len(sys.argv) > 2 else "AAPL"

    panel = pd.read_csv(path, parse_dates=["date"])
    train_and_evaluate(panel, ticker)
