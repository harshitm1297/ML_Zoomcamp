"""Event-based model selection and evaluation for the midterm project."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from features import model_input


RANDOM_STATE = 42
PARAMETERS = {
    "dummy": [{}],
    "logistic_regression": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "decision_tree": [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth in (2, 4, 6)
        for leaf in (10, 25)
    ],
    "random_forest": [
        {"max_depth": depth, "min_samples_leaf": leaf}
        for depth in (4, 8, None)
        for leaf in (5, 15)
    ],
}


def split_by_event(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(outer.split(frame, groups=frame.event_id))
    train_val = frame.iloc[train_val_idx].copy()
    test = frame.iloc[test_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, val_idx = next(inner.split(train_val, groups=train_val.event_id))
    train = train_val.iloc[train_idx].copy()
    val = train_val.iloc[val_idx].copy()
    assert not (set(train.event_id) & set(val.event_id))
    assert not (set(train.event_id) & set(test.event_id))
    assert not (set(val.event_id) & set(test.event_id))
    return train, val, test


def make_model(name: str, params: dict):
    preprocessing = [SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)]
    if name == "dummy":
        classifier = DummyClassifier(strategy="prior")
    elif name == "logistic_regression":
        preprocessing.append(StandardScaler())
        classifier = LogisticRegression(
            C=params["C"], max_iter=2000, solver="liblinear", random_state=RANDOM_STATE
        )
    elif name == "decision_tree":
        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    elif name == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE, **params
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return make_pipeline(*preprocessing, classifier)


def score(y_true, probability, threshold: float = 0.5) -> dict:
    predicted = probability >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def choose_threshold(y_true, probability) -> float:
    """Choose a validation threshold with the highest F1; ties prefer higher precision."""
    candidates = np.arange(0.1, 0.91, 0.05)
    best = None
    for threshold in candidates:
        s = score(y_true, probability, float(threshold))
        denominator = s["precision"] + s["recall"]
        f1 = 2 * s["precision"] * s["recall"] / denominator if denominator else 0.0
        candidate = (f1, s["precision"], float(threshold))
        if best is None or candidate > best:
            best = candidate
    return best[2]


def out_of_fold_probability(frame: pd.DataFrame, name: str, params: dict) -> np.ndarray:
    """Predict each development event only from models fit on other events."""
    x, y = model_input(frame), frame.mutual_interest
    probability = np.full(len(frame), np.nan)
    folds = GroupKFold(n_splits=5)
    for train_idx, val_idx in folds.split(x, y, groups=frame.event_id):
        model = make_model(name, params)
        model.fit(x.iloc[train_idx], y.iloc[train_idx])
        probability[val_idx] = model.predict_proba(x.iloc[val_idx])[:, 1]
    if np.isnan(probability).any():
        raise RuntimeError("Some development rows have no out-of-fold prediction")
    return probability


def compare_models(development: pd.DataFrame) -> pd.DataFrame:
    y = development.mutual_interest
    rows = []
    for name, variants in PARAMETERS.items():
        for params in variants:
            probability = out_of_fold_probability(development, name, params)
            rows.append({"model": name, "parameters": params, **score(y, probability)})
    return pd.DataFrame(rows).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    ).reset_index(drop=True)


def split_summary(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> dict:
    result = {}
    for name, part in (("train", train), ("validation", val), ("test", test)):
        result[name] = {
            "rows": len(part), "events": int(part.event_id.nunique()),
            "positives": int(part.mutual_interest.sum()),
            "positive_rate": float(part.mutual_interest.mean()),
        }
    return result
