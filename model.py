"""ML risk model: train, evaluate, score, and explain a single applicant.

Everything produced by training (model, baseline, metrics, held-out scored
rows) is saved to one file, so the app can show results immediately on the next
start without re-uploading or retraining.
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from data import FEATURES, TARGET

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "risk_model.joblib")


def risk_band(p: float) -> str:
    if p < 0.05:
        return "LOW"
    if p < 0.15:
        return "MODERATE"
    if p < 0.35:
        return "HIGH"
    return "VERY HIGH"


def train(df: pd.DataFrame, save: bool = True):
    source = df.attrs.get("source", "unknown")
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    # Histogram-based gradient boosting: same idea, but trains in seconds on 150k rows
    model = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.06, max_depth=4, l2_regularization=1.0, random_state=42
    )
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "auc": float(roc_auc_score(y_te, proba)),
        "default_rate": float(y.mean()),
        "report": classification_report(y_te, proba > 0.5, zero_division=0),
        "source": source,
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
    # importance on a sample of the hold-out set keeps this step to a few seconds
    idx = X_te.sample(min(8000, len(X_te)), random_state=42).index
    imp = permutation_importance(model, X_te.loc[idx], y_te.loc[idx], scoring="roc_auc",
                                 n_repeats=3, random_state=42)
    metrics["importance"] = dict(
        sorted(zip(FEATURES, imp.importances_mean), key=lambda kv: -kv[1])
    )
    baseline = X_tr.median().to_dict()

    # Held-out rows the model never saw, scored: honest "previous data" results.
    test_scored = X_te.copy()
    test_scored["actual_default"] = y_te.values
    test_scored["default_probability"] = proba
    test_scored["risk_band"] = [risk_band(p) for p in proba]
    test_scored = test_scored.sort_values("default_probability", ascending=False)

    if save:
        joblib.dump({"model": model, "baseline": baseline, "metrics": metrics,
                     "test_scored": test_scored}, MODEL_PATH)
    return model, baseline, metrics


def load_artifacts() -> dict | None:
    """Everything saved by train(), or None if nothing (usable) is saved yet."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        obj = joblib.load(MODEL_PATH)
    except Exception:
        # Saved with a different Python / scikit-learn version (e.g. on a hosted
        # server): ignore it, the app will retrain with the local versions.
        return None
    if not isinstance(obj, dict) or "metrics" not in obj or "test_scored" not in obj:
        return None  # older save format
    return obj


def load_model():
    obj = joblib.load(MODEL_PATH)
    return obj["model"], obj["baseline"]


def score(model, baseline: dict, applicant: dict) -> dict:
    """Return default probability, band, and per-feature local contributions.

    Contribution of a feature = drop in predicted risk when that feature is
    replaced by the population median (a simple, model-agnostic local
    explanation; no extra libraries needed).
    """
    row = pd.DataFrame([applicant])[FEATURES]
    p = float(model.predict_proba(row)[0, 1])
    contribs = {}
    for f in FEATURES:
        alt = row.copy()
        alt[f] = baseline[f]
        contribs[f] = p - float(model.predict_proba(alt)[0, 1])
    drivers = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))[:4]
    return {
        "default_probability": round(p, 4),
        "risk_band": risk_band(p),
        "top_drivers": [
            {"feature": f, "value": applicant[f], "impact": round(c, 4),
             "direction": "raises risk" if c > 0 else "lowers risk"}
            for f, c in drivers
        ],
    }