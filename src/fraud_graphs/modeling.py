from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _try_import_xgboost():
    try:
        import xgboost as xgb  # type: ignore

        return xgb
    except Exception:
        return None


def asymmetric_focal_objective(
    gamma_pos: float = 1.0,
    gamma_neg: float = 4.0,
) -> Any:
    gamma_pos = float(gamma_pos)
    gamma_neg = float(gamma_neg)

    def _objective(preds: np.ndarray, dtrain: Any) -> Tuple[np.ndarray, np.ndarray]:
        y = dtrain.get_label().astype(np.float64)
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-7, 1.0 - 1e-7)
        q = 1.0 - p

        grad_pos = (q**gamma_pos) * (gamma_pos * p * np.log(p) + p - 1.0)
        d_b = p * q * (gamma_pos * np.log(p) + gamma_pos + 1.0)
        hess_pos = (q**gamma_pos) * (-gamma_pos * p * (gamma_pos * p * np.log(p) + p - 1.0) + d_b)

        grad_neg = (p**gamma_neg) * (p - gamma_neg * q * np.log(q))
        d_d = p * q * (1.0 + gamma_neg * (np.log(q) + 1.0))
        hess_neg = (p**gamma_neg) * (
            gamma_neg * q * (p - gamma_neg * q * np.log(q)) + d_d
        )

        grad = y * grad_pos + (1.0 - y) * grad_neg
        hess = y * hess_pos + (1.0 - y) * hess_neg
        hess = np.clip(hess, 1e-6, None)
        return grad, hess

    return _objective


@dataclass
class TrainResult:
    model: Any
    backend: str
    valid_pred: np.ndarray
    test_pred: np.ndarray


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    random_state: int = 42,
) -> TrainResult:
    xgb = _try_import_xgboost()
    if xgb is not None:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        dtest = xgb.DMatrix(X_test)

        params = {
            "booster": "gbtree",
            "max_depth": 8,
            "eta": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "lambda": 2.0,
            "alpha": 0.0,
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "seed": random_state,
        }
        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=600,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=50,
            verbose_eval=False,
            obj=asymmetric_focal_objective(gamma_pos=1.0, gamma_neg=4.0),
        )
        valid_pred = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
        test_pred = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
        return TrainResult(model=bst, backend="xgboost+asym_focal", valid_pred=valid_pred, test_pred=test_pred)

    pos_weight = max(1.0, ((len(y_train) - y_train.sum()) / max(1, y_train.sum())))
    sample_weight = np.where(y_train > 0, pos_weight * 1.2, 1.0).astype(np.float64)

    clf = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=8,
        max_iter=350,
        l2_regularization=1.0,
        random_state=random_state,
    )
    clf.fit(X_train, y_train, sample_weight=sample_weight)
    valid_pred = clf.predict_proba(X_valid)[:, 1]
    test_pred = clf.predict_proba(X_test)[:, 1]
    return TrainResult(model=clf, backend="sklearn_hgb_fallback", valid_pred=valid_pred, test_pred=test_pred)


def predict_scores(model: Any, backend: str, X: np.ndarray) -> np.ndarray:
    if backend.startswith("xgboost"):
        xgb = _try_import_xgboost()
        if xgb is None:
            raise RuntimeError("xgboost backend requested but xgboost is unavailable.")
        dmat = xgb.DMatrix(X)
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None or best_iter < 0:
            return model.predict(dmat)
        return model.predict(dmat, iteration_range=(0, best_iter + 1))
    return model.predict_proba(X)[:, 1]


def explain_prediction(
    model: Any,
    backend: str,
    x_row: np.ndarray,
    feature_names: List[str],
    top_k: int = 12,
) -> Dict[str, Any]:
    if backend.startswith("xgboost"):
        xgb = _try_import_xgboost()
        if xgb is None:
            raise RuntimeError("xgboost backend requested but xgboost is unavailable.")

        dmat = xgb.DMatrix(x_row.reshape(1, -1), feature_names=feature_names)
        contrib = model.predict(dmat, pred_contribs=True)[0]
        bias = float(contrib[-1])
        feat_contrib = contrib[:-1]

        order = np.argsort(np.abs(feat_contrib))[::-1]
        top_n = max(1, min(top_k, len(order)))
        selected = order[:top_n]
        remaining = order[top_n:]

        steps = [(feature_names[i], float(feat_contrib[i])) for i in selected]
        if len(remaining) > 0:
            other = float(feat_contrib[remaining].sum())
            if abs(other) > 1e-12:
                steps.append(("other_features", other))

        current_logit = bias
        waterfall = []
        for feat, delta in steps:
            before = float(_sigmoid(current_logit))
            current_logit += delta
            after = float(_sigmoid(current_logit))
            waterfall.append(
                {
                    "feature": feat,
                    "delta_logit": delta,
                    "prob_before": before,
                    "prob_after": after,
                }
            )

        final_logit = float(bias + feat_contrib.sum())
        return {
            "backend": backend,
            "base_logit": bias,
            "base_probability": float(_sigmoid(bias)),
            "final_logit": final_logit,
            "final_probability": float(_sigmoid(final_logit)),
            "waterfall": waterfall,
            "top_features": [
                {"feature": feature_names[i], "delta_logit": float(feat_contrib[i])}
                for i in selected
            ],
        }

    pred = float(model.predict_proba(x_row.reshape(1, -1))[:, 1][0])
    return {
        "backend": backend,
        "base_probability": float(pred),
        "final_probability": float(pred),
        "waterfall": [],
        "top_features": [],
        "note": "Detailed SHAP-like contributions are available when using xgboost backend.",
    }


def evaluate_scores(y_true: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    roc = float(roc_auc_score(y_true, preds))
    pr = float(average_precision_score(y_true, preds))
    precision, recall, _ = precision_recall_curve(y_true, preds)
    mask = precision >= 0.90
    recall_at_90_precision = float(recall[mask].max()) if np.any(mask) else 0.0
    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "recall_at_90_precision": recall_at_90_precision,
    }
