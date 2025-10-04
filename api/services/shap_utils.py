from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def _safe_feature_names(feature_names: List[str]) -> List[str]:
    return [str(c) for c in feature_names]

def compute_global_importance(model, feature_names: List[str]) -> Dict[str, float]:
    names = _safe_feature_names(feature_names)
    if hasattr(model, "feature_importances_"):
        imp = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        if imp.ndim == 1 and imp.size == len(names):
            s = imp / (imp.sum() + 1e-12)
            return {n: float(v) for n, v in sorted(zip(names, s), key=lambda x: x[1], reverse=True)}
    return {n: 0.0 for n in names}

def explain_samples(
    model,
    X: np.ndarray | pd.DataFrame,
    feature_names: List[str],
    max_display: int = 10,
) -> Dict[str, object]:
    names = _safe_feature_names(feature_names)
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

    out = {
        "global_importance": compute_global_importance(model, names),
        "samples": [],
    }

    try:
        import shap  # type: ignore
        # TreeExplainer is fast for tree models
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model)

        shap_values = explainer(X_arr)
        vals = getattr(shap_values, "values", None)
        if vals is None:
            raise RuntimeError("No SHAP values produced")

        vals = np.asarray(vals)
        if vals.ndim == 3:
            vals = vals.mean(axis=2)

        for i in range(min(len(X_arr), 256)):  
            row_vals = vals[i]
            idx = np.argsort(-np.abs(row_vals))[:max_display]
            tops = []
            for j in idx:
                tops.append({
                    "feature": names[j],
                    "value": float(X_arr[i, j]),
                    "contribution": float(row_vals[j]),
                })
            out["samples"].append({"top": tops})

        mean_abs = np.mean(np.abs(vals), axis=0)
        gi = {n: float(v) for n, v in sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)}
        out["global_importance"] = gi
        return out

    except Exception as e:
        log.info("SHAP not available or failed (%s). Falling back to feature_importances_.", e)
        return out
