from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException
from api.utils.constants import MODELS_DIR
from api.services.pipeline import _lazy_boot_tabular, get_model_and_features
from api.services.shap_utils import compute_global_importance
import json
import logging

log = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["metrics"])

@router.get("/summary")
def summary() -> Dict[str, Any]:
    mpath = Path(MODELS_DIR) / "metrics.json"
    if mpath.exists():
        try:
            return json.loads(mpath.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(500, f"Failed to read metrics.json: {e}")
    return {"message": "metrics.json not found", "available": False}

@router.get("/feature-importance")
def feature_importance() -> Dict[str, Any]:
    _lazy_boot_tabular()
    model, features = get_model_and_features()
    if model is None or not features:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return {"importance": compute_global_importance(model, features)}
