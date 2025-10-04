from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from pydantic import BaseModel, Field

from api.utils.io import read_table, normalize_schema
from api.utils.constants import (
    assert_artifacts_available,
    log_artifact_paths,
    PARAMS_JSON_PATH,
)
from api.services.pipeline import (
    predict_tab,
    predict_curve,            # ONNX curve model (optional; returns None if unavailable)
    get_model_and_features,   # for SHAP/fallback
    align_features,           # for SHAP/fallback
)
from api.services.shap_utils import explain_samples
from api.services.conformal import load_tau, top1_with_confidence
from api.services.vetting import apply_qc
from api.services import curves  

from api.models.request import PredictRequest
from api.models.response import PredictResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/inference", tags=["inference"])

def _artifacts_ok():
    log_artifact_paths()
    assert_artifacts_available(["PREPROCESSOR_PATH", "FEATURE_LIST_PATH", "TAB_MODEL_PATH"])

@router.get("/health", summary="Healthcheck")
def health() -> Dict[str, Any]:
    return {"status": "ok"}

@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict from JSON rows (tabular model)",
)
def predict(req: PredictRequest, _=Depends(_artifacts_ok)):
    if not req.rows:
        raise HTTPException(400, "Empty payload: 'rows' must contain at least one row.")

    df = pd.DataFrame(req.rows)
    df = normalize_schema(df, req.mission)

    try:
        out = predict_tab(df, return_labels=req.return_labels)
    except Exception as e:
        log.exception("Inference failed: %s", e)
        raise HTTPException(500, f"Inference failed: {e}")

    return out

@router.post(
    "/predict-file",
    response_model=PredictResponse,
    summary="Predict from uploaded file (CSV/TSV/FITS; tabular model)",
)
def predict_file(
    file: UploadFile = File(...),
    mission: str = Query(None, description="kepler | k2 | tess — if raw columns file, specify mission"),
    _=Depends(_artifacts_ok),
):
    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded.")

    try:
        df = read_table(file.file.read(), suffix=Path(file.filename).suffix.lower())
        df = normalize_schema(df, mission)
        return predict_tab(df)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("File inference failed: %s", e)
        raise HTTPException(500, f"File inference failed: {e}")

@router.post(
    "/explain",
    summary="Explain first N rows via SHAP (tree models) or fallback feature importances",
)
def explain(
    req: PredictRequest,
    top_n: int = Query(1, ge=1, le=256, description="How many first rows to explain"),
    max_display: int = Query(10, ge=1, le=64, description="Top features to display per row"),
    _=Depends(_artifacts_ok),
):
    if not req.rows:
        raise HTTPException(400, "Empty payload: 'rows' must contain at least one row.")

    df = pd.DataFrame(req.rows)
    df = normalize_schema(df, req.mission)
    model, feat_names = get_model_and_features()
    X = align_features(df).head(top_n)

    try:
        out = explain_samples(model, X, feat_names, max_display=max_display)
        return out
    except Exception as e:
        log.exception("Explain failed: %s", e)
        raise HTTPException(500, f"Explain failed: {e}")

class ConformalRequest(BaseModel):
    proba: List[List[float]] = Field(..., description="Per-row class probabilities")

@router.post(
    "/conformal",
    summary="Top-1 confidence using conformal threshold tau from params.json (or default)",
)
def conformal(req: ConformalRequest):
    if not req.proba:
        raise HTTPException(400, "Empty 'proba'.")

    try:
        tau = load_tau(PARAMS_JSON_PATH)
        results = [top1_with_confidence(row, tau) for row in req.proba]
        return {"tau": tau, "results": results}
    except Exception as e:
        log.exception("Conformal failed: %s", e)
        raise HTTPException(500, f"Conformal failed: {e}")

@router.post(
    "/vet",
    summary="QC vetting flags from qc.yaml (ratio, impact, depth) + is_valid",
)
def vet(req: PredictRequest):
    if not req.rows:
        raise HTTPException(400, "Empty payload: 'rows' must contain at least one row.")

    df = pd.DataFrame(req.rows)
    df = normalize_schema(df, req.mission)

    try:
        out_df = apply_qc(df)
        flags = (
            out_df[["qc_ratio_high", "qc_impact_high", "qc_depth_low", "is_valid"]]
            .fillna(False)
            .astype(bool)
            .to_dict(orient="records")
        )
        return {"n": int(len(out_df)), "flags": flags}
    except Exception as e:
        log.exception("Vetting failed: %s", e)
        raise HTTPException(500, f"Vetting failed: {e}")

@router.post(
    "/predict-curve",
    summary="Predict from a lightcurve file (CSV/TSV/FITS) using ONNX model (if available)",
)
def predict_curve_endpoint(
    file: UploadFile = File(...),
    period_days: float | None = Query(None),
    duration_hours: float | None = Query(None),
    _=Depends(_artifacts_ok),   
):
    if not file or not file.filename:
        raise HTTPException(400, "No file uploaded.")

    try:
        lc = curves.load_lightcurve(
            file.file.read(),
            suffix=Path(file.filename).suffix.lower(),
        )
        vec = curves.prepare_curve_input(
            lc,
            period_days=period_days,
            duration_hours=duration_hours,
            fold_if_possible=True,
        )
        proba = predict_curve(vec)
        if proba is None:
            raise HTTPException(501, "Curve model is not available on this server.")
        return {"proba": [proba], "n": 1}

    except HTTPException:
        raise
    except Exception as e:
        log.exception("Predict-curve failed: %s", e)
        raise HTTPException(500, f"Predict-curve failed: {e}")
