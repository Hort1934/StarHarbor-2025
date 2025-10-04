from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Union
import json
import logging
import joblib
import numpy as np
import pandas as pd

from api.utils.constants import (
    PREPROCESSOR_PATH,
    FEATURE_LIST_PATH,
    TARGET_MAP_PATH,
    TAB_MODEL_PATH,
    FUSE_MODEL_PATH,
    SCALER_PATH,
    CNN_ONNX_PATH,
    PARAMS_JSON_PATH,
    assert_artifacts_available,
    log_artifact_paths,
)

log = logging.getLogger(__name__)

_PREPROCESSOR = None
_FEATURES: list[str] = []
_TAB_MODEL = None
_TARGET_MAP = None                  
_CNN_SESSION = None                  
_SCALER = None                      
_FUSE = None                         
_PARAMS: dict = {}                   

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
def _lazy_boot_tabular() -> None:
    global _PREPROCESSOR, _FEATURES, _TAB_MODEL, _TARGET_MAP

    if _TAB_MODEL is not None and _FEATURES:
        return

    from api.utils.constants import (
        TAB_MODEL_PATH,
        PREPROCESSOR_PATH,
        FEATURE_LIST_PATH,
        TARGET_MAP_PATH,   
    )
    log_artifact_paths()
    assert_artifacts_available(["PREPROCESSOR_PATH", "FEATURE_LIST_PATH", "TAB_MODEL_PATH"])

    log.info("Loading preprocessor: %s", PREPROCESSOR_PATH)
    _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

    log.info("Loading feature list: %s", FEATURE_LIST_PATH)
    _FEATURES = json.loads(Path(FEATURE_LIST_PATH).read_text(encoding="utf-8"))
    if not isinstance(_FEATURES, list) or not _FEATURES:
        raise ValueError("feature_list.json is empty or invalid")

    log.info("Loading tabular model: %s", TAB_MODEL_PATH)
    _TAB_MODEL = joblib.load(TAB_MODEL_PATH)
    
    if 'TARGET_MAP_PATH' in locals() and TARGET_MAP_PATH and Path(TARGET_MAP_PATH).exists():
        try:
            _TARGET_MAP = json.loads(Path(TARGET_MAP_PATH).read_text(encoding="utf-8"))
            log.info("Loaded target map with %d classes", len(_TARGET_MAP))
        except Exception:
            log.warning("Failed to read TARGET_MAP_PATH %s", TARGET_MAP_PATH)

def _lazy_boot_curve() -> None:
    global _CNN_SESSION, _SCALER, _FUSE, _PARAMS

    # onnx session
    if _CNN_SESSION is None:
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as e:
            log.info("onnxruntime not available, curve model disabled: %s", e)
            return

        if CNN_ONNX_PATH.exists():
            log.info("Loading ONNX model: %s", CNN_ONNX_PATH)
            _CNN_SESSION = ort.InferenceSession(
                str(CNN_ONNX_PATH),
                providers=["CPUExecutionProvider"],
            )
        else:
            log.info("CNN_ONNX_PATH not found: %s", CNN_ONNX_PATH)

    # scaler
    if _SCALER is None and SCALER_PATH.exists():
        try:
            log.info("Loading curve scaler: %s", SCALER_PATH)
            _SCALER = joblib.load(SCALER_PATH)
        except Exception as e:
            log.warning("Failed to load scaler %s: %s", SCALER_PATH, e)

    # params
    if not _PARAMS and PARAMS_JSON_PATH.exists():
        _PARAMS = _load_json(PARAMS_JSON_PATH)

    # fuse
    if _FUSE is None and FUSE_MODEL_PATH.exists():
        try:
            log.info("Loading fuse model: %s", FUSE_MODEL_PATH)
            _FUSE = joblib.load(FUSE_MODEL_PATH)
        except Exception as e:
            log.warning("Failed to load fuse model: %s", e)


def _align_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    _lazy_boot_tabular()

    X = df.copy()
    for col in _FEATURES:
        if col not in X.columns:
            X[col] = np.nan
    return X[_FEATURES]

def predict_tab(df_norm: pd.DataFrame, *, return_labels: bool = True) -> dict:
    _lazy_boot_tabular()

    if df_norm.empty:
        return {"proba": [], "classes": _TARGET_MAP if return_labels else None, "n": 0}

    X = _align_feature_frame(df_norm)

    # transform
    try:
        X_tr = _PREPROCESSOR.transform(X)
    except AttributeError:
        X_tr = _PREPROCESSOR.fit_transform(X)

    # predict
    if hasattr(_TAB_MODEL, "predict_proba"):
        proba = _TAB_MODEL.predict_proba(X_tr)
    elif hasattr(_TAB_MODEL, "predict"):
        pred = _TAB_MODEL.predict(X_tr)
        proba = np.vstack([1 - pred, pred]).T if pred.ndim == 1 else pred
    else:
        raise RuntimeError("Tabular model does not support predict(_proba)")

    out = {
        "proba": proba.tolist(),
        "classes": _TARGET_MAP if (return_labels and _TARGET_MAP) else None,
        "n": int(len(df_norm)),
    }

    # ---------------- (Variant B) ----------------
    # qc_df = apply_qc(df_norm)
    # out["qc_flags"] = (
    #     qc_df[["qc_ratio_high", "qc_impact_high", "qc_depth_low", "is_valid"]]
    #     .fillna(False)
    #     .astype(bool)
    #     .values
    #     .tolist()
    # )
    # tau = load_tau(PARAMS_JSON_PATH)
    # out["conformal"] = [top1_with_confidence(row.tolist(), tau) for row in proba]
    # ------------------------------------------------------

    return out


def predict_curve(lightcurve: Union[List[float], np.ndarray]) -> Optional[List[float]]:
    try:
        _lazy_boot_curve()
    except ImportError:
        log.info("predict_curve: onnxruntime not available, skipping.")
        return None

    if _CNN_SESSION is None:
        log.info("predict_curve: CNN session not initialized, returning None.")
        return None

    x = np.asarray(lightcurve, dtype=np.float32).reshape(-1)

    if _SCALER is not None:
        x2 = _SCALER.transform(x.reshape(1, -1)).astype(np.float32)
    else:
        if np.all(np.isfinite(x)) and (x.max() - x.min()) > 0:
            x2 = ((x - x.min()) / (x.max() - x.min())).reshape(1, -1).astype(np.float32)
        else:
            x2 = x.reshape(1, -1)

    inp_name = _CNN_SESSION.get_inputs()[0].name
    shape = _CNN_SESSION.get_inputs()[0].shape
    if len(shape) == 3 and shape[1] == 1:      # (N, C, L)
        inp = x2.reshape(1, 1, -1)
    elif len(shape) == 3 and shape[2] == 1:    # (N, L, C)
        inp = x2.reshape(1, -1, 1)
    else:
        inp = x2

    outputs = _CNN_SESSION.run(None, {inp_name: inp})
    proba = outputs[0]
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)
    return proba[0].astype(float).tolist()


def predict_fused(
    df_norm: pd.DataFrame,
    lightcurve: Optional[Union[List[float], np.ndarray]] = None,
    *,
    alpha: Optional[float] = None,
) -> dict:
    tab = predict_tab(df_norm)

    curve_proba = None
    if lightcurve is not None:
        curve_proba = predict_curve(lightcurve)

    if not curve_proba:
        return tab

    tab_vec = np.asarray(tab["proba"][0], dtype=float)
    cur_vec = np.asarray(curve_proba, dtype=float)

    if _FUSE is not None:
        try:
            fused = _FUSE.predict_proba(np.c_[tab_vec, cur_vec].reshape(1, -1))[0]
        except Exception as e:
            log.warning("Fuse model failed, fallback to weighted sum: %s", e)
            fused = None
    else:
        fused = None

    if fused is None:
        w = alpha if alpha is not None else float(_PARAMS.get("fuse_weight_tab", 0.5))
        fused = w * tab_vec + (1.0 - w) * cur_vec

    fused = fused / (fused.sum() + 1e-12)

    return {
        "proba": [fused.tolist()],
        "classes": tab.get("classes"),
        "n": 1,
        "parts": {
            "tab": tab["proba"][0],
            "curve": curve_proba,
            "alpha": float(alpha if alpha is not None else _PARAMS.get("fuse_weight_tab", 0.5)),
        },
    }

def get_model_and_features():
    _lazy_boot_tabular()
    return _TAB_MODEL, _FEATURES

def align_features(df: pd.DataFrame) -> pd.DataFrame:
    return _align_feature_frame(df)



