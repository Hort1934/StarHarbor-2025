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
_FEATURES: List[str] = []            
_TARGET_MAP: Optional[List[str]] = None
_TAB_MODEL = None                    
_CNN_SESSION = None                  
_SCALER = None                      
_FUSE = None                         
_PARAMS: dict = {}                   

def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _lazy_boot_tabular():
    global _TAB_MODEL, _FEATURES, _PREPROCESSOR, _TARGET_MAP
    if _TAB_MODEL is not None and _FEATURES:
        return

    from api.utils.constants import (
        TAB_MODEL_PATH,
        PREPROCESSOR_PATH,
        FEATURE_LIST_PATH,
        TARGET_MAP_PATH,
    )
    import json, joblib

    if _PREPROCESSOR is None:
        _PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)

    if not _FEATURES:
        with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
            _FEATURES = json.load(f)

    if _TARGET_MAP is None and TARGET_MAP_PATH.exists():
        try:
            with open(TARGET_MAP_PATH, "r", encoding="utf-8") as f:
                target_dict = json.load(f)
                # Convert to list format: index -> class name
                _TARGET_MAP = [""] * len(target_dict)
                for class_name, idx in target_dict.items():
                    _TARGET_MAP[idx] = class_name
        except Exception as e:
            log.warning("Failed to load target map: %s", e)
            _TARGET_MAP = ["fp", "candidate", "confirmed"]  # fallback

    if _TAB_MODEL is None:
        _TAB_MODEL = joblib.load(TAB_MODEL_PATH)


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
    
    # Add missing feature columns filled with NaN
    for col in _FEATURES:
        if col not in X.columns:
            X[col] = np.nan
    
    # Compute derived features if possible
    try:
        # Log features
        if "period_days" in X.columns and "log_period" in _FEATURES:
            X["log_period"] = np.log10(X["period_days"].clip(lower=0.1))
            
        if "duration_hours" in X.columns and "log_duration" in _FEATURES:
            X["log_duration"] = np.log10(X["duration_hours"].clip(lower=0.1))
            
        if "stellar_teff_k" in X.columns and "log_teff" in _FEATURES:
            X["log_teff"] = np.log10(X["stellar_teff_k"].clip(lower=1000))
        
        # Period-based ratios
        if "duration_hours" in X.columns and "period_days" in X.columns and "dur_over_p13" in _FEATURES:
            period_hours = X["period_days"] * 24
            X["dur_over_p13"] = X["duration_hours"] / (period_hours ** (1/3))
            
        # Duration ratio (duration / period)
        if "duration_hours" in X.columns and "period_days" in X.columns and "duration_ratio" in _FEATURES:
            X["duration_ratio"] = X["duration_hours"] / (X["period_days"] * 24)
        
        # Planet radius estimates
        if "depth_ppm" in X.columns and "stellar_radius_rsun" in X.columns:
            # k = sqrt(depth_ppm / 1e6)
            if "k_est" in _FEATURES:
                X["k_est"] = np.sqrt(X["depth_ppm"] / 1e6)
            
            # rp = k * R_star (in Earth radii)
            if "rp_est_rearth" in _FEATURES:
                k_val = np.sqrt(X["depth_ppm"] / 1e6)
                r_star_rearth = X["stellar_radius_rsun"] * 109.16  # 1 R_sun ≈ 109.16 R_earth
                X["rp_est_rearth"] = k_val * r_star_rearth
        
        # Depth over stellar radius
        if "depth_ppm" in X.columns and "stellar_radius_rsun" in X.columns and "depth_over_rstar" in _FEATURES:
            X["depth_over_rstar"] = X["depth_ppm"] / (X["stellar_radius_rsun"] * 1e6)
        
        # Compare k vs measured radius
        if "k_est" in X.columns and "rp_rearth" in X.columns and "k_vs_rp" in _FEATURES:
            X["k_vs_rp"] = X["k_est"] / X["rp_rearth"].clip(lower=0.1)
        
        # Relative insolation
        if "insolation_earth" in X.columns and "insolation_rel_earth" in _FEATURES:
            X["insolation_rel_earth"] = np.log10(X["insolation_earth"].clip(lower=0.01))
            
        # Round period for binning
        if "period_days" in X.columns and "period_rounded" in _FEATURES:
            X["period_rounded"] = np.round(X["period_days"], 1)
            
    except Exception as e:
        print(f"Warning: Error computing derived features: {e}")
    
    # Ensure all feature columns are numeric, convert strings to NaN
    feature_df = X[_FEATURES].copy()
    for col in feature_df.columns:
        if feature_df[col].dtype == 'object':
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
    
    return feature_df

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
    return _TAB_MODEL, list(_FEATURES)

def align_features(df: pd.DataFrame) -> pd.DataFrame:
    return _align_feature_frame(df)
