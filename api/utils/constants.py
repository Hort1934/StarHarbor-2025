from __future__ import annotations
from pathlib import Path
from typing import Iterable
import os
import logging

log = logging.getLogger(__name__)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = Path(os.getenv("DATA_DIR", REPO_ROOT / "data")).resolve()
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", REPO_ROOT / "models")).resolve()

PROCESSED_DIR: Path = Path(os.getenv("PROCESSED_DIR", DATA_DIR / "processed")).resolve()
FEATURES_DIR: Path = Path(os.getenv("FEATURES_DIR", DATA_DIR / "features")).resolve()

PREPROCESSOR_PATH: Path = MODELS_DIR / "preprocessor.pkl"
FEATURE_LIST_PATH: Path = MODELS_DIR / "feature_list.json"
TARGET_MAP_PATH: Path = Path(os.getenv("TARGET_MAP_PATH", MODELS_DIR / "target_map.json")).resolve()

TAB_MODEL_PATH: Path = MODELS_DIR / "tab_xgb.pkl"
FUSE_MODEL_PATH: Path = Path(os.getenv("FUSE_MODEL_PATH", MODELS_DIR / "fuse.joblib")).resolve()
SCALER_PATH: Path = Path(os.getenv("SCALER_PATH", MODELS_DIR / "scaler.bin")).resolve()

CNN_ONNX_PATH: Path = Path(os.getenv("CNN_ONNX_PATH", MODELS_DIR / "cnn.onnx")).resolve()
PARAMS_JSON_PATH: Path = Path(os.getenv("PARAMS_JSON_PATH", MODELS_DIR / "params.json")).resolve()

def assert_artifacts_available(required: Iterable[str] | None = None) -> None:
    names = list(required) if required is not None else [
        "PREPROCESSOR_PATH",
        "FEATURE_LIST_PATH",
        "TAB_MODEL_PATH",
    ]
    missing = []
    for name in names:
        p = globals().get(name)
        if not isinstance(p, Path):
            missing.append(f"{name} (not a Path)")
            continue
        if not p.exists():
            missing.append(f"{name} -> {p}")
    if missing:
        details = "\n".join(f"  - {m}" for m in missing)
        raise FileNotFoundError(
            "Required model artifacts are missing:\n"
            f"{details}\n"
            "Hint: set ENV vars to override paths, e.g. PREPROCESSOR_PATH=/app/models/preprocessor.pkl"
        )

def log_artifact_paths() -> None:
    paths = {
        "REPO_ROOT": REPO_ROOT,
        "DATA_DIR": DATA_DIR,
        "MODELS_DIR": MODELS_DIR,
        "PROCESSED_DIR": PROCESSED_DIR,
        "FEATURES_DIR": FEATURES_DIR,
        "PREPROCESSOR_PATH": PREPROCESSOR_PATH,
        "FEATURE_LIST_PATH": FEATURE_LIST_PATH,
        "TARGET_MAP_PATH": TARGET_MAP_PATH,
        "TAB_MODEL_PATH": TAB_MODEL_PATH,
        "FUSE_MODEL_PATH": FUSE_MODEL_PATH,
        "SCALER_PATH": SCALER_PATH,
        "CNN_ONNX_PATH": CNN_ONNX_PATH,
        "PARAMS_JSON_PATH": PARAMS_JSON_PATH,
    }
    for k, v in paths.items():
        log.info("%s = %s", k, v)

__all__ = [
    "REPO_ROOT",
    "DATA_DIR", "MODELS_DIR", "PROCESSED_DIR", "FEATURES_DIR",
    "PREPROCESSOR_PATH", "FEATURE_LIST_PATH", "TARGET_MAP_PATH",
    "TAB_MODEL_PATH", "FUSE_MODEL_PATH", "SCALER_PATH",
    "CNN_ONNX_PATH", "PARAMS_JSON_PATH",
    "assert_artifacts_available", "log_artifact_paths",
]
