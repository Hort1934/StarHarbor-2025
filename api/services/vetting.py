from __future__ import annotations
from typing import Dict
import logging
import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)

def load_qc_config(path: str = "data/schema/qc.yaml") -> Dict[str, float]:
    defaults = {
        "duration_period_max_ratio": 0.20,
        "impact_max": 1.5,
        "min_depth_ppm": 0.0,
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        defaults.update({k: float(v) for k, v in cfg.items()})
    except FileNotFoundError:
        log.info("QC config not found, using defaults.")
    return defaults

def apply_qc(df: pd.DataFrame, qc_cfg: Dict[str, float] | None = None) -> pd.DataFrame:
    qc = qc_cfg or load_qc_config()
    df = df.copy()

    period = pd.to_numeric(df.get("period_days"), errors="coerce")
    duration_h = pd.to_numeric(df.get("duration_hours"), errors="coerce")
    impact = pd.to_numeric(df.get("impact"), errors="coerce")
    depth_ppm = pd.to_numeric(df.get("depth_ppm"), errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = duration_h / (period * 24.0)

    df["qc_ratio_high"] = ratio > qc["duration_period_max_ratio"]
    df["qc_impact_high"] = impact > qc["impact_max"]
    df["qc_depth_low"] = depth_ppm < qc["min_depth_ppm"]

    df["is_valid"] = ~(df[["qc_ratio_high", "qc_impact_high", "qc_depth_low"]].fillna(False).any(axis=1))
    return df
