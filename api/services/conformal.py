from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

DEFAULT_TAU = 0.6  # default decision threshold

def load_tau(params_json: Path | None) -> float:
    if params_json and params_json.exists():
        try:
            data = json.loads(params_json.read_text(encoding="utf-8"))
            return float(data.get("conformal_tau", DEFAULT_TAU))
        except Exception:
            return DEFAULT_TAU
    return DEFAULT_TAU

def predict_set(proba: List[float], tau: float) -> List[int]:
    return [i for i, p in enumerate(proba) if float(p) >= tau]

def top1_with_confidence(proba: List[float], tau: float) -> Dict[str, object]:
    if not proba:
        return {"top": None, "confident": False, "tau": tau}
    top = int(max(range(len(proba)), key=lambda i: proba[i]))
    return {"top": top, "confident": float(proba[top]) >= tau, "tau": tau}
