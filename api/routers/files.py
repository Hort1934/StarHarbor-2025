from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

SAMPLES_DIR = Path("data/samples").resolve()

router = APIRouter(prefix="/files", tags=["files"])

@router.get("/samples")
def list_samples() -> Dict[str, Any]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    files = [p.name for p in SAMPLES_DIR.iterdir() if p.is_file()]
    return {"count": len(files), "files": files}

@router.get("/samples/{name}")
def get_sample(name: str):
    p = SAMPLES_DIR / name
    if not p.exists():
        raise HTTPException(404, "Sample not found")
    return FileResponse(str(p))
