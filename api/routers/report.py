from __future__ import annotations
from typing import Any, Dict, List
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import csv
import io

router = APIRouter(prefix="/report", tags=["report"])

@router.post("/export-csv")
def export_csv(rows: List[Dict[str, Any]]):
    if not rows:
        return StreamingResponse(io.BytesIO(b""), media_type="text/csv")
    buf = io.StringIO()
    fieldnames = sorted({k for r in rows for k in r.keys()})
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fieldnames})
    data = io.BytesIO(buf.getvalue().encode("utf-8"))
    return StreamingResponse(data, media_type="text/csv", headers={
        "Content-Disposition": 'attachment; filename="predictions.csv"'
    })
