from __future__ import annotations

from typing import List, Optional
from pydantic import Field

from api.models.common import (
    AppBaseModel,
    WithCount,
    WithServerTime,
    ProbaMatrix,
    ClassNames,
    QCFlags,
    ConformalTop1,
)

class PredictResponse(WithCount, WithServerTime):
    proba: ProbaMatrix = Field(..., description="Per-row class probability vectors")
    classes: Optional[ClassNames] = Field(default=None, description="Class names in proba order")

class VetResponse(WithCount, WithServerTime):
    flags: List[QCFlags] = Field(..., description="QC flags per row")


class ConformalResponse(WithServerTime):
    tau: float = Field(..., ge=0.0, le=1.0)
    results: List[ConformalTop1]

