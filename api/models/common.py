from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field, ConfigDict, field_validator

class AppBaseModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )

class Mission(str, Enum):
    KEPLER = "kepler"
    K2 = "k2"
    TESS = "tess"

class FileSuffix(str, Enum):
    CSV = ".csv"
    TSV = ".tsv"
    FITS = ".fits"
    FIT = ".fit"

Row = Dict[str, Any]

ProbaVector = List[float]
ProbaMatrix = List[ProbaVector]

ClassNames = List[str]

class WithCount(AppBaseModel):
    n: int = Field(..., ge=0, description="Number of items in the payload")

class WithServerTime(AppBaseModel):
    server_time: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"),
        description="Server timestamp (UTC, ISO 8601)",
    )

class Pagination(AppBaseModel):
    page: int = Field(1, ge=1, description="Page index (1-based)")
    page_size: int = Field(50, ge=1, le=1000, description="Page size")

class QCFlags(AppBaseModel):
    qc_ratio_high: bool = Field(False, description="Duration/Period ratio above threshold")
    qc_impact_high: bool = Field(False, description="Impact parameter above threshold")
    qc_depth_low: bool = Field(False, description="Transit depth below threshold")
    is_valid: bool = Field(True, description="Row considered valid after QC")

class ConformalTop1(AppBaseModel):
    top: int = Field(..., ge=0)
    confident: bool = Field(...)
    tau: float = Field(..., ge=0.0, le=1.0)

class ErrorDetail(AppBaseModel):
    loc: Optional[Sequence[Union[str, int]]] = Field(
        default=None, description="Location of the error, e.g. ['body','rows',0,'period_days']"
    )
    msg: str = Field(..., description="Human-readable error message")
    type: Optional[str] = Field(default=None, description="Machine-readable error type code")

class ErrorResponse(WithServerTime):
    status: str = Field(default="error")
    code: int = Field(..., description="HTTP status code")
    errors: List[ErrorDetail] = Field(
        default_factory=list,
        description="List of field-level or general errors",
    )

class ProbaPayload(AppBaseModel):
    proba: ProbaMatrix = Field(..., description="Per-row class probability vectors")

    @field_validator("proba")
    @classmethod
    def _check_proba(cls, v: ProbaMatrix) -> ProbaMatrix:
        if not v:
            return v
        cols = len(v[0])
        if cols == 0:
            raise ValueError("Probability vectors must be non-empty.")
        for i, row in enumerate(v):
            if len(row) != cols:
                raise ValueError(f"All probability vectors must have the same length (row {i}).")
            for p in row:
                if p < 0.0 or p > 1.0:
                    raise ValueError("Probabilities must be within [0,1].")
        return v

AppBaseModel.model_rebuild()
QCFlags.model_rebuild()
ConformalTop1.model_rebuild()
ErrorDetail.model_rebuild()
ErrorResponse.model_rebuild()
ProbaPayload.model_rebuild()
