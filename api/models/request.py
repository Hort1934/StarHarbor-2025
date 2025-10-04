from __future__ import annotations

from typing import List, Optional
from pydantic import Field

from api.models.common import (
    AppBaseModel,
    Mission,
    Row,
    ProbaPayload,  
)

class PredictRequest(AppBaseModel):
    rows: List[Row] = Field(
        default_factory=list,
        description="Array of rows: normalized columns OR raw KOI/K2/TOI."
    )
    mission: Optional[Mission] = Field(
        default=None,
        description="If rows are RAW mission columns, specify kepler|k2|tess to auto-normalize."
    )
    return_labels: bool = Field(default=True)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "rows": [{"Orbital Period [days]": 9.48, "Transit Duration [hours]": 2.96, "Transit Depth [ppm]": 500}],
                    "mission": "kepler",
                    "return_labels": True
                }
            ]
        }
    }


class ExplainRequest(PredictRequest):
    pass


class VetRequest(PredictRequest):
    pass


class ConformalRequest(ProbaPayload):
    pass

