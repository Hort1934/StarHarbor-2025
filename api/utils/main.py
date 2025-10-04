from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.utils.constants import log_artifact_paths, assert_artifacts_available
from api.routers import inference

import logging

log = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    log_artifact_paths()
    try:
        assert_artifacts_available(["PREPROCESSOR_PATH", "FEATURE_LIST_PATH", "TAB_MODEL_PATH"])
        log.info("Artifacts check: OK")
    except Exception as e:
        log.warning("Artifacts check failed: %s", e)
    yield
    # ── shutdown (cleanup) ────────────────────────────────

app = FastAPI(
    title="Exoplanet Vetting API",
    version="0.1.0",
    description="ML inference service for KOI/K2/TOI-based exoplanet identification",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,  
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(inference.router)

from api.routers import files, metrics, report
app.include_router(files.router)
app.include_router(metrics.router)
app.include_router(report.router)

# option B
def _try_include(module: str):
    try:
        mod = __import__(f"api.routers.{module}", fromlist=["router"])
        app.include_router(mod.router)   # type: ignore[attr-defined]
        log.info("Router mounted: %s", module)
    except Exception as e:
        log.info("Router %s not mounted (%s)", module, e)

# _try_include("files")
# _try_include("metrics")
# _try_include("report")

@app.get("/", tags=["health"])
def index():
    return {"status": "ok", "service": "exoplanet-vetting", "docs": "/docs"}

@app.get("/ping", tags=["health"])
def ping():
    return {"pong": True}
