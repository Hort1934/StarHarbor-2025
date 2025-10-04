#мінімум, щоб не впало
from .inference import router as inference_router
from .files import router as files_router
from .metrics import router as metrics_router
from .report import router as report_router

__all__ = [
    "inference_router",
    "files_router",
    "metrics_router",
    "report_router",
]
