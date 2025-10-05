import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    import uvicorn

    os.environ.setdefault("DATA_DIR", str(REPO_ROOT / "data"))
    os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "models"))
    
    print(f"Starting StarHarbor API server...")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Data directory: {os.environ['DATA_DIR']}")
    print(f"Model directory: {os.environ['MODEL_DIR']}")
    print(f"API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api.utils.main:app",  
        host="0.0.0.0",
        port=8000,
        reload=False,  
        log_level="info"
    )