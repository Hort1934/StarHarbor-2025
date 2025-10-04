#!/usr/bin/env python3
"""
Скрипт для запуска API сервера StarHarbor
"""
import sys
import os
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

if __name__ == "__main__":
    import uvicorn
    
    # Устанавливаем переменные окружения если не заданы
    os.environ.setdefault("DATA_DIR", str(REPO_ROOT / "data"))
    os.environ.setdefault("MODEL_DIR", str(REPO_ROOT / "models"))
    
    print(f"Starting StarHarbor API server...")
    print(f"Repository root: {REPO_ROOT}")
    print(f"Data directory: {os.environ['DATA_DIR']}")
    print(f"Model directory: {os.environ['MODEL_DIR']}")
    print(f"API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "api.utils.main:app",  # import string instead of app object
        host="0.0.0.0",
        port=8000,
        reload=False,  # disable reload for now
        log_level="info"
    )
