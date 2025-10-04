# API Flow (таблиця етапів)

| Етап / Рівень           | Файл/Папка                 | Опис |
|-------------------------|----------------------------|------|
| Клієнт                  | Client (UI/Frontend, cURL) | Виклики HTTP до API |
| Старт FastAPI           | `api/utils/main.py`        | Запуск FastAPI app, `include_router(...)`, CORS |
| Роутери                 | `routers/`                 | `inference`: /health, /predict, /predict-file, /explain, /conformal, /vet, (опц. /predict-curve)<br>`files` (опц.)<br>`metrics, report` (опц.) |
| Сервіси (ядро)          | `api/services/`            | `pipeline.py`: lazy-load моделей, `predict_tab`, `predict_curve`, `predict_fused`, `align_features()` (для SHAP) |
| Пояснення (Explain)     | `shap_utils.py`            | SHAP для дерев, fallback на feature_importances |
| Криві (Curve)           | `curves.py` (опц.)         | `load_lightcurve`, detrend, resample → `predict_curve` |
| Конформальний аналіз    | `conformal.py`             | `load_tau(params.json)`, `top1_with_confidence(τ)` → /conformal |
| Веттинг (QC)            | `vetting.py`               | `apply_qc(df)` → ratio/impact/depth, прапорці з `data/schema/qc.yaml` → /vet |
| Utils                   | `api/utils/`               | `constants.py`: шляхи артефактів, `assert_artifacts_available()`<br>`io.py`: читання таблиць, `normalize_schema(df, mission)` |
| Артефакти/Моделі        | `models/`                  | `preprocessor.pkl`, `feature_list.json`, `target_map.json`, `tab_xgb.pkl`, `cnn.onnx` (опц.), `scaler.bin` (опц.), `fuse.joblib` (опц.), `params.json` |
| Відповідь               | JSON Response              | `{ proba, classes, n, (опц.) explain, vet flags, conformal }` |

