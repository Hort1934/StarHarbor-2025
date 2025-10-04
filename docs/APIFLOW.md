# Взаємодія api 
   
              ┌─────────────────────────────┐
              │           Client            │
              │  (UI/Frontend, cURL, etc.)  │
              └──────────────┬──────────────┘
                             │  HTTP
                            down
                    api/utils/main.py
                   (FastAPI app startup,
                 include_router(...), CORS)
                             │
               ┌─────────────┼─────────────────┐
               │             │                 │
             down           down              down
        routers/inference   routers/files   routers/metrics,report
         /health            (опц.)           (опц.)
         /predict
         /predict-file
         /explain
         /conformal
         /vet
         /predict-curve (опц.)

                             │
                            down
                      api/services/*
   ┌─────────────────────────────────────────────────────────────┐
   │ pipeline.py                                                 │
   │  - predict_tab(df) -> lazy-load:                            │
   │     PREPROCESSOR_PATH, FEATURE_LIST_PATH, TAB_MODEL_PATH    │
   │     transform -> model.predict_proba -> {proba, classes, n} │
   │  - get_model_and_features(), align_features() (для SHAP)    │
   │  - predict_curve(vec) (ONNX, если доступен)                 │
   │  - predict_fused(df, vec) (опц.)                            │
   └─────────────────────────────────────────────────────────────┘
            │              │                    │
            │              │                    │
            │              │                    ├───────────┐
            │              │                                │
            │            down                               down
            │      shap_utils.py                      conformal.py
            │      explain_samples(model, X,          load_tau(params.json),
            │      feature_names) -> SHAP для         top1_with_confidence
            │      деревьев, fallback на              (порог τ)
            │      feature_importances_               ────────────────> /conformal
            │              up
            │              │
            │         curves.py (опц.)
            │         load_lightcurve/prepare_curve_input
            │         (fold, detrend, resample) -> predict_curve / fused
            │
      vetting.py
      apply_qc(df) -> флаги ratio/impact/depth + is_valid
      (порогa беруться з data/schema/qc.yaml) ─────────────> /vet

                             │
                            down
                       api/utils/*
     ┌─────────────────────────────────────────────────────────────┐
     │ constants.py:                                               │
     │   шляхи к артефактам:                                       │
     │   PREPROCESSOR_PATH, FEATURE_LIST_PATH, TARGET_MAP_PATH,    │
     │   TAB_MODEL_PATH, FUSE_MODEL_PATH, SCALER_PATH,             │
     │   CNN_ONNX_PATH, PARAMS_JSON_PATH                           │
     │   + assert_artifacts_available(), log_artifact_paths()      │
     │                                                             │
     │ io.py:                                                      │
     │   read_table(path|bytes, suffix) -> DataFrame               │
     │   normalize_schema(df, mission) -> KOI/K2/TOI -> canonical  │
     └─────────────────────────────────────────────────────────────┘

                             │
                            down
                         Artifacts / Models
   models/{preprocessor.pkl, feature_list.json, target_map.json,
           tab_xgb.pkl (або інша таблична), cnn.onnx (опц.),
           scaler.bin (опц.), fuse.joblib (опц.), params.json}

                             │
                            down
                      JSON HTTP Response
      { proba: [[...]], classes: [...], n, (опц.) explain, vet flags, conformal }
