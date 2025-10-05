# Data Pipeline: from raw to features

Ця схема описує, як у проекті організована папка data/ та рух даних

               ┌───────────────────┐
               │   Зовнішні дані   │
               │ (KOI, TOI, K2 CSV/|
               │     FITS и др.)   |
               └─────────┬─────────┘
                         │
                         ▼
                data_ingest.py
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
       raw/           schema/           QC-звіти
  (недоторкані     (YAML/JSON:          (Markdown,
  копії з          mapping колонок,     summary,
  таймштампами,    unit conversion,     stats)
  parquet/CSV)     qc.yaml і ін.)
         │
         ▼
    processed/
  (очищені та уніфіковані 
   датасети, готові до фіч)

                         │
                         ▼
                prepare_features.py
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
   engineer_features   build_preprocessor  split_data
 (створення нових       (импьютер,         (train/val/test
  ознак, logи,          scaler, OHE,       random/by_mission,
  астропоказники,       збереження         group-aware)
  one-hot mission)      preprocessor.pkl)

                         │
                         ▼
                      features/
    (X_train, X_val, X_test, y_train, y_val, y_test,
     feature_list.json, target_map.json, summary.md)
