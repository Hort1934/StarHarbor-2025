# Data Pipeline: from raw to features

Ця схема описує, як у проекті організована папка data/ та рух даних

# Data Pipeline (таблиця етапів)

| Етап                  | Файл/Папка             | Опис |
|-----------------------|------------------------|------|
| Зовнішні дані         | CSV/FITS (KOI, TOI, K2)| Початкові дані з місій, таблиці та FITS-файли |
| Імпорт                | `data_ingest.py`       | Завантаження даних, формування копій та схем |
| Raw                   | `raw/`                 | Недоторкані дані (копії з таймштампами, parquet/CSV) |
| Schema                | `schema/`              | YAML/JSON (mapping колонок, summary, unit conversion, qc.yaml) |
| QC-звіти              | `QC-звіти/`            | Markdown, summary, статистики якості даних |
| Processed             | `processed/`           | Очищені та уніфіковані датасети, готові для фіч |
| Фічі                  | `prepare_features.py`  | Побудова ознак, імп’ютер, логі, астропоказники, one-hot mission |
| Препроцесор           | `build_preprocessor`   | Масштабування (scaler), OHE, збереження `preprocessor.pkl` |
| Спліт даних           | `split_data`           | Розділення на train/val/test (random/by_mission, group-aware) |
| Features              | `features/`            | X_train, X_val, X_test, y_train, y_val, y_test, `feature_list.json`, `target_map.json`, summary.md |

