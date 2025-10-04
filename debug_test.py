#!/usr/bin/env python3
"""
Тестовый скрипт для отладки StarHarbor API
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Добавляем проект в путь
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

def test_data_loading():
    """Тестирование загрузки данных"""
    print("=== Тест загрузки данных ===")
    
    from api.utils.io import read_table, normalize_schema
    
    # Загружаем тестовый файл
    df = read_table("test_data.csv")
    print(f"Загружено строк: {len(df)}")
    print(f"Столбцы: {list(df.columns)}")
    print("Первая строка:")
    print(df.iloc[0].to_dict())
    
    # Нормализация схемы
    df_norm = normalize_schema(df, None)
    print(f"После нормализации: {len(df_norm)} строк, {len(df_norm.columns)} столбцов")
    
    return df_norm

def test_model_loading():
    """Тестирование загрузки модели"""
    print("\n=== Тест загрузки модели ===")
    
    from api.services.pipeline import _lazy_boot_tabular, get_model_and_features
    
    try:
        _lazy_boot_tabular()
        model, features = get_model_and_features()
        print(f"Модель загружена: {type(model)}")
        print(f"Количество признаков: {len(features)}")
        print(f"Первые 5 признаков: {features[:5]}")
        return model, features
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None, None

def test_prediction(df_norm, model, features):
    """Тестирование предсказаний"""
    print("\n=== Тест предсказаний ===")
    
    if model is None or features is None:
        print("Модель не загружена, пропускаем тест")
        return
    
    from api.services.pipeline import predict_tab, _align_feature_frame
    
    try:
        # Проверяем выравнивание признаков
        X = _align_feature_frame(df_norm)
        print(f"Форма данных после выравнивания: {X.shape}")
        print(f"Типы данных:")
        print(X.dtypes.value_counts())
        
        # Проверяем на NaN
        nan_counts = X.isnull().sum()
        print(f"NaN значений: {nan_counts.sum()}")
        if nan_counts.sum() > 0:
            print("Столбцы с NaN:")
            print(nan_counts[nan_counts > 0])
        
        # Пытаемся сделать предсказание
        result = predict_tab(df_norm, return_labels=True)
        print("Предсказание успешно!")
        print(f"Классы: {result.get('classes')}")
        print(f"Количество строк: {result.get('count')}")
        
        for i, proba in enumerate(result.get('proba', [])[:3]):
            max_idx = np.argmax(proba)
            classes = result.get('classes', [])
            class_name = classes[max_idx] if max_idx < len(classes) else f'class_{max_idx}'
            print(f"Строка {i+1}: {class_name} (вероятность: {max(proba):.3f})")
            
    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        import traceback
        traceback.print_exc()

def test_simple_api():
    """Тестирование API здоровья"""
    print("\n=== Тест API ===")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/inference/health", timeout=5)
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"API недоступен: {e}")

if __name__ == "__main__":
    print("StarHarbor Debug Script")
    print("=" * 50)
    
    # Тестируем по частям
    df_norm = test_data_loading()
    model, features = test_model_loading()
    test_prediction(df_norm, model, features)
    test_simple_api()
    
    print("\n=== Тест завершен ===")