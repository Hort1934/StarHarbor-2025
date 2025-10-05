#!/usr/bin/env python3
"""
Retrain the StarHarbor exoplanet classification model with proper labels
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

def load_and_prepare_data():
    """Load NASA data and prepare training dataset"""
    log.info("Loading NASA source files...")
    
    from api.utils.io import read_and_normalize
    from api.services.pipeline import _align_feature_frame
    
    # Load all three missions
    datasets = []
    
    # Kepler data
    try:
        kepler_df = read_and_normalize("data/sources/kepler.csv", mission="kepler")
        log.info(f"Loaded Kepler data: {len(kepler_df)} rows")
        datasets.append(kepler_df)
    except Exception as e:
        log.warning(f"Failed to load Kepler data: {e}")
    
    # K2 data  
    try:
        k2_df = read_and_normalize("data/sources/k2.csv", mission="k2")
        log.info(f"Loaded K2 data: {len(k2_df)} rows")
        datasets.append(k2_df)
    except Exception as e:
        log.warning(f"Failed to load K2 data: {e}")
    
    # TESS data
    try:
        tess_df = read_and_normalize("data/sources/tess.csv", mission="tess")
        log.info(f"Loaded TESS data: {len(tess_df)} rows")
        datasets.append(tess_df)
    except Exception as e:
        log.warning(f"Failed to load TESS data: {e}")
    
    if not datasets:
        raise ValueError("No datasets loaded successfully!")
    
    # Combine all datasets
    combined_df = pd.concat(datasets, ignore_index=True)
    log.info(f"Combined dataset: {len(combined_df)} rows")
    
    return combined_df

def create_labels(df):
    """Create proper labels from disposition columns"""
    log.info("Creating labels from dispositions...")
    
    # Initialize labels
    labels = []
    
    for _, row in df.iterrows():
        # Check different label columns based on mission
        disposition = None
        
        # Kepler labels
        if pd.notna(row.get('label_raw')):
            disposition = str(row['label_raw']).upper().strip()
        
        # K2 labels  
        elif pd.notna(row.get('disposition')):
            disposition = str(row['disposition']).upper().strip()
            
        # TESS labels
        elif pd.notna(row.get('tfopwg_disposition')):
            disposition = str(row['tfopwg_disposition']).upper().strip()
        
        # Map dispositions to our classes
        if disposition in ['CONFIRMED', 'CONFIRMED PLANET']:
            labels.append(2)  # confirmed
        elif disposition in ['CANDIDATE', 'PLANETARY CANDIDATE', 'PC', 'CP']:
            labels.append(1)  # candidate  
        elif disposition in ['FALSE POSITIVE', 'FP', 'NOT A PLANET']:
            labels.append(0)  # false positive
        else:
            # Default to false positive for unknown/missing labels
            labels.append(0)
    
    labels = np.array(labels)
    log.info(f"Label distribution: {np.bincount(labels)}")
    log.info(f"  0 (fp): {np.sum(labels == 0)}")
    log.info(f"  1 (candidate): {np.sum(labels == 1)}")  
    log.info(f"  2 (confirmed): {np.sum(labels == 2)}")
    
    return labels

def balance_dataset(X, y, min_samples_per_class=500):
    """Balance the dataset by undersampling majority class and oversampling minority classes"""
    log.info("Balancing dataset...")
    
    unique, counts = np.unique(y, return_counts=True)
    log.info(f"Original distribution: {dict(zip(unique, counts))}")
    
    # Find samples for each class
    indices_by_class = {}
    for class_label in unique:
        indices_by_class[class_label] = np.where(y == class_label)[0]
    
    # Balance by sampling
    balanced_indices = []
    
    for class_label in unique:
        class_indices = indices_by_class[class_label]
        
        if len(class_indices) >= min_samples_per_class:
            # Undersample if too many
            sampled = np.random.choice(class_indices, min_samples_per_class, replace=False)
        else:
            # Oversample if too few
            sampled = np.random.choice(class_indices, min_samples_per_class, replace=True)
        
        balanced_indices.extend(sampled)
    
    # Shuffle
    balanced_indices = np.random.permutation(balanced_indices)
    
    X_balanced = X.iloc[balanced_indices]
    y_balanced = y[balanced_indices]
    
    unique, counts = np.unique(y_balanced, return_counts=True)
    log.info(f"Balanced distribution: {dict(zip(unique, counts))}")
    
    return X_balanced, y_balanced

def train_model(X_train, y_train, X_val, y_val):
    """Train the Random Forest model"""
    log.info("Training Random Forest model...")
    
    # Initialize model with good parameters for exoplanet classification
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',  # Handle any remaining imbalance
        random_state=42,
        n_jobs=-1
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = model.predict(X_val)
    val_proba = model.predict_proba(X_val)
    
    log.info("Validation Results:")
    log.info(f"Classification Report:\n{classification_report(y_val, val_pred)}")
    log.info(f"Confusion Matrix:\n{confusion_matrix(y_val, val_pred)}")
    
    return model

def save_model_artifacts(model, feature_names, X_train, y_train, X_val, y_val, X_test, y_test):
    """Save all model artifacts"""
    log.info("Saving model artifacts...")
    
    models_dir = Path("models")
    
    # Save the trained model
    joblib.dump(model, models_dir / "tab_xgb.pkl")
    log.info("Saved trained model")
    
    # Save feature list
    with open(models_dir / "feature_list.json", 'w') as f:
        json.dump(feature_names, f, indent=2)
    log.info("Saved feature list")
    
    # Update target map
    target_map = {"fp": 0, "candidate": 1, "confirmed": 2}
    with open(models_dir / "target_map.json", 'w') as f:
        json.dump(target_map, f, indent=2)
    log.info("Saved target map")
    
    # Save data splits
    pd.DataFrame(X_train).to_parquet(models_dir / "X_train.parquet")
    pd.DataFrame(X_val).to_parquet(models_dir / "X_val.parquet") 
    pd.DataFrame(X_test).to_parquet(models_dir / "X_test.parquet")
    
    pd.DataFrame({"target": y_train}).to_parquet(models_dir / "y_train.parquet")
    pd.DataFrame({"target": y_val}).to_parquet(models_dir / "y_val.parquet")
    pd.DataFrame({"target": y_test}).to_parquet(models_dir / "y_test.parquet")
    
    log.info("Saved training data splits")
    
    # Create preprocessor (StandardScaler)
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, models_dir / "preprocessor.pkl")
    log.info("Saved preprocessor")

def main():
    """Main retraining pipeline"""
    log.info("🚀 Starting StarHarbor model retraining...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    try:
        # Load and prepare data
        df = load_and_prepare_data()
        
        # Create labels
        y = create_labels(df)
        
        # Prepare features using existing pipeline
        from api.services.pipeline import _align_feature_frame, _lazy_boot_tabular
        
        # Load feature list
        _lazy_boot_tabular()
        X = _align_feature_frame(df)
        
        log.info(f"Feature matrix shape: {X.shape}")
        log.info(f"Target vector shape: {y.shape}")
        
        # Remove rows with too many missing features (>50% NaN)
        valid_rows = X.isna().sum(axis=1) < (X.shape[1] * 0.5)
        X = X[valid_rows]
        y = y[valid_rows]
        
        log.info(f"After filtering: {X.shape[0]} samples")
        
        # Fill remaining NaN values
        X = X.fillna(0)
        
        # Balance dataset
        X_balanced, y_balanced = balance_dataset(X, y)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_balanced, y_balanced, test_size=0.15, random_state=42, stratify=y_balanced
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, random_state=42, stratify=y_temp
        )
        
        log.info(f"Training set: {X_train.shape[0]} samples")
        log.info(f"Validation set: {X_val.shape[0]} samples") 
        log.info(f"Test set: {X_test.shape[0]} samples")
        
        # Train model
        model = train_model(X_train, y_train, X_val, y_val)
        
        # Final evaluation on test set
        test_pred = model.predict(X_test)
        log.info("Final Test Results:")
        log.info(f"Classification Report:\n{classification_report(y_test, test_pred)}")
        
        # Save everything
        save_model_artifacts(
            model, list(X.columns), 
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        log.info("✅ Model retraining completed successfully!")
        log.info("🔄 Restart the API server to use the new model")
        
    except Exception as e:
        log.error(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()