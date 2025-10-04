import argparse
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare ML features from processed exoplanet dataset")
    parser.add_argument("--input", required=True)
    parser.add_argument("--missions", nargs="+", choices=["kepler", "k2", "tess", "all"], default=["all"])
    parser.add_argument("--target", choices=["label_3way", "binary"], default="label_3way")
    parser.add_argument("--drop-invalid", choices=["yes", "no"], default="yes")
    parser.add_argument("--outdir", default="data/features/")
    parser.add_argument("--split", choices=["random", "by_mission", "random_grouped"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation percentage from train (0..0.5)")
    parser.add_argument("--group-col", type=str, default="system_key", help="Group column (to avoid leaks)")

    return parser.parse_args()

def setup_logging():
    os.makedirs("logs", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/prepare_features_{ts}.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    return ts

def load_and_filter(path, missions, drop_invalid):
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if "all" not in missions:
        df = df[df["mission"].isin(missions)]

    for c in ["period_days", "duration_hours", "depth_ppm"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    if drop_invalid == "yes":
        if "is_valid" in df.columns:
            df = df[df["is_valid"] == True]
        if "is_superseded" in df.columns:
            df = df[df["is_superseded"] != True]
    df = df[df["label_3way"].notna()]
    return df

def map_targets(df, target_mode, outdir):
    label_map = {"confirmed": 2, "candidate": 1, "fp": 0, "unknown": 0}

    df["label_3way"] = df["label_3way"].astype(str).str.lower().str.strip()
    df["target"] = df["label_3way"].map(label_map)

    if target_mode == "binary":
        df["target"] = df["target"].apply(lambda x: 1 if x in (1, 2) else 0)

    df = df[df["target"].notna()]
    df["target"] = df["target"].astype(int)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "target_map.json"), "w") as f:
        json.dump({"confirmed": 2, "candidate": 1, "fp": 0}, f)  # карта классов для UI/чтения
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        "period_days", "duration_hours", "depth_ppm",
        "rp_rearth", "stellar_radius_rsun", "stellar_teff_k",
        "stellar_logg_cgs", "sma_au", "mission"
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = np.nan

    period_d = pd.to_numeric(df["period_days"], errors="coerce")
    dur_h = pd.to_numeric(df["duration_hours"], errors="coerce")
    depth_ppm = pd.to_numeric(df["depth_ppm"], errors="coerce")
    r_p_re = pd.to_numeric(df["rp_rearth"], errors="coerce")
    r_s_rsun = pd.to_numeric(df["stellar_radius_rsun"], errors="coerce")
    teff = pd.to_numeric(df["stellar_teff_k"], errors="coerce")
    logg = pd.to_numeric(df.get("stellar_logg_cgs", df.get("stellar_logg")), errors="coerce")
    sma_au = pd.to_numeric(df["sma_au"], errors="coerce")

    with np.errstate(divide="ignore", invalid="ignore"):
        df["duration_ratio"] = dur_h / (period_d * 24.0)

    depth_frac = np.clip(depth_ppm, 0, None) / 1e6
    k_est = np.sqrt(depth_frac)
    df["k_est"] = k_est

    df["rp_est_rearth"] = k_est * r_s_rsun * 109.17

    denom = (r_p_re / (r_s_rsun * 109.17))
    df["k_vs_rp"] = np.where(np.isfinite(denom) & (denom != 0), k_est / denom, np.nan)

    df["log_period"] = np.log1p(np.clip(period_d, 0, None))
    df["log_duration"] = np.log1p(np.clip(dur_h, 0, None))
    df["log_teff"] = np.log(np.where(teff > 0, teff, np.nan))
    df["dur_over_p13"] = np.where(period_d > 0, dur_h / np.cbrt(period_d), np.nan)

    df["depth_over_rstar"] = k_est  

    df["insolation_rel_earth"] = np.where(
        (r_s_rsun > 0) & (teff > 0) & (sma_au > 0),
        (r_s_rsun ** 2) * ((teff / 5772.0) ** 4) / (sma_au ** 2),
        np.nan
    )

    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    if flag_cols:
        df["fp_flags_sum"] = df[flag_cols].sum(axis=1)

    if df["mission"].dtype == "O" or str(df["mission"].dtype).startswith("category"):
        df = pd.get_dummies(df, columns=["mission"], prefix="mission", dummy_na=False)

    return df

def build_preprocessor(df: pd.DataFrame, outdir: str):
    y = df["target"]
    cat_candidates = [
        "stellar_spectral_type", "stellar_class", "spectral_type",
        "disposition_source", "discovery_method"
    ]
    cat_cols = [c for c in cat_candidates if c in df.columns and df[c].dtype == "object"]

    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "target"]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ]) if cat_cols else "drop"

    transformers = [("num", numeric_pipe, num_cols)]
    if cat_cols:
        transformers.append(("cat", categorical_pipe, cat_cols))

    preprocessor = ColumnTransformer(transformers)

    X = df[num_cols + cat_cols] if cat_cols else df[num_cols]
    X_proc = preprocessor.fit_transform(X)

    feat_names = []
    feat_names.extend(num_cols)
    if cat_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
        ohe_names = ohe.get_feature_names_out(cat_cols).tolist()
        feat_names.extend(ohe_names)

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(outdir, "preprocessor.pkl"))
    return X_proc, y, feat_names

def split_data(df: pd.DataFrame, X, y: pd.Series, strategy: str, seed: int, outdir: str,
               val_size: float = 0.15, group_col: str = "system_key"):
    rng = np.random.RandomState(seed)

    def _as_idx(a):
        return np.array(a, dtype=int)

    if strategy == "random" and (y.nunique() < 2 or len(y) < 10):
        from sklearn.model_selection import train_test_split
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=seed)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=seed)
        splits = {
            "train": list(y_train.index),
            "val": list(y_val.index),
            "test": list(y_test.index)
        }
        with open(os.path.join(outdir, "splits.json"), "w") as f:
            json.dump(splits, f, indent=2)
        return X_train, X_val, X_test, y_train, y_val, y_test


    elif strategy == "by_mission":
        if not {"mission_kepler", "mission_k2", "mission_tess"}.issubset(set(df.columns)):
            raise ValueError("Expected columns mission_kepler, mission_k2, mission_tess after engineer_features().")

        kepler_idx = df.index[df["mission_kepler"] == 1]
        other_idx  = df.index[(df["mission_k2"] == 1) | (df["mission_tess"] == 1)]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
        kepler_y = y.loc[kepler_idx]
        train_k_idx, val_k_idx = next(sss.split(np.zeros(len(kepler_idx)), kepler_y))

        train_idx = kepler_idx.to_numpy()[train_k_idx]
        val_idx = kepler_idx.to_numpy()[val_k_idx]
        test_idx = other_idx.to_numpy()

        X_train, y_train = X[_as_idx(train_idx)], y.loc[train_idx]
        X_val, y_val = X[_as_idx(val_idx)], y.loc[val_idx]
        X_test, y_test = X[_as_idx(test_idx)], y.loc[test_idx]

        splits = {
            "train": list(map(int, train_idx)),
            "val": list(map(int, val_idx)),
            "test": list(map(int, test_idx))
        }

    elif strategy == "random_grouped":
        if group_col not in df.columns:
            raise ValueError(f"group_col='{group_col}' not found in data for random_grouped.")

        groups = df[group_col].astype(str).fillna("NA")

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_full_idx, test_idx = next(sgkf.split(np.zeros(len(y)), y, groups=groups))

        train_groups = groups.iloc[train_full_idx]
        y_train_full = y.iloc[train_full_idx]

        unique_groups = pd.Index(train_groups.unique())
        rng.shuffle(unique_groups.values)
        cut = int(np.ceil(len(unique_groups) * val_size))
        val_groups = set(unique_groups[:cut])
        mask_val = train_groups.isin(val_groups)

        val_idx = train_full_idx[mask_val.values]
        train_idx = train_full_idx[~mask_val.values]

        X_train, y_train = X[_as_idx(train_idx)], y.iloc[train_idx]
        X_val,   y_val = X[_as_idx(val_idx)], y.iloc[val_idx]
        X_test,  y_test = X[_as_idx(test_idx)], y.iloc[test_idx]

        splits = {
            "train": list(map(int, df.index[train_idx])),
            "val": list(map(int, df.index[val_idx])),
            "test": list(map(int, df.index[test_idx]))
        }

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    with open(os.path.join(outdir, "splits.json"), "w") as f:
        json.dump(splits, f, indent=2)

    return X_train, X_val, X_test, y_train, y_val, y_test

def save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, feature_list, outdir):
    pd.DataFrame(X_train, columns=feature_list).to_parquet(os.path.join(outdir, "X_train.parquet"))
    pd.DataFrame(X_val, columns=feature_list).to_parquet(os.path.join(outdir, "X_val.parquet"))
    pd.DataFrame(X_test, columns=feature_list).to_parquet(os.path.join(outdir, "X_test.parquet"))
    y_train.to_frame("target").to_parquet(os.path.join(outdir, "y_train.parquet"))
    y_val.to_frame("target").to_parquet(os.path.join(outdir, "y_val.parquet"))
    y_test.to_frame("target").to_parquet(os.path.join(outdir, "y_test.parquet"))
    with open(os.path.join(outdir, "feature_list.json"), "w") as f:
        json.dump(feature_list, f)

def generate_summary(df, outdir):
    report = ["# Feature Preparation Summary",
              f"Rows: {len(df)}",
              f"Columns: {len(df.columns)}",
              "## Missingness",
              df.isna().sum().to_string(),
              "## Basic Stats",
              df.describe().to_string()]
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n\n".join(report))

def main():
    args = parse_args()
    np.random.seed(args.seed)
    ts = setup_logging()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_and_filter(args.input, args.missions, args.drop_invalid)
    df = map_targets(df, args.target, args.outdir)
    df = engineer_features(df)
    X_proc, y, feature_list = build_preprocessor(df, args.outdir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
    df, X_proc, y, args.split, args.seed, args.outdir,
    val_size=args.val_size, group_col=args.group_col)
    save_outputs(X_train, X_val, X_test, y_train, y_val, y_test, feature_list, args.outdir)
    generate_summary(df, args.outdir)
    logging.info("Feature preparation complete.")

if __name__ == "__main__":
    main()
