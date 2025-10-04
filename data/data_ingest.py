import argparse
import os
import sys
import yaml
import json
import logging
import hashlib
from datetime import datetime
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
import astropy
from astropy.io import fits
from astropy.table import Table
import importlib

try:
    HAS_TABULATE = True
except Exception:
    HAS_TABULATE = False

def now_ts_for_path() -> str:
    return datetime.now().strftime("%Y%m%d")

def now_ts_for_log() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def file_checksum(path: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def df_to_markdown(df: pd.DataFrame, max_rows: int = 50) -> str:
    if df is None or df.empty:
        return "_No data_"
    df_show = df.head(max_rows)
    if HAS_TABULATE:
        return df_show.to_markdown(index=False)
    header = "| " + " | ".join(df_show.columns.astype(str)) + " |"
    sep = "| " + " | ".join(["---"] * len(df_show.columns)) + " |"
    rows = ["| " + " | ".join(map(lambda x: "" if pd.isna(x) else str(x), row)) + " |"
            for row in df_show.values]
    return "\n".join([header, sep] + rows)

def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan] * len(df), index=df.index)

def round_period(val: Optional[float], ndigits: int = 4) -> Optional[float]:
    if pd.isna(val):
        return np.nan
    try:
        return round(float(val), ndigits)
    except Exception:
        return np.nan

def setup_logging(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    timestamp = now_ts_for_log()
    log_path = os.path.join(log_dir, f"data_ingest_{timestamp}.log")

    logger = logging.getLogger()
    logger.handlers = []  
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

    logging.info("Starting data ingestion pipeline")
    return timestamp

def parse_args():
    parser = argparse.ArgumentParser(description="NASA Exoplanet Data Ingestion")
    parser.add_argument("--missions", nargs="+", choices=["kepler", "k2", "tess"], required=True)
    parser.add_argument("--sources", nargs="+", required=True,
                        help='Local paths or "auto" to resolve from data/sources/<mission>.(csv|tsv|fits)')
    parser.add_argument("--outdir-raw", default="data/raw/")
    parser.add_argument("--outdir-processed", default="data/processed/")
    parser.add_argument("--format", choices=["csv", "parquet"], default="parquet")
    parser.add_argument("--deduplicate", choices=["yes", "no"], default="no")
    parser.add_argument("--emit-qc-report", choices=["yes", "no"], default="no")
    return parser.parse_args()

def resolve_sources(missions: List[str]) -> List[str]:
    resolved = []
    for mission in missions:
        base = os.path.join("data", "sources", mission)
        candidates = [f"{base}.csv", f"{base}.tsv", f"{base}.fits"]
        found = next((c for c in candidates if os.path.exists(c)), None)
        if not found:
            raise FileNotFoundError(
                f"Auto source not found for {mission}. Expect one of: {', '.join(candidates)}"
            )
        resolved.append(found)
    return resolved

def _read_csv_flex(path: str) -> pd.DataFrame:
    for kw in (
        {"sep": None, "engine": "python"},  
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
        {"sep": "|"},
    ):
        try:
            return pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="skip", **kw)
        except Exception:
            pass
    raise

def load_raw(source: str, mission: str, outdir_raw: str, timestamp_for_names: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    ext = os.path.splitext(source)[-1].lower()
    logging.info(f"Loading raw source for {mission}: {source}")
    
    if ext in [".csv", ".tsv"]:
        df = _read_csv_flex(source)

    elif ext == ".fits":
        with fits.open(source) as hdul:
            table_hdu = next((hdu for hdu in hdul if hasattr(hdu, "data") and hdu.data is not None), None)
            if table_hdu is None:
                raise ValueError("No table HDU with data found in FITS.")
            tab = Table(table_hdu.data)
            df = tab.to_pandas()
            df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    required_ids = {
        "kepler": ["kepid", "koi_name"],
        "k2": ["epic_id"],
        "tess": ["tic_id", "toi"]
    }
    missing = [c for c in required_ids.get(mission, []) if c not in df.columns]
    if missing:
        logging.warning(f"Missing identifiers for {mission}: {missing}")

    os.makedirs(outdir_raw, exist_ok=True)

    date_tag = timestamp_for_names or now_ts_for_path()
    raw_ext = ext.lstrip(".")
    raw_basename = f"{mission}_raw_{date_tag}.{raw_ext}"
    raw_path = os.path.join(outdir_raw, raw_basename)

    try:
        if os.path.isfile(source):
            with open(source, "rb") as src_f, open(raw_path, "wb") as dst_f:
                dst_f.write(src_f.read())
        else:
            if ext in [".csv", ".tsv"]:
                df.to_csv(raw_path, index=False)
            elif ext == ".fits":
                raw_path = os.path.join(outdir_raw, f"{mission}_raw_{date_tag}.csv")
                df.to_csv(raw_path, index=False)
    except Exception as e:
        logging.error(f"Failed to save exact raw copy for {mission}: {e}")

    parquet_path = os.path.join(outdir_raw, f"{mission}_raw_{date_tag}.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as e:
        logging.error(f"Failed to save parquet raw for {mission}: {e}")

    saved_info = {}
    if os.path.exists(raw_path):
        saved_info["raw_path"] = raw_path
        saved_info["raw_checksum_sha256"] = file_checksum(raw_path)
    if os.path.exists(parquet_path):
        saved_info["raw_parquet_path"] = parquet_path
        saved_info["raw_parquet_checksum_sha256"] = file_checksum(parquet_path)

    return df, saved_info

def normalize_schema(df: pd.DataFrame, mission: str) -> pd.DataFrame:
    try:
        schema_mod = importlib.import_module(f"data.schema.{mission}")
        if hasattr(schema_mod, "normalize"):
            return schema_mod.normalize(df.copy())
    except ModuleNotFoundError:
        pass

    schema_path = os.path.join("data", "schema", f"{mission}_schema.yaml")
    if not os.path.exists(schema_path):
        logging.warning(f"Schema YAML not found for {mission}: {schema_path}. Proceeding without renames.")
        schema = {"column_map": {}, "unit_conversion": {}}
    else:
        with open(schema_path, "r") as f:
            schema = yaml.safe_load(f) or {}

    column_map = schema.get("column_map", {})
    unit_conv  = schema.get("unit_conversion", {})

    df = df.rename(columns=column_map)

    for col, factor in unit_conv.items():
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce") * factor
            except Exception as e:
                logging.warning(f"Unit conversion failed for {col}: {e}")

    df["mission"] = mission

    if "object_id" not in df.columns:
        if mission == "kepler":
            df["object_id"] = safe_series(df, "koi_name").fillna(safe_series(df, "kepid"))
        elif mission == "tess":
            df["object_id"] = safe_series(df, "toi").fillna(safe_series(df, "tic_id"))
        elif mission == "k2":
            planet_series = safe_series(df, "planet_name").astype(str)
            epic_id = planet_series.str.extract(r"(EPIC\s*\d+)", expand=False)
            df["object_id"] = epic_id.fillna(planet_series)
        else:
            df["object_id"] = np.nan

    key_series = df.apply(
        lambda r: f"{mission}|{str(r.get('object_id'))}|{round_period(r.get('period_days'))}",
        axis=1
    )
    df["row_id"] = key_series.apply(lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest())
    df["system_key"] = key_series

    wanted_cols = [
        "mission", "object_id", "planet_name",
        "kepid", "epic_id", "tic_id",
        "koi_name", "toi",
        "period_days", "epoch_bjd", "duration_hours", "depth_ppm", "snr", "impact",
        "rp_rearth", "eq_temp_k", "insolation_earth",
        "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun", "stellar_distance_pc",
        "mag_kepler", "mag_tess",
        "flag_centroid", "flag_eclipse", "flag_ephemeris_match", "flag_not_transit_like",
        "label_raw", "label_3way",
        "updated_at"
    ]
    for c in wanted_cols:
        if c not in df.columns:
            df[c] = np.nan

    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    if "duration_days" in df.columns and "duration_hours" in df.columns:
        mask_convert = df["duration_hours"].isna() & df["duration_days"].notna()
        df.loc[mask_convert, "duration_hours"] = pd.to_numeric(df.loc[mask_convert, "duration_days"], errors="coerce") * 24.0

    if "depth_fraction" in df.columns and "depth_ppm" in df.columns:
        mask_convert = df["depth_ppm"].isna() & df["depth_fraction"].notna()
        df.loc[mask_convert, "depth_ppm"] = pd.to_numeric(df.loc[mask_convert, "depth_fraction"], errors="coerce") * 1e6

    if "stellar_radius_m" in df.columns and "stellar_radius_rsun" in df.columns:
        RSUN_M = 6.957e8
        mask_convert = df["stellar_radius_rsun"].isna() & df["stellar_radius_m"].notna()
        df.loc[mask_convert, "stellar_radius_rsun"] = pd.to_numeric(df.loc[mask_convert, "stellar_radius_m"], errors="coerce") / RSUN_M

    df["label_3way"] = df.get("label_3way", pd.Series([np.nan] * len(df), index=df.index))
    raw = df.get("label_raw", pd.Series([np.nan] * len(df), index=df.index)).astype(str).str.upper()

    def map_label(x: str) -> str:
        if pd.isna(x) or x.strip() == "":
            return "unknown"
        if "CONFIRM" in x or x == "C" or x == "CONFIRMED":
            return "confirmed"
        if "CANDIDATE" in x or x == "PC":
            return "candidate"
        if "FALSE" in x or "FP" in x or "FALSE POSITIVE" in x:
            return "fp"
        return "unknown"

    df["label_3way"] = raw.apply(map_label)
    return df

def load_qc_config(path: str = "data/schema/qc.yaml") -> dict:
    defaults = {
        "duration_period_max_ratio": 0.20,
        "impact_max": 1.5,
        "min_depth_ppm": 0.0,
    }
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            defaults.update(cfg)
    except FileNotFoundError:
        logging.warning("QC config not found, using defaults.")
    return defaults

def apply_qc_checks(df: pd.DataFrame, qc_cfg: dict | None = None) -> pd.DataFrame:
    qc = qc_cfg or load_qc_config()
    ratio = df["duration_hours"] / (df["period_days"] * 24.0)
    df["qc_ratio"] = ratio

    df["qc_bad_ratio"] = ratio > qc["duration_period_max_ratio"]
    df["qc_bad_impact"] = df["impact"] > qc["impact_max"]
    df["qc_low_depth"] = df["depth_ppm"] < qc["min_depth_ppm"]

    df["is_valid"] = ~(df["qc_bad_ratio"] | df["qc_bad_impact"])
    return df

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    period = pd.to_numeric(safe_series(df, "period_days"), errors="coerce")
    df["period_rounded"] = period.apply(lambda v: round_period(v, 4))

    if "updated_at" not in df.columns:
        df["updated_at"] = pd.NaT
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    fill_ts = pd.Timestamp("1970-01-01")
    df["updated_at_filled"] = df["updated_at"].fillna(fill_ts)

    df.sort_values(["mission", "object_id", "period_rounded", "updated_at_filled"],
                   ascending=[True, True, True, False],
                   inplace=True)

    subset_keys = ["mission", "object_id", "period_rounded"]
    df["is_superseded"] = df.duplicated(subset=subset_keys, keep="first")

    return df

def generate_qc_report(df: pd.DataFrame, qc_outdir: str, timestamp_path: str) -> str:
    os.makedirs(qc_outdir, exist_ok=True)
    report_lines = [f"# QC Report ({timestamp_path})", ""]

    if "mission" in df.columns:
        counts = df["mission"].value_counts(dropna=False).rename_axis("mission").reset_index(name="count")
        report_lines += ["## Row counts per mission", df_to_markdown(counts), ""]
    else:
        report_lines += ["## Row counts per mission", "_Column 'mission' missing_", ""]

    if "label_3way" in df.columns:
        classes = df["label_3way"].value_counts(dropna=False).rename_axis("label_3way").reset_index(name="count")
        report_lines += ["## Class distribution", df_to_markdown(classes), ""]
    else:
        report_lines += ["## Class distribution", "_Column 'label_3way' missing_", ""]

    missing = df.isna().sum().reset_index().rename(columns={"index": "column", 0: "missing"})
    report_lines += ["## Missing value counts", df_to_markdown(missing), ""]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        report_lines += ["## Value ranges (numeric)", df_to_markdown(df[numeric_cols].describe()), ""]
    else:
        report_lines += ["## Value ranges (numeric)", "_No numeric columns_", ""]

    anomalies_mask = pd.Series([False] * len(df), index=df.index)
    period = pd.to_numeric(safe_series(df, "period_days"), errors="coerce")
    duration = pd.to_numeric(safe_series(df, "duration_hours"), errors="coerce")
    impact = pd.to_numeric(safe_series(df, "impact"), errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = duration / period
    anomalies_mask |= (ratio > 0.2).fillna(False)
    anomalies_mask |= (impact > 1.5).fillna(False)
    anomalies_cols = [c for c in ["mission", "object_id", "duration_hours", "period_days", "impact", "qc_flag"] if c in df.columns]
    anomalies_df = df.loc[anomalies_mask, anomalies_cols]
    report_lines += ["## Top anomalies", df_to_markdown(anomalies_df, max_rows=100), ""]

    qc_path = os.path.join(qc_outdir, f"qc_report_{timestamp_path}.md")
    with open(qc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logging.info(f"QC report saved: {qc_path}")
    return qc_path

def save_outputs(df: pd.DataFrame,
                 outdir_processed: str,
                 fmt: str,
                 timestamp_path: str,
                 missions: List[str],
                 raw_artifacts: Dict[str, Dict[str, str]],
                 deduplicate_enabled: bool) -> Dict[str, str]:
    os.makedirs(outdir_processed, exist_ok=True)

    combined_path = os.path.join(outdir_processed, f"exoplanets_common_{timestamp_path}.{fmt}")
    if fmt == "csv":
        df.to_csv(combined_path, index=False)
    else:
        df.to_parquet(combined_path, index=False)
    logging.info(f"Saved combined file: {combined_path}")

    per_mission_paths = {}
    for mission in missions:
        df_m = df[df["mission"] == mission]
        path = os.path.join(outdir_processed, f"{mission}_processed_{timestamp_path}.{fmt}")
        if fmt == "csv":
            df_m.to_csv(path, index=False)
        else:
            df_m.to_parquet(path, index=False)
        per_mission_paths[mission] = path
        logging.info(f"Saved {mission} file: {path}")

    metadata = {
        "timestamp": timestamp_path,
        "row_count": int(len(df)),
        "missions": missions,
        "format": fmt,
        "schema_version": "1.0",
        "deduplication_enabled": deduplicate_enabled,
        "deduplication_key": "(mission, object_id, round(period_days, 1e-4))",
        "raw_artifacts": raw_artifacts,
        "processed_checksums": {
            "combined_sha256": file_checksum(combined_path),
            **{f"{m}_sha256": file_checksum(per_mission_paths[m]) for m in missions}
        },
        "library_versions": {
            "python": sys.version.split(" ")[0],
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "astropy": astropy.__version__,
            "pyyaml": yaml.__version__,
            "tabulate_available": HAS_TABULATE
        }
    }
    meta_path = os.path.join(outdir_processed, f"metadata_{timestamp_path}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logging.info(f"Saved metadata: {meta_path}")

    artifacts = {"combined_path": combined_path, "metadata_path": meta_path, **per_mission_paths}
    return artifacts

def main():
    args = parse_args()
    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.outdir_raw, exist_ok=True)
    os.makedirs(args.outdir_processed, exist_ok=True)
    os.makedirs(os.path.join(args.outdir_processed, "qc"), exist_ok=True)

    log_timestamp = setup_logging("logs")
    date_tag = now_ts_for_path()  

    if args.sources == ["auto"]:
        try:
            sources = resolve_sources(args.missions)
        except Exception as e:
            logging.error(f"Auto source resolution failed: {e}")
            sys.exit(1)
    else:
        if len(args.missions) != len(args.sources):
            logging.error("Mismatch between number of missions and sources.")
            sys.exit("Error: --missions and --sources must have the same length (or use --sources auto).")
        sources = args.sources

    all_dfs = []
    raw_artifacts: Dict[str, Dict[str, str]] = {}

    for mission, source in zip(args.missions, sources):
        try:
            df_raw, saved_info = load_raw(source, mission, args.outdir_raw, date_tag)
            raw_artifacts[mission] = saved_info

            df_norm = normalize_schema(df_raw, mission)
            df_qc = apply_qc_checks(df_norm)

            if args.deduplicate == "yes":
                df_qc = deduplicate(df_qc)

            all_dfs.append(df_qc)
            logging.info(f"Completed pipeline for {mission}: rows={len(df_qc)}")
        except Exception as e:
            logging.exception(f"Error processing {mission}: {e}")
            continue

    if not all_dfs:
        logging.error("No dataframes were successfully processed. Exiting.")
        sys.exit(1)

    df_all = pd.concat(all_dfs, ignore_index=True)
    artifacts = save_outputs(
        df_all,
        args.outdir_processed,
        args.format,
        date_tag,
        args.missions,
        raw_artifacts,
        deduplicate_enabled=(args.deduplicate == "yes")
    )

    if args.emit_qc_report == "yes":
        qc_dir = os.path.join(args.outdir_processed, "qc")
        try:
            generate_qc_report(df_all, qc_dir, date_tag)
        except Exception as e:
            logging.exception(f"Failed to generate QC report: {e}")

    logging.info("Data ingestion pipeline finished.")

if __name__ == "__main__":
    main()
