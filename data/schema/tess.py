from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional

def _sexagesimal_to_deg(val: Optional[str], is_ra: bool) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip().replace(" ", "")
    try:
        if "h" in s or "d" in s or "m" in s or "s" in s:
            s = s.lower().replace("h", ":").replace("d", ":").replace("m", ":").replace("s", "")
        parts = s.split(":")
        if len(parts) < 2:
            return np.nan
        sign = 1.0
        if not is_ra and parts[0].startswith("-"):
            sign = -1.0
        h_d = float(parts[0])
        m   = float(parts[1]) if len(parts) > 1 else 0.0
        sec = float(parts[2]) if len(parts) > 2 else 0.0
        if is_ra:
            hours = abs(h_d) + m/60.0 + sec/3600.0
            return hours * 15.0
        else:
            deg = abs(h_d) + m/60.0 + sec/3600.0
            return sign * deg
    except Exception:
        return np.nan

COLUMN_MAP: dict[str, str] = {
    # From actual TESS CSV file column names
    "toi": "toi_name",
    "tid": "tic_id_raw", 
    "tfopwg_disp": "tfopwg_disposition",

    "rastr": "ra_sexagesimal",
    "ra": "ra_deg",
    "decstr": "dec_sexagesimal", 
    "dec": "dec_deg",
    "st_pmra": "pm_ra_masyr",
    "st_pmdec": "pm_dec_masyr",

    "pl_tranmid": "epoch_bjd",
    "pl_orbper": "period_days",
    "pl_trandurh": "duration_hours", 
    "pl_trandep": "depth_ppm",
    "pl_rade": "rp_rearth",

    "pl_insol": "insolation_earth",
    "pl_eqt": "eq_temp_k",

    "st_tmag": "mag_tess",
    "st_dist": "stellar_distance_pc",
    "st_teff": "stellar_teff_k",
    "st_logg": "stellar_logg_cgs",
    "st_rad": "stellar_radius_rsun",

    "toi_created": "created_at",
    "rowupdate": "updated_at",
}

REQUIRED_COLS = ["tid", "toi"]

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Clean empty strings
    df = df.replace('', np.nan)

    # Rename columns
    rename_cols = {c: COLUMN_MAP[c] for c in COLUMN_MAP if c in df.columns}
    df = df.rename(columns=rename_cols)

    df["mission"] = "tess"

    # Convert numeric columns
    numeric_cols = [
        "tic_id_raw", "period_days", "duration_hours", "depth_ppm",
        "epoch_bjd", "rp_rearth", "insolation_earth", "eq_temp_k",
        "stellar_distance_pc", "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun",
        "pm_ra_masyr", "pm_dec_masyr", "ra_deg", "dec_deg", "mag_tess"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert sexagesimal coordinates if needed
    if "ra_deg" not in df.columns and "ra_sexagesimal" in df.columns:
        df["ra_deg"] = df["ra_sexagesimal"].apply(lambda x: _sexagesimal_to_deg(x, is_ra=True))
    if "dec_deg" not in df.columns and "dec_sexagesimal" in df.columns:
        df["dec_deg"] = df["dec_sexagesimal"].apply(lambda x: _sexagesimal_to_deg(x, is_ra=False))

    # Create TIC ID from raw
    if "tic_id" not in df.columns:
        df["tic_id"] = pd.to_numeric(df.get("tic_id_raw"), errors="coerce")

    # Create object ID
    if "object_id" not in df.columns:
        mask = df["tic_id"].notna()
        obj = np.where(mask, "TIC " + df["tic_id"].astype("Int64").astype(str), df.get("toi_name", ""))
        df["object_id"] = obj

    # Extract numeric TOI
    if "toi_numeric" not in df.columns and "toi_name" in df.columns:
        toi_num = df["toi_name"].astype(str).str.extract(r"(\d+\.\d+|\d+)", expand=False)
        df["toi_numeric"] = pd.to_numeric(toi_num, errors="coerce")

    # Ensure string columns
    if "tfopwg_disposition" in df.columns:
        df["tfopwg_disposition"] = df["tfopwg_disposition"].astype(str)

    # Create wanted columns with NaN if missing
    wanted = [
        "mission", "object_id", "toi_name", "tic_id", "tic_id_raw", "toi_numeric",
        "period_days", "duration_hours", "depth_ppm", "epoch_bjd",
        "rp_rearth", "insolation_earth", "eq_temp_k",
        "pm_ra_masyr", "pm_dec_masyr",
        "stellar_distance_pc", "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun",
        "ra_deg", "dec_deg", "mag_tess", "tfopwg_disposition",
        "created_at", "updated_at",
    ]
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan

    return df