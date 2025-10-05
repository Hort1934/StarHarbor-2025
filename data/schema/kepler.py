from __future__ import annotations
import numpy as np
import pandas as pd

COLUMN_MAP: dict[str, str] = {
    "kepid": "kepid",
    "kepoi_name": "koi_name", 
    "kepler_name": "planet_name",

    "koi_disposition": "label_raw",
    "koi_pdisposition": "label_raw_kepler",
    "koi_score": "disposition_score",

    "koi_fpflag_nt": "flag_not_transit_like",
    "koi_fpflag_ss": "flag_eclipse", 
    "koi_fpflag_co": "flag_centroid",
    "koi_fpflag_ec": "flag_ephemeris_match",

    "koi_period": "period_days",
    "koi_time0bk": "epoch_bkjd",
    "koi_impact": "impact",
    "koi_duration": "duration_hours",
    "koi_depth": "depth_ppm", 
    "koi_model_snr": "snr",

    "koi_prad": "rp_rearth",
    "koi_teq": "eq_temp_k",
    "koi_insol": "insolation_earth",

    "koi_steff": "stellar_teff_k",
    "koi_slogg": "stellar_logg_cgs", 
    "koi_srad": "stellar_radius_rsun",

    "ra": "ra_deg",
    "dec": "dec_deg", 
    "koi_kepmag": "mag_kepler",

    "koi_tce_plnt_num": "tce_planet_number",
    "koi_tce_delivname": "tce_delivery",
}

REQUIRED_COLS: list[str] = ["kepid", "kepoi_name"]

BKJD_TO_BJD_OFFSET = 2_454_833.0

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Clean empty strings
    df = df.replace('', np.nan)

    # Rename columns using the map
    rename_cols = {c: COLUMN_MAP[c] for c in COLUMN_MAP if c in df.columns}
    df = df.rename(columns=rename_cols)

    df["mission"] = "kepler"

    # Set object_id
    if "object_id" not in df.columns:
        if "koi_name" in df.columns:
            kepid_col = df["kepid"] if "kepid" in df.columns else pd.Series([0] * len(df))
            df["object_id"] = df["koi_name"].fillna(kepid_col.astype(str))
        else:
            kepid_col = df["kepid"] if "kepid" in df.columns else pd.Series([0] * len(df))
            df["object_id"] = kepid_col.astype(str)

    # Convert BKJD to BJD
    if "epoch_bkjd" in df.columns and "epoch_bjd" not in df.columns:
        df["epoch_bjd"] = pd.to_numeric(df["epoch_bkjd"], errors="coerce") + BKJD_TO_BJD_OFFSET

    # Convert numeric columns
    num_like = [
        "kepid", "period_days", "epoch_bjd", "impact",
        "duration_hours", "depth_ppm", "snr", 
        "rp_rearth", "eq_temp_k", "insolation_earth",
        "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun",
        "ra_deg", "dec_deg", "disposition_score",
        "tce_planet_number",
    ]
    for c in num_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert flag columns to integers
    for c in ["flag_not_transit_like", "flag_eclipse", "flag_centroid", "flag_ephemeris_match"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Sum false positive flags
    flag_cols = ["flag_not_transit_like", "flag_eclipse", "flag_centroid", "flag_ephemeris_match"]
    available_flags = [c for c in flag_cols if c in df.columns]
    if available_flags:
        df["fp_flags_sum"] = df[available_flags].fillna(0).sum(axis=1)
    else:
        df["fp_flags_sum"] = 0

    # Ensure label_raw is string
    if "label_raw" in df.columns:
        df["label_raw"] = df["label_raw"].astype(str)

    # Create desired columns with NaN if missing
    wanted = [
        "mission", "object_id", "planet_name",
        "kepid", "koi_name", 
        "period_days", "epoch_bjd", "duration_hours", "depth_ppm", "snr", "impact",
        "rp_rearth", "eq_temp_k", "insolation_earth",
        "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun", 
        "ra_deg", "dec_deg", "mag_kepler",
        "flag_centroid", "flag_eclipse", "flag_ephemeris_match", "flag_not_transit_like",
        "fp_flags_sum", "label_raw", "disposition_score"
    ]
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan

    return df