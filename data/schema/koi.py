from __future__ import annotations
import numpy as np
import pandas as pd

COLUMN_MAP: dict[str, str] = {
    "KepID": "kepid",
    "KOI Name": "koi_name",
    "Kepler Name": "planet_name",

    "Exoplanet Archive Disposition": "label_raw",
    "Disposition Using Kepler Data": "label_raw_kepler",
    "Disposition Score": "disposition_score",

    "Not Transit-Like False Positive Flag": "flag_not_transit_like",
    "Stellar Eclipse False Positive Flag": "flag_eclipse",
    "Centroid Offset False Positive Flag": "flag_centroid",
    "Ephemeris Match Indicates Contamination False Positive Flag": "flag_ephemeris_match",

    "Orbital Period [days]": "period_days",
    "Transit Epoch [BKJD]": "epoch_bkjd",        
    "Impact Parameter": "impact",
    "Transit Duration [hrs]": "duration_hours",
    "Transit Depth [ppm]": "depth_ppm",
    "Transit Signal-to-Noise": "snr",

    "Planetary Radius [Earth radii]": "rp_rearth",
    "Equilibrium Temperature [K]": "eq_temp_k",
    "Insolation Flux [Earth flux]": "insolation_earth",

    "Stellar Effective Temperature [K]": "stellar_teff_k",
    "Stellar Surface Gravity [log10(cm/s**2)]": "stellar_logg_cgs",
    "Stellar Radius [Solar radii]": "stellar_radius_rsun",

    "RA [decimal degrees]": "ra_deg",
    "Dec [decimal degrees]": "dec_deg",
    "Kepler-band [mag]": "mag_kepler",

    "TCE Planet Number": "tce_planet_number",
    "TCE Delivery": "tce_delivery",
}

REQUIRED_COLS: list[str] = ["KepID", "KOI Name"]

BKJD_TO_BJD_OFFSET = 2_454_833.0

UNIT_CONVERSIONS: dict[str, float] = {
    "Transit Depth (fraction)": 1e6,   #depth_ppm

    "Transit Duration [days]": 24.0,   #duration_hours

    "Stellar Radius [m]": 1.0 / 6.957e8,   #stellar_radius_rsun
}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_cols = {c: COLUMN_MAP[c] for c in COLUMN_MAP if c in df.columns}
    df = df.rename(columns=rename_cols)

    df["mission"] = "kepler"

    if "object_id" not in df.columns:
        if "koi_name" in df.columns:
            df["object_id"] = df["koi_name"].fillna(df.get("kepid"))
        else:
            df["object_id"] = df.get("kepid")

    if "epoch_bkjd" in df.columns and "epoch_bjd" not in df.columns:
        df["epoch_bjd"] = pd.to_numeric(df["epoch_bkjd"], errors="coerce") + BKJD_TO_BJD_OFFSET

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

    for c in ["flag_not_transit_like", "flag_eclipse", "flag_centroid", "flag_ephemeris_match"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    if "label_raw" in df.columns:
        df["label_raw"] = df["label_raw"].astype(str)

    wanted = [
        "mission", "object_id", "planet_name",
        "kepid", "koi_name",
        "period_days", "epoch_bjd", "duration_hours", "depth_ppm", "snr", "impact",
        "rp_rearth", "eq_temp_k", "insolation_earth",
        "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun",
        "ra_deg", "dec_deg", "mag_kepler",
        "flag_centroid", "flag_eclipse", "flag_ephemeris_match", "flag_not_transit_like",
        "label_raw", "disposition_score"
    ]
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan

    return df
