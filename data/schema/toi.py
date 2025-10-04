from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict

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

def _apply_unit_conversions(df: pd.DataFrame, rules: Dict[str, Tuple[str, float]]) -> pd.DataFrame:
    out = df.copy()
    for raw_col, (target_col, factor) in rules.items():
        if raw_col in out.columns:
            raw_vals = pd.to_numeric(out[raw_col], errors="coerce") * factor
            if target_col in out.columns:
                mask = out[target_col].isna()
                out.loc[mask, target_col] = raw_vals[mask]
            else:
                out[target_col] = raw_vals
    return out

COLUMN_MAP: dict[str, str] = {
    "TESS Object of Interest": "toi_name",
    "TESS Input Catalog ID": "tic_id_raw",
    "TFOPWG Disposition (CP | FP | KB | PC)": "tfopwg_disposition",

    "RA [sexagesimal]": "ra_sexagesimal",
    "Dec [sexagesimal]": "dec_sexagesimal",
    "PMRA [mas/yr]": "pm_ra_masyr",
    "PMDec [mas/yr]": "pm_dec_masyr",

    "Planet Transit Midpoint [BJD]": "epoch_bjd",
    "Planet Orbital Period [days]": "period_days",
    "Planet Transit Duration [hours]": "duration_hours",
    "Planet Transit Depth [ppm]": "depth_ppm",
    "Planet Radius [R_Earth]": "rp_rearth",

    "Planet Insolation [Earth flux]": "insolation_earth",
    "Planet Equilibrium Temperature [K]": "eq_temp_k",

    "TESS Magnitude": "mag_tess",
    "Stellar Distance [pc]": "stellar_distance_pc",
    "Stellar Effective Temperature [K]": "stellar_teff_k",
    "Stellar log(g) [cm/s^2]": "stellar_logg_cgs",
    "Stellar Radius [R_Sun]": "stellar_radius_rsun",

    "TOI Created Date": "created_at",
    "Date Modified": "updated_at",
}

COLUMN_MAP.update({
    "RA [deg]": "ra_deg",
    "Dec [deg]": "dec_deg",
    "RA [degrees]": "ra_deg",
    "Dec [degrees]": "dec_deg",
    "TIC ID": "tic_id_raw",
    "TIC": "tic_id_raw",
    "TOI": "toi_name",
})

REQUIRED_COLS = ["TESS Input Catalog ID", "TESS Object of Interest"]

UNIT_CONVERSIONS: dict[str, Tuple[str, float]] = {
    #якщо попадеться тривалість у днях
    "Planet Transit Duration [days]": ("duration_hours", 24.0),
    #якщо глибина в частках
    "Planet Transit Depth [fraction]": ("depth_ppm", 1e6),
    #якщо радіус в км
    "Planet Radius [km]": ("rp_rearth", 1.0 / 6371.0),
    #координати в годинах/arcsec (рідко)
    "RA [hours]": ("ra_deg", 15.0),
    "Dec [arcsec]": ("dec_deg", 1.0 / 3600.0),
}

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_cols = {c: COLUMN_MAP[c] for c in COLUMN_MAP if c in df.columns}
    df = df.rename(columns=rename_cols)

    df["mission"] = "tess"

    df = _apply_unit_conversions(df, UNIT_CONVERSIONS)

    numeric_cols = [
        "tic_id_raw",
        "period_days", "duration_hours", "depth_ppm",
        "epoch_bjd", "rp_rearth",
        "insolation_earth", "eq_temp_k",
        "stellar_distance_pc", "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun",
        "pm_ra_masyr", "pm_dec_masyr",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "ra_deg" not in df.columns and "ra_sexagesimal" in df.columns:
        df["ra_deg"] = df["ra_sexagesimal"].apply(lambda x: _sexagesimal_to_deg(x, is_ra=True))
    if "dec_deg" not in df.columns and "dec_sexagesimal" in df.columns:
        df["dec_deg"] = df["dec_sexagesimal"].apply(lambda x: _sexagesimal_to_deg(x, is_ra=False))

    for dc in ["created_at", "updated_at"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    if "tfopwg_disposition" in df.columns:
        df["tfopwg_disposition"] = df["tfopwg_disposition"].astype(str)

    if "tic_id" not in df.columns:
        df["tic_id"] = pd.to_numeric(df.get("tic_id_raw"), errors="coerce")

    if "object_id" not in df.columns:
        mask = df["tic_id"].notna()
        obj = np.where(mask, "TIC " + df["tic_id"].astype("Int64").astype(str), df.get("toi_name"))
        df["object_id"] = obj


    if "toi_numeric" not in df.columns and "toi_name" in df.columns:
        toi_num = df["toi_name"].astype(str).str.extract(r"(\d+\.\d+|\d+)", expand=False)
        df["toi_numeric"] = pd.to_numeric(toi_num, errors="coerce")

    wanted = [
        "mission", "object_id",
        "toi_name", "tic_id", "tic_id_raw", "toi_numeric",
        "period_days", "duration_hours", "depth_ppm", "epoch_bjd",
        "rp_rearth", "insolation_earth", "eq_temp_k",
        "pm_ra_masyr", "pm_dec_masyr",
        "stellar_distance_pc", "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun",
        "ra_deg", "dec_deg", "mag_tess",
        "tfopwg_disposition",
        "created_at", "updated_at",
    ]
    for c in wanted:
        if c not in df.columns:
            df[c] = np.nan

    return df
