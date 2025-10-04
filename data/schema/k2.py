from __future__ import annotations
import re
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

def _extract_epic(name: Optional[str]) -> Optional[str]:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return None
    m = re.search(r"(EPIC\s*\d+)", str(name), flags=re.IGNORECASE)
    return m.group(1).upper().replace("  ", " ") if m else None

COLUMN_MAP: dict[str, str] = {
    "Planet Name": "planet_name",
    "Host Name": "host_name",
    "Default Parameter Set": "default_param_set",
    "Archive Disposition": "label_raw",
    "Archive Disposition Reference": "label_ref",
    "Number of Stars": "n_stars",
    "Number of Planets": "n_planets",
    "Discovery Method": "discovery_method",
    "Discovery Year": "discovery_year",
    "Discovery Facility": "discovery_facility",
    "Solution Type": "solution_type",
    "Controversial Flag": "controversial_flag",
    "Planetary Parameter Reference": "planet_param_ref",

    "Orbital Period [days]": "period_days",
    "Orbit Semi-Major Axis [au]": "sma_au",

    "Planet Radius [Earth Radius]": "rp_rearth",
    "Planet Radius [Jupiter Radius]": "rp_rjup",

    "Planet Mass or Mass*sin(i) [Earth Mass]": "mp_sini_mearth",
    "Planet Mass or Mass*sin(i) [Jupiter Mass]": "mp_sini_mjup",
    "Planet Mass or Mass*sin(i) Provenance": "mass_provenance",

    "Eccentricity": "ecc",
    "Insolation Flux [Earth Flux]": "insolation_earth",
    "Equilibrium Temperature [K]": "eq_temp_k",
    "Data show Transit Timing Variations": "ttv_flag",

    "Stellar Parameter Reference": "stellar_param_ref",
    "Spectral Type": "stellar_spectral_type",
    "Stellar Effective Temperature [K]": "stellar_teff_k",
    "Stellar Radius [Solar Radius]": "stellar_radius_rsun",
    "Stellar Mass [Solar mass]": "stellar_mass_msun",
    "Stellar Metallicity [dex]": "stellar_metallicity_dex",
    "Stellar Metallicity Ratio": "stellar_metallicity_ratio",
    "Stellar Surface Gravity [log10(cm/s**2)]": "stellar_logg_cgs",
    "System Parameter Reference": "system_param_ref",

    "RA [sexagesimal]": "ra_sexagesimal",
    "Dec [sexagesimal]": "dec_sexagesimal",
    "Distance [pc]": "stellar_distance_pc",

    "V (Johnson) Magnitude": "mag_v",
    "Ks (2MASS) Magnitude": "mag_ks",
    "Gaia Magnitude": "mag_gaia",

    "Date of Last Update": "updated_at",
    "Planetary Parameter Reference Publication Date": "param_pub_date",
    "Release": "release_date",   

    "RA [deg]": "ra_deg",
    "Dec [deg]": "dec_deg",
    "RA [degrees]": "ra_deg",
    "Dec [degrees]": "dec_deg",
    "Kepmag": "mag_kepler",
    "Kepler Magnitude": "mag_kepler",
}

REQUIRED_COLS = ["Host Name", "Planet Name"]  

UNIT_CONVERSIONS: dict[str, tuple[str, float]] = {
    #Період|тривалість
    "Orbital Period [hours]": ("period_days", 1.0 / 24.0),
    "Transit Duration [days]": ("duration_hours", 24.0),  #якщо раптом є в K2-експорті

    #Радіус планети
    "Planet Radius [km]": ("rp_rearth", 1.0 / 6371.0),

    #Маса планети
    "Planet Mass or Mass*sin(i) [kg]": ("mp_sini_mearth", 1.0 / 5.9722e24),
    "Planet Mass or Mass*sin(i) [M_sun]": ("mp_sini_mjup", 1.0 / 0.000954588),  #1 M_sun ≈ 1047.35 M_jup

    #Радіус зірки
    "Stellar Radius [m]": ("stellar_radius_rsun", 1.0 / 6.957e8),

    #Маса зірки
    "Stellar Mass [kg]": ("stellar_mass_msun", 1.0 / 1.98847e30),

    #Температура зірки (зазвичай і так у K)
    "Stellar Effective Temperature [C]": ("stellar_teff_k", 1.0),  # приклад-заглушка (якщо зустрінеться - замінити на +273.15 вручну)

    #відстань
    "Distance [ly]": ("stellar_distance_pc", 1.0 / 3.26156),  # 1 pc ≈ 3.26156 ly

    #Координати
    "RA [hours]": ("ra_deg", 15.0),  #годинник -> градуси
    #Dec у градусах зазвичай вже у deg; якщо в arcsec:
    "Dec [arcsec]": ("dec_deg", 1.0 / 3600.0),
}

def _apply_unit_conversions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for raw_col, (target_col, factor) in UNIT_CONVERSIONS.items():
        if raw_col in out.columns:
            raw_vals = pd.to_numeric(out[raw_col], errors="coerce") * factor
            if target_col in out.columns:
                mask = out[target_col].isna()
                out.loc[mask, target_col] = raw_vals[mask]
            else:
                out[target_col] = raw_vals
    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_cols = {c: COLUMN_MAP[c] for c in COLUMN_MAP if c in df.columns}
    df = df.rename(columns=rename_cols)

    df["mission"] = "k2"

    df = _apply_unit_conversions(df)

    numeric_cols = [
        "period_days", "sma_au",
        "rp_rearth", "rp_rjup",
        "mp_sini_mearth", "mp_sini_mjup",
        "ecc", "insolation_earth", "eq_temp_k",
        "stellar_teff_k", "stellar_radius_rsun", "stellar_mass_msun",
        "stellar_metallicity_dex", "stellar_metallicity_ratio",
        "stellar_logg_cgs", "stellar_distance_pc",
        "mag_v", "mag_ks", "mag_gaia",
        "n_stars", "n_planets", "discovery_year",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["ttv_flag", "controversial_flag"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
            df[c + "_bool"] = df[c].astype(bool)


    if "ra_sexagesimal" in df.columns:
        df["ra_deg"] = df["ra_sexagesimal"].apply(lambda x: _sexagesimal_to_deg(x, is_ra=True))
    if "dec_sexagesimal" in df.columns:
        df["dec_deg"] = df["dec_sexagesimal"].apply(lambda x: _sexagesimal_to_deg(x, is_ra=False))

    if "rp_rearth" in df.columns and "rp_rjup" in df.columns:
        mask = df["rp_rearth"].isna() & df["rp_rjup"].notna()
        df.loc[mask, "rp_rearth"] = df.loc[mask, "rp_rjup"] * 11.209
    if "mp_sini_mearth" in df.columns and "mp_sini_mjup" in df.columns:
        mask = df["mp_sini_mearth"].isna() & df["mp_sini_mjup"].notna()
        df.loc[mask, "mp_sini_mearth"] = df.loc[mask, "mp_sini_mjup"] * 317.828

    for dc in ["updated_at", "param_pub_date", "release_date"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    if "object_id" not in df.columns:
        epic_from_host = df["host_name"].apply(_extract_epic) if "host_name" in df.columns else pd.Series([None]*len(df))
        epic_from_plan = df["planet_name"].apply(_extract_epic) if "planet_name" in df.columns else pd.Series([None]*len(df))
        obj = epic_from_host.fillna(epic_from_plan)
        if obj.isna().all():
            obj = df.get("planet_name", pd.Series([np.nan]*len(df)))
        df["object_id"] = obj

    if "epic_id" not in df.columns:
        epic_numeric = df["object_id"].astype(str).str.extract(r"EPIC\s*(\d+)", flags=re.IGNORECASE)[0]
        df["epic_id"] = pd.to_numeric(epic_numeric, errors="coerce")

    wanted = [
        "mission", "object_id",
        "planet_name", "host_name",
        "period_days", "sma_au",
        "rp_rearth", "rp_rjup",
        "mp_sini_mearth", "mp_sini_mjup", "mass_provenance",
        "ecc", "insolation_earth", "eq_temp_k",
        "ttv_flag",
        "stellar_teff_k", "stellar_logg_cgs", "stellar_radius_rsun", "stellar_mass_msun",
        "stellar_metallicity_dex", "stellar_metallicity_ratio",
        "stellar_distance_pc",
        "ra_deg", "dec_deg",
        "mag_v", "mag_ks", "mag_gaia",
        "label_raw", "label_ref",
        "discovery_method", "discovery_year", "discovery_facility",
        "solution_type", "controversial_flag",
        "planet_param_ref", "stellar_param_ref", "system_param_ref",
        "updated_at", "param_pub_date", "release_date",
        "n_stars", "n_planets", "default_param_set",
    ]

    wanted_extra = [
    "mag_kepler",  
    "epic_id",
    ] 

    for c in wanted_extra:
        if c not in df.columns:
            df[c] = np.nan

    return df
