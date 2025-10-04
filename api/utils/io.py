from __future__ import annotations
from typing import Optional, Union
import io
import importlib
from pathlib import Path
import pandas as pd

def read_table(path_or_bytes: Union[str, Path, bytes], *, suffix: Optional[str] = None) -> pd.DataFrame:
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        sfx = p.suffix.lower()
        if sfx in {".csv", ".tsv"}:
            sep = "," if sfx == ".csv" else "\t"
            return pd.read_csv(p, sep=sep)
        if sfx in {".parquet", ".pq"}:
            return pd.read_parquet(p)
        if sfx in {".fits", ".fit"}:
            from astropy.io import fits
            from astropy.table import Table
            with fits.open(str(p)) as hdul:
                h = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                if h is None:
                    raise ValueError("No table HDU found in FITS.")
                tab = Table(h.data)
                df = tab.to_pandas()
                # bytes -> str 
                df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
                return df
        raise ValueError(f"Unsupported file type: {sfx}")

    # bytes-like:
    sfx = (suffix or "").lower()
    if sfx in {".csv", ".tsv"}:
        sep = "," if sfx == ".csv" else "\t"
        return pd.read_csv(io.BytesIO(path_or_bytes), sep=sep)
    if sfx in {".parquet", ".pq"}:
        return pd.read_parquet(io.BytesIO(path_or_bytes))
    if sfx in {".fits", ".fit"}:
        from astropy.io import fits
        from astropy.table import Table
        with fits.open(io.BytesIO(path_or_bytes)) as hdul:
            h = next((h for h in hdul if getattr(h, "data", None) is not None), None)
            if h is None:
                raise ValueError("No table HDU found in FITS.")
            tab = Table(h.data)
            df = tab.to_pandas()
            df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
            return df

    raise ValueError("Provide a valid suffix ('.csv' | '.tsv' | '.parquet' | '.fits') for bytes input.")


def normalize_schema(df: pd.DataFrame, mission: Optional[str]) -> pd.DataFrame:
    if not mission:
        return df

    mission_norm = mission.strip().lower()
    try:
        mod = importlib.import_module(f"data.schema.{mission_norm}")
    except ModuleNotFoundError:
        # no schema module 
        return df

    if hasattr(mod, "normalize"):
        return mod.normalize(df.copy())
    return df


def read_and_normalize(
    path_or_bytes: Union[str, Path, bytes],
    *,
    mission: Optional[str] = None,
    suffix: Optional[str] = None,
) -> pd.DataFrame:
    df = read_table(path_or_bytes, suffix=suffix)
    return normalize_schema(df, mission)

