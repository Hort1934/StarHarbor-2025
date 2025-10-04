from __future__ import annotations
from typing import Optional, Union
import io as _io
import importlib
from pathlib import Path
import pandas as pd
import csv

def _read_csv_robust(path_or_buf, *, sep_hint: Optional[str] = None):
    if sep_hint in (",", "\t", ";", "|"):
        try:
            return pd.read_csv(
                path_or_buf,
                sep=sep_hint,
                engine="python",
                encoding="utf-8-sig",
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                on_bad_lines="skip",
                skip_blank_lines=True,
            )
        except Exception:
            pass

    try:
        return pd.read_csv(
            path_or_buf,
            sep=None,                   
            engine="python",
            encoding="utf-8-sig",
            quotechar='"',
            doublequote=True,
            escapechar="\\",
            on_bad_lines="skip",
            skip_blank_lines=True,
        )
    except Exception:
        pass

    for s in (",", ";", "\t", "|"):
        try:
            return pd.read_csv(
                path_or_buf,
                sep=s,
                engine="python",
                encoding="utf-8-sig",
                quotechar='"',
                doublequote=True,
                escapechar="\\",
                on_bad_lines="skip",
                skip_blank_lines=True,
            )
        except Exception:
            continue

    if isinstance(path_or_buf, (str, Path)):
        with open(path_or_buf, "r", encoding="utf-8-sig", errors="replace") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return pd.read_csv(
            path_or_buf,
            sep=dialect.delimiter,
            engine="python",
            encoding="utf-8-sig",
            quotechar=dialect.quotechar or '"',
            doublequote=True,
            escapechar="\\",
            on_bad_lines="skip",
            skip_blank_lines=True,
        )

    return pd.read_csv(path_or_buf)

def read_table(path_or_bytes: Union[str, Path, bytes], *, suffix: Optional[str] = None) -> pd.DataFrame:
    if isinstance(path_or_bytes, (str, Path)):
        p = Path(path_or_bytes)
        sfx = p.suffix.lower()
        if sfx in {".csv", ".tsv"}:
            sep = "," if sfx == ".csv" else "\t"
            return _read_csv_robust(p, sep_hint=sep)
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
                df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
                return df
        raise ValueError(f"Unsupported file type: {sfx}")

    sfx = (suffix or "").lower()
    if sfx in {".csv", ".tsv"}:
        sep = "," if sfx == ".csv" else "\t"
        return _read_csv_robust(_io.BytesIO(path_or_bytes), sep_hint=sep)
    if sfx in {".parquet", ".pq"}:
        return pd.read_parquet(_io.BytesIO(path_or_bytes))
    if sfx in {".fits", ".fit"}:
        from astropy.io import fits
        from astropy.table import Table
        with fits.open(_io.BytesIO(path_or_bytes)) as hdul:
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

