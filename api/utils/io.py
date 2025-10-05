from __future__ import annotations
from typing import Optional, Union
import importlib
from pathlib import Path
import pandas as pd
import csv
import io as _io

def _read_csv_robust(path_or_buf, *, sep_hint: Optional[str] = None):
    """
    Read CSV with robust handling of NASA Exoplanet Archive files that contain comment headers.
    """
    # Common kwargs for all reading attempts
    common_kwargs = {
        "engine": "python",
        "encoding": "utf-8-sig",
        "quotechar": '"',
        "doublequote": True,
        "escapechar": "\\",
        "on_bad_lines": "skip",
        "skip_blank_lines": True,
        "comment": "#",  # Skip comment lines starting with #
    }
    
    if sep_hint in (",", "\t", ";", "|"):
        try:
            return pd.read_csv(path_or_buf, sep=sep_hint, **common_kwargs)
        except Exception:
            pass

    # Try automatic separator detection
    try:
        return pd.read_csv(path_or_buf, sep=None, **common_kwargs)
    except Exception:
        pass

    # Try common separators
    for s in (",", ";", "\t", "|"):
        try:
            return pd.read_csv(path_or_buf, sep=s, **common_kwargs)
        except Exception:
            continue

    # Fallback with dialect detection
    if isinstance(path_or_buf, (str, Path)):
        try:
            with open(path_or_buf, "r", encoding="utf-8-sig", errors="replace") as f:
                # Skip comment lines when detecting dialect
                lines = []
                for line in f:
                    if not line.strip().startswith("#") and line.strip():
                        lines.append(line)
                        if len(lines) >= 10:  # Use first 10 non-comment lines for detection
                            break
                
                if lines:
                    sample = "".join(lines)
                    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
                    return pd.read_csv(
                        path_or_buf,
                        sep=dialect.delimiter,
                        quotechar=dialect.quotechar or '"',
                        **common_kwargs
                    )
        except Exception:
            pass

    # Final fallback
    return pd.read_csv(path_or_buf, comment="#")

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

