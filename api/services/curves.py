from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

def load_lightcurve(
    src: Union[str, Path, bytes],
    *,
    suffix: Optional[str] = None,
    time_col_candidates=("time", "btjd", "bkjd", "t_bjd", "BJD", "TIME"),
    flux_col_candidates=("flux", "PDCSAP_FLUX", "SAP_FLUX", "flux_norm", "FLUX"),
) -> pd.DataFrame:
    df = _read_table(src, suffix=suffix)
    lower = {c.lower(): c for c in df.columns}
    t_col = _first_present(lower, time_col_candidates)
    f_col = _first_present(lower, flux_col_candidates)

    if t_col is None or f_col is None:
        raise ValueError("Could not find time/flux columns in the provided table.")

    out = pd.DataFrame({
        "time": pd.to_numeric(df[lower[t_col]], errors="coerce"),
        "flux": pd.to_numeric(df[lower[f_col]], errors="coerce"),
    }).dropna()

    mask = np.isfinite(out["time"].values) & np.isfinite(out["flux"].values)
    out = out.loc[mask]
    out = out.sort_values("time").reset_index(drop=True)
    return out

def preprocess_lightcurve(
    lc: pd.DataFrame,
    *,
    detrend: bool = True,
    clip_sigma: float = 4.0,
    normalize: bool = True,
    resample_len: int = 2048,
) -> np.ndarray:
    if lc.empty or "time" not in lc or "flux" not in lc:
        raise ValueError("Preprocess expects DataFrame with columns ['time','flux'].")

    t = lc["time"].values.astype(float)
    y = lc["flux"].values.astype(float)

    if clip_sigma and clip_sigma > 0:
        mu = np.nanmedian(y)
        sig = 1.4826 * np.nanmedian(np.abs(y - mu))  # robust MAD->sigma
        if np.isfinite(sig) and sig > 0:
            m = np.abs(y - mu) <= (clip_sigma * sig)
            t, y = t[m], y[m]

    if detrend:
        y = _detrend(y)

    if normalize:
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        if np.isfinite(ymin) and np.isfinite(ymax) and (ymax - ymin) > 0:
            y = (y - ymin) / (ymax - ymin)

    arr = _resample_to_fixed(t, y, resample_len=resample_len)
    return arr.astype(np.float32)

def fold_lightcurve(
    lc: pd.DataFrame,
    period_days: float,
    t0: Optional[float] = None,
    *,
    duration_hours: Optional[float] = None,
    window_factor: float = 3.0,
    resample_len: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    if period_days is None or period_days <= 0:
        raise ValueError("Positive period_days required for folding.")

    time = lc["time"].values.astype(float)
    flux = lc["flux"].values.astype(float)

    if t0 is None:
        t0 = np.nanmedian(time)

    phase = ((time - t0) / period_days) % 1.0
    phase[phase >= 0.5] -= 1.0  

    if duration_hours and duration_hours > 0:
        dur_frac = (duration_hours / 24.0) / period_days
        width = window_factor * dur_frac
        sel = (phase >= -width) & (phase <= width)
        phase, flux = phase[sel], flux[sel]

    idx = np.argsort(phase)
    phase, flux = phase[idx], flux[idx]

    ymin, ymax = np.nanmin(flux), np.nanmax(flux)
    if np.isfinite(ymin) and np.isfinite(ymax) and (ymax - ymin) > 0:
        flux = (flux - ymin) / (ymax - ymin)

    xg = np.linspace(-0.5, 0.5, resample_len, endpoint=False)
    yg = np.interp(xg, phase, flux, left=np.nan, right=np.nan)
    yg = _nanfix_1d(yg, fill_value=0.5)
    return xg.astype(np.float32), yg.astype(np.float32)

def prepare_curve_input(
    lc: pd.DataFrame,
    *,
    period_days: Optional[float] = None,
    duration_hours: Optional[float] = None,
    resample_len: int = 2048,
    fold_if_possible: bool = True,
) -> np.ndarray:
    if fold_if_possible and period_days and period_days > 0:
        _, y = fold_lightcurve(
            lc, period_days=period_days, t0=None,
            duration_hours=duration_hours, resample_len=resample_len
        )
        return y
    return preprocess_lightcurve(lc, resample_len=resample_len)

def guess_period_naive(lc: pd.DataFrame) -> Optional[float]:
    try:
        from astropy.timeseries import LombScargle  # type: ignore
    except Exception:
        log.info("astropy.timeseries not available; skip period guess.")
        return None

    t = lc["time"].values.astype(float)
    y = lc["flux"].values.astype(float)
    if len(t) < 10:
        return None

    baseline = np.nanmax(t) - np.nanmin(t)
    if not np.isfinite(baseline) or baseline <= 0:
        return None
    fmin = 1.0 / (baseline * 2.0)
    fmax = 24.0  
    freq = np.linspace(fmin, fmax, 5000)
    try:
        pwr = LombScargle(t, y).power(freq)
        best = freq[np.argmax(pwr)]
        period_days = 1.0 / best if best > 0 else None
        return float(period_days) if period_days else None
    except Exception as e:
        log.info("LombScargle failed: %s", e)
        return None

def _read_table(src: Union[str, Path, bytes], *, suffix: Optional[str]) -> pd.DataFrame:
    if isinstance(src, (str, Path)):
        p = Path(src)
        sfx = p.suffix.lower()
        if sfx in {".csv", ".tsv"}:
            sep = "," if sfx == ".csv" else "\t"
            return pd.read_csv(p, sep=sep, engine="python", low_memory=False)
        if sfx in {".fits", ".fit"}:
            try:
                from astropy.io import fits
                from astropy.table import Table
            except Exception as e:
                raise RuntimeError(f"FITS provided but astropy is not installed: {e}")
            with fits.open(str(p)) as hdul:
                h = next((h for h in hdul if getattr(h, "data", None) is not None), None)
                if h is None:
                    raise ValueError("No table HDU with data found in FITS.")
                tab = Table(h.data)
                df = tab.to_pandas()
                df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
                return df
        raise ValueError(f"Unsupported file type: {sfx}")

    # bytes-like
    sfx = (suffix or "").lower()
    if sfx in {".csv", ".tsv"}:
        sep = "," if sfx == ".csv" else "\t"
        return pd.read_csv(io.BytesIO(src), sep=sep, engine="python", low_memory=False)
    if sfx in {".fits", ".fit"}:
        try:
            from astropy.io import fits
            from astropy.table import Table
        except Exception as e:
            raise RuntimeError(f"FITS provided but astropy is not installed: {e}")
        with fits.open(io.BytesIO(src)) as hdul:
            h = next((h for h in hdul if getattr(h, "data", None) is not None), None)
            if h is None:
                raise ValueError("No table HDU with data found in FITS.")
            tab = Table(h.data)
            df = tab.to_pandas()
            df.columns = [c.decode() if isinstance(c, bytes) else c for c in df.columns]
            return df

    raise ValueError("Provide a valid suffix '.csv'|'.tsv'|'.fits' for bytes input.")


def _first_present(lower_map: Dict[str, str], cands) -> Optional[str]:
    for c in cands:
        if c.lower() in lower_map:
            return c.lower()
    return None


def _detrend(y: np.ndarray) -> np.ndarray:
    y = y.astype(float)
    if y.size < 9:
        return y - np.nanmedian(y)

    try:
        from scipy.signal import savgol_filter  
        win = max(9, (len(y) // 50) * 2 + 1)  
        win = min(win, len(y) - (1 - len(y) % 2))  
        poly = 2
        trend = savgol_filter(y, window_length=win, polyorder=poly, mode="interp")
        return y - trend + np.nanmedian(y)
    except Exception:
        pass

    # fallback
    k = max(9, len(y) // 50)
    if k % 2 == 0:
        k += 1
    if k >= len(y):
        return y - np.nanmedian(y)

    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    med = np.empty_like(y)
    for i in range(len(y)):
        med[i] = np.nanmedian(ypad[i:i + k])
    return y - med + np.nanmedian(y)


def _resample_to_fixed(t: np.ndarray, y: np.ndarray, *, resample_len: int) -> np.ndarray:
    if t.size == 0:
        return np.zeros(resample_len, dtype=float)
    t0, t1 = np.nanmin(t), np.nanmax(t)
    if not (np.isfinite(t0) and np.isfinite(t1)) or t1 <= t0:
        return np.nan_to_num(y[:resample_len], nan=np.nanmedian(y) if len(y) else 0.0)

    grid = np.linspace(t0, t1, resample_len)
    yg = np.interp(grid, t, y, left=np.nan, right=np.nan)
    yg = _nanfix_1d(yg, fill_value=np.nanmedian(yg[np.isfinite(yg)]) if np.any(np.isfinite(yg)) else 0.5)
    return yg


def _nanfix_1d(x: np.ndarray, fill_value: float = 0.5) -> np.ndarray:
    x = x.copy()
    n = len(x)
    # forward
    for i in range(1, n):
        if not np.isfinite(x[i]) and np.isfinite(x[i - 1]):
            x[i] = x[i - 1]
    # backward
    for i in range(n - 2, -1, -1):
        if not np.isfinite(x[i]) and np.isfinite(x[i + 1]):
            x[i] = x[i + 1]
    # final
    x[~np.isfinite(x)] = fill_value
    return x
