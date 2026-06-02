"""
data.py — API fetching & quality-control layer
================================================
All external data comes through here. Each function:
  1. Calls a real REST API
  2. Parses the JSON response into a DataFrame
  3. Applies basic QC flags

APIs used (all free, no authentication required):
  - Open-Meteo Marine API  : https://open-meteo.com/en/docs/marine-weather-api
  - Open-Meteo Archive API : https://open-meteo.com/en/docs/historical-weather-api
  - IFREMER ERDDAP         : https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.html
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)

# ── Cape Verde / OSCM study region ──────────────────────────────────────────
OSCM_LAT  =  16.88
OSCM_LON  = -24.99
BBOX      = dict(lat_min=12.0, lat_max=20.0, lon_min=-27.0, lon_max=-20.0)


# ── HTTP utility ─────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, timeout: int = 25) -> dict | None:
    """GET request with retry and structured error handling."""
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            log.warning(f"HTTP {r.status_code} from {url}")
            return None
        except requests.exceptions.RequestException as e:
            log.warning(f"Request error (attempt {attempt+1}): {e}")
            time.sleep(3)
    return None


# ── Quality control ──────────────────────────────────────────────────────────

def _qc_range(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Return boolean Series: True = value within valid range."""
    return s.between(lo, hi) | s.isna()

def _qc_spike(s: pd.Series, threshold: float) -> pd.Series:
    """Return boolean Series: True = not a spike (Argo QC Test 9 style)."""
    med = s.rolling(5, center=True, min_periods=1).median()
    return (s - med).abs().le(threshold) | s.isna()

def qc_marine(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "wave_height" in df.columns:
        df["wh_qc"] = (
            _qc_range(df["wave_height"], 0, 20) &
            _qc_spike(df["wave_height"], 3.0)
        ).map({True: 1, False: 4})   # Argo QC: 1=good, 4=bad
    if "wave_period" in df.columns:
        df["wp_qc"] = _qc_range(df["wave_period"], 1, 30).map({True: 1, False: 4})
    return df

def qc_argo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "temp" in df.columns:
        df["temp_qc"] = (
            _qc_range(df["temp"], -2, 35) &
            _qc_spike(df["temp"], 5.0)
        ).map({True: 1, False: 4})
    if "psal" in df.columns:
        df["psal_qc"] = _qc_range(df["psal"], 2, 42).map({True: 1, False: 4})
    return df


# ── API 1: Open-Meteo Marine ─────────────────────────────────────────────────

def fetch_marine(days_back: int = 30) -> pd.DataFrame:
    """
    Hourly wave & wind-wave variables for the OSCM point.
    Endpoint: https://marine-api.open-meteo.com/v1/marine
    """
    now   = datetime.now(timezone.utc)
    end   = now.strftime("%Y-%m-%d")
    start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

    data = _get("https://marine-api.open-meteo.com/v1/marine", params={
        "latitude":  OSCM_LAT,
        "longitude": OSCM_LON,
        "hourly":    "wave_height,wave_direction,wave_period,wind_wave_height",
        "start_date": start,
        "end_date":   end,
        "timezone":   "UTC",
    })

    if data is None:
        return _fallback_marine(days_back)

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return qc_marine(df), True


def _fallback_marine(days_back: int) -> pd.DataFrame:
    """Realistic synthetic data matching real API schema — used when API is unreachable."""
    now   = datetime.now(timezone.utc).replace(tzinfo=None)
    index = pd.date_range(end=now, periods=days_back * 24, freq="h")
    rng   = np.random.default_rng(42)
    t     = np.linspace(0, 4 * np.pi, len(index))

    df = pd.DataFrame({
        "wave_height":      np.clip(1.2 + 0.8*np.sin(t/6) + 0.4*rng.standard_normal(len(index)), 0, None),
        "wave_direction":   (180 + 40*np.sin(t/12) + 20*rng.standard_normal(len(index))) % 360,
        "wave_period":      np.clip(8 + 2*np.sin(t/8) + 0.5*rng.standard_normal(len(index)), 1, None),
        "wind_wave_height": np.clip(0.8 + 0.5*np.sin(t/4) + 0.3*rng.standard_normal(len(index)), 0, None),
    }, index=index)
    # Inject a few synthetic spikes for QC demo
    spike_idx = rng.integers(0, len(df), size=8)
    df.iloc[spike_idx, df.columns.get_loc("wave_height")] += rng.uniform(5, 10, size=8)
    return qc_marine(df), False


# ── API 2: Open-Meteo Archive (climate) ──────────────────────────────────────

def fetch_climate(days_back: int = 30) -> pd.DataFrame:
    """
    Daily temperature, precipitation, wind for the OSCM point.
    Endpoint: https://archive-api.open-meteo.com/v1/archive
    """
    now   = datetime.now(timezone.utc)
    end   = now.strftime("%Y-%m-%d")
    start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

    data = _get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude":  OSCM_LAT,
        "longitude": OSCM_LON,
        "daily":     "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "start_date": start,
        "end_date":   end,
        "timezone":   "UTC",
    })

    if data is None:
        return _fallback_climate(days_back)

    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    return df, True


def _fallback_climate(days_back: int) -> pd.DataFrame:
    now   = datetime.now(timezone.utc).replace(tzinfo=None)
    index = pd.date_range(end=now, periods=days_back, freq="D")
    rng   = np.random.default_rng(7)
    t     = np.linspace(0, 2 * np.pi, len(index))
    t_max = 27 + 2*np.sin(t) + 0.5*rng.standard_normal(len(index))
    t_min = t_max - 5 - rng.uniform(1, 3, len(index))
    df = pd.DataFrame({
        "temperature_2m_max": t_max,
        "temperature_2m_min": t_min,
        "precipitation_sum":  np.clip(rng.exponential(0.3, len(index)), 0, 20),
        "windspeed_10m_max":  np.clip(8 + 3*np.sin(t/2) + rng.standard_normal(len(index)), 0, None),
    }, index=index)
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    return df, False


# ── API 3: IFREMER ERDDAP — Argo float profiles ───────────────────────────

def fetch_argo(days_back: int = 90) -> pd.DataFrame:
    """
    Argo float T/S profiles from the Cape Verde region.
    Endpoint: https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json
    """
    now   = datetime.now(timezone.utc)
    end   = now.strftime("%Y-%m-%d")
    start = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")

    # ERDDAP requires variable list AND constraints in a single URL query string.
    # Passing constraints via requests params causes double-encoding (400 error).
    lat_min, lat_max = BBOX["lat_min"], BBOX["lat_max"]
    lon_min, lon_max = BBOX["lon_min"], BBOX["lon_max"]
    variables = "platform_number,time,latitude,longitude,pres,temp,psal"
    url = (
        "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.json"
        f"?{variables}"
        f"&latitude>={lat_min}&latitude<={lat_max}"
        f"&longitude>={lon_min}&longitude<={lon_max}"
        f"&time>={start}T00:00:00Z&time<={end}T23:59:59Z"
        f"&.limit=500&.orderBy=time"
    )
    data = _get(url)

    if data is None:
        return _fallback_argo(days_back)

    table = data.get("table", {})
    if not table.get("rows"):
        return _fallback_argo(days_back)

    df = pd.DataFrame(table["rows"], columns=table["columnNames"])
    df["time"] = pd.to_datetime(df["time"])
    for col in ["latitude", "longitude", "pres", "temp", "psal"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return qc_argo(df), True


def _fallback_argo(days_back: int) -> pd.DataFrame:
    """Synthetic Argo profiles: 6 floats, ~5 profiles each, 50 depth levels."""
    rng          = np.random.default_rng(13)
    float_ids    = [f"39{1000+i}" for i in range(6)]
    now          = datetime.now(timezone.utc).replace(tzinfo=None)
    records      = []
    pressure_levels = np.array([0,5,10,20,30,50,75,100,150,200,300,400,500,
                                  600,700,800,900,1000,1200,1500,1800,2000])

    for fid in float_ids:
        base_lat = rng.uniform(BBOX["lat_min"], BBOX["lat_max"])
        base_lon = rng.uniform(BBOX["lon_min"], BBOX["lon_max"])
        n_profiles = rng.integers(4, 7)
        for p in range(n_profiles):
            t_obs = now - timedelta(days=int(rng.integers(1, days_back)))
            lat   = base_lat + rng.uniform(-0.5, 0.5)
            lon   = base_lon + rng.uniform(-0.5, 0.5)
            for pres in pressure_levels:
                # Realistic T/S profiles: warm salty surface, cold fresh deep
                temp  = 26 - 0.012 * pres + rng.normal(0, 0.3)
                psal  = 36.5 + 0.8 * np.exp(-pres/100) - 0.001*pres + rng.normal(0, 0.05)
                records.append({
                    "platform_number": fid,
                    "time": t_obs,
                    "latitude": round(lat, 4),
                    "longitude": round(lon, 4),
                    "pres": float(pres),
                    "temp": round(float(temp), 3),
                    "psal": round(float(psal), 3),
                })

    df = pd.DataFrame(records)
    return qc_argo(df), False
