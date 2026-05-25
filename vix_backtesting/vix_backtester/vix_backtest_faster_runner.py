import os
import math
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
from time import perf_counter

import numpy as np
import pandas as pd

import vix_backtest_config as cfg
from data_preprocessing_file import preprocess_index_data

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1600)


# ============================================================
# OPTION PRICE FILL + NEAREST-STRIKE FALLBACK
# All four values are read from vix_backtest_config.py
# ============================================================
OPTION_PRICE_FILL_METHOD = getattr(cfg, "OPTION_PRICE_FILL_METHOD", "ffill+bfill")
OPTION_PRICE_FILL_LIMIT = getattr(cfg, "OPTION_PRICE_FILL_LIMIT", None)
NEAREST_STRIKE_FALLBACK_ENABLED = getattr(cfg, "NEAREST_STRIKE_FALLBACK_ENABLED", True)
MAX_STRIKE_FALLBACK_STEPS = getattr(cfg, "MAX_STRIKE_FALLBACK_STEPS", 5)


# ============================================================
# HELPERS
# ============================================================
def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def safe_int(x):
    try:
        if pd.isna(x):
            return np.nan
        return int(float(x))
    except Exception:
        return np.nan


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


# ── fast path: if the string already looks like YYYY-MM-DD, skip pd.to_datetime
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def normalize_date_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if len(s) >= 10 and _DATE_RE.match(s[:10]):
        return s[:10]
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_date_str_dayfirst(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if len(s) >= 10 and _DATE_RE.match(s[:10]):
        return s[:10]
    try:
        ts = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.strftime("%Y-%m-%d")
    except Exception:
        return None


_TIME_RE_HM  = re.compile(r"^\d{2}:\d{2}$")
_TIME_RE_HMS = re.compile(r"^\d{2}:\d{2}:\d{2}$")

def normalize_time_str(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if _TIME_RE_HM.match(s):
        return s
    if _TIME_RE_HMS.match(s):
        return s[:5]
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.strftime("%H:%M")
    except Exception:
        return None


def combine_date_time(date_val, time_val):
    d = normalize_date_str(date_val)
    t = normalize_time_str(time_val)
    if d is None or t is None:
        return pd.NaT
    return pd.to_datetime(f"{d} {t}", errors="coerce")


def is_on_or_after_time(ts, cutoff_time_str):
    if pd.isna(ts):
        return False
    ts = pd.Timestamp(ts)
    cutoff = normalize_time_str(cutoff_time_str)
    if cutoff is None:
        return False
    return ts.strftime("%H:%M") >= cutoff


def get_forced_option_price(requested_ts, expiry_date):
    """
    Force a non-null option price when all bar/price resolution paths fail.
    """
    if not getattr(cfg, "FORCE_OPTION_PRICE_IF_MISSING", False):
        return np.nan

    req_ts = pd.Timestamp(requested_ts) if not pd.isna(requested_ts) else pd.NaT
    expiry_str = normalize_date_str(expiry_date)
    req_day = req_ts.strftime("%Y-%m-%d") if not pd.isna(req_ts) else None

    if (req_day == expiry_str) and is_on_or_after_time(req_ts, getattr(cfg, "FORCED_EXPIRY_CUTOFF_TIME", "15:15")):
        return safe_float(getattr(cfg, "FORCED_EXPIRY_PRICE", 0.0))
    return safe_float(getattr(cfg, "FORCED_NON_EXPIRY_PRICE", 0.05))


def minutes_diff(ts1, ts2):
    if pd.isna(ts1) or pd.isna(ts2):
        return np.nan
    return abs((pd.Timestamp(ts1) - pd.Timestamp(ts2)).total_seconds() / 60.0)


def parse_timeframe_to_minutes(tf):
    s = str(tf).strip().lower()
    if s.endswith("min"):
        return int(s[:-3])
    if s.endswith("m"):
        return int(s[:-1])
    raise ValueError(f"Unsupported timeframe format: {tf}")


def round_to_step(value, step):
    if pd.isna(value):
        return np.nan
    return int(round(float(value) / step) * step)


def round_to_50(value):
    return round_to_step(value, cfg.STRIKE_STEP)


def round_to_step_directional(value, step, direction):
    """
    Round to nearest step, but if exactly at midpoint:
    - CE / UP   -> round upward
    - PE / DOWN -> round downward

    direction accepted:
    - "UP"   : midpoint goes upward
    - "DOWN" : midpoint goes downward
    """
    if pd.isna(value):
        return np.nan

    x = float(value)
    step = float(step)
    q = x / step

    lower = math.floor(q) * step
    upper = math.ceil(q) * step

    dist_lower = abs(x - lower)
    dist_upper = abs(upper - x)

    eps = 1e-12
    if abs(dist_lower - dist_upper) <= eps:
        return int(upper if direction == "UP" else lower)

    return int(lower if dist_lower < dist_upper else upper)


def round_to_50_up_on_mid(value):
    return round_to_step_directional(value, cfg.STRIKE_STEP, "UP")


def round_to_50_down_on_mid(value):
    return round_to_step_directional(value, cfg.STRIKE_STEP, "DOWN")


def round_series_to_step_directional(values, step, direction):
    """
    Vectorised directional rounding for a pandas Series / numpy array.
    Midpoint rule:
    - direction='UP'   -> midpoint rounds upward
    - direction='DOWN' -> midpoint rounds downward
    """
    s = pd.to_numeric(values, errors="coerce")
    arr = s.to_numpy(dtype="float64", copy=False)
    out = np.full(arr.shape, np.nan, dtype="float64")

    valid = ~np.isnan(arr)
    if not valid.any():
        return pd.Series(out, index=s.index)

    x = arr[valid]
    step = float(step)
    q = x / step
    lower = np.floor(q) * step
    upper = np.ceil(q) * step
    dist_lower = np.abs(x - lower)
    dist_upper = np.abs(upper - x)
    eps = 1e-12

    midpoint = np.abs(dist_lower - dist_upper) <= eps
    choose_upper = dist_upper < dist_lower
    choose_lower = dist_lower < dist_upper

    rounded = np.where(
        midpoint,
        upper if direction == "UP" else lower,
        np.where(choose_lower, lower, upper),
    )
    out[valid] = rounded
    return pd.Series(out, index=s.index)


def make_trade_id(row_idx, entry_ts):
    ts = pd.Timestamp(entry_ts)
    return f"T_{ts.strftime('%Y%m%d_%H%M')}_{int(row_idx):07d}"


def normalize_option_type(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if s in {"CE", "CALL", "C"}:
        return "CE"
    if s in {"PE", "PUT", "P"}:
        return "PE"
    return s


def make_contract_key(trade_day, expiry_date, strike, option_type):
    d = normalize_date_str(trade_day)
    e = normalize_date_str(expiry_date)
    ot = normalize_option_type(option_type)
    strike_int = safe_int(strike)
    if d is None or e is None or ot is None or pd.isna(strike_int):
        return None
    return f"{d}|{e}|{ot}|{strike_int}"


def load_table_auto(path):
    p = str(path).lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    if p.endswith(".parquet"):
        return pd.read_parquet(path)
    if p.endswith(".xlsx") or p.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file: {path}")


# ============================================================
# INDEX SIGNAL BARS
# ============================================================
def build_index_signal_bars(index_path, timeframe, manual_entry_times=None, restrict_to_idx=False):
    raw = load_table_auto(index_path)
    raw = normalize_column_names(raw)

    bars = preprocess_index_data(
        df=raw,
        timeframe=timeframe,
        keep_partial_bars=True,
        filter_type_idx_only=restrict_to_idx,
        default_session_start=cfg.SESSION_START,
    )

    if bars.empty:
        return pd.DataFrame(columns=[
            "timestamp", "trade_day", "entry_ts", "entry_time",
            "Open", "High", "Low", "Close", "spot_open"
        ])

    bars = bars.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], errors="coerce")
    bars = bars.dropna(subset=["timestamp"]).copy()

    if "trade_date" in bars.columns:
        bars["trade_day"] = pd.to_datetime(bars["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        bars["trade_day"] = bars["timestamp"].dt.strftime("%Y-%m-%d")

    bars["entry_ts"] = bars["timestamp"]
    bars["entry_time"] = bars["timestamp"].dt.strftime("%H:%M")
    bars["spot_open"] = pd.to_numeric(bars["Open"], errors="coerce")

    bars = bars[(bars["trade_day"] >= cfg.START_DATE) & (bars["trade_day"] <= cfg.END_DATE)].copy()

    if manual_entry_times is not None:
        bars = bars[bars["entry_time"].isin(set(manual_entry_times))].copy()

    bars = bars.sort_values(["entry_ts"]).reset_index(drop=True)
    return bars


# EXPIRY / DTE PREP
# ============================================================
def standardize_expiry_columns(df):
    out = df.copy()
    rename_map = {}
    col_lookup = {c.lower(): c for c in out.columns}
    candidate_map = {
        "date": ["date", "tradedate", "trade_date"],
        "expirydate": ["expirydate", "expiry_date", "expiry"],
        "day": ["day", "weekday"],
        "dte": ["dte", "days_to_expiry", "daysleft", "days_left"],
    }
    target = {"date": "Date", "expirydate": "ExpiryDate", "day": "Day", "dte": "DTE"}
    for logical, candidates in candidate_map.items():
        for cand in candidates:
            if cand in col_lookup:
                rename_map[col_lookup[cand]] = target[logical]
                break
    return out.rename(columns=rename_map)


def prepare_expiry_map(raw_df):
    df = normalize_column_names(raw_df)
    df = standardize_expiry_columns(df)
    missing = [c for c in ["Date", "ExpiryDate", "DTE"] if c not in df.columns]
    if missing:
        raise ValueError(f"Expiry map missing columns: {missing}")
    df["Date"] = df["Date"].map(normalize_date_str_dayfirst)
    df["ExpiryDate"] = df["ExpiryDate"].map(normalize_date_str_dayfirst)
    if "Day" in df.columns:
        df["Day"] = df["Day"].astype(str).str.strip()
    df["DTE"] = pd.to_numeric(df["DTE"], errors="coerce")
    df = df.dropna(subset=["Date", "ExpiryDate", "DTE"]).copy()
    df = df[(df["Date"] >= cfg.START_DATE) & (df["Date"] <= cfg.END_DATE)].copy()
    df["trade_day"] = df["Date"]
    df["expiry_date"] = df["ExpiryDate"]
    df["dte"] = df["DTE"].astype(int)
    keep = ["trade_day", "expiry_date", "dte"]
    if "Day" in df.columns:
        keep.append("Day")
    df = df[keep].copy()
    # Guard: warn loudly if any trade_day maps to more than one expiry
    dup_days = df[df.duplicated(subset=["trade_day"], keep=False)]["trade_day"].unique()
    if len(dup_days) > 0:
        import warnings
        warnings.warn(
            f"prepare_expiry_map: {len(dup_days)} trade_day(s) have duplicate expiry rows "
            f"— keeping first occurrence. Duplicates: {sorted(dup_days[:10].tolist())}",
            stacklevel=2,
        )
    df = df.drop_duplicates(subset=["trade_day"], keep="first").sort_values(["trade_day"]).reset_index(drop=True)
    return df


def attach_expiry_and_dte(signal_df, expiry_df):
    merge_cols = ["trade_day", "expiry_date", "dte"] + (["Day"] if "Day" in expiry_df.columns else [])
    return signal_df.merge(expiry_df[merge_cols], on="trade_day", how="left")


# ============================================================
# SIGNAL PANEL
# ============================================================
def build_signal_panel(index_signal_df, expiry_df):
    out = attach_expiry_and_dte(index_signal_df, expiry_df)
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], errors="coerce")
    out["trade_day"] = out["trade_day"].map(normalize_date_str)
    out["entry_time"] = out["entry_time"].map(normalize_time_str)
    out["expiry_date"] = out["expiry_date"].map(normalize_date_str)
    out = out.dropna(subset=["trade_day", "entry_ts", "entry_time", "spot_open", "expiry_date", "dte"]).copy()
    out["dte"] = pd.to_numeric(out["dte"], errors="coerce")
    out = out.dropna(subset=["dte"]).copy()
    out = out[out["dte"] >= 0].copy()
    out["spot_open"] = pd.to_numeric(out["spot_open"], errors="coerce")
    out = out.dropna(subset=["spot_open"]).copy()
    out = out[out["spot_open"] > 0].copy()
    excluded_expiries = {
        normalize_date_str(x)
        for x in getattr(cfg, "EXCLUDED_EXPIRIES", [])
        if normalize_date_str(x) is not None
    }
    if excluded_expiries:
        out = out[~out["expiry_date"].isin(excluded_expiries)].copy()
    out = out.sort_values(["entry_ts"]).reset_index(drop=True)
    out["signal_row_id"] = np.arange(len(out))
    return out


# ============================================================
# FEATURE SHEET
# ============================================================
def load_feature_workbook_auto(path):
    """
    Auto-pick the feature sheet from the workbook.
    Preference order:
    1) features_calculations
    2) features_standardised
    3) first sheet
    """
    xls = pd.ExcelFile(path)
    sheet_names = list(xls.sheet_names)

    preferred_order = ["features_calculations", "features_standardised"]
    chosen_sheet = None
    lower_map = {str(s).strip().lower(): s for s in sheet_names}

    for pref in preferred_order:
        if pref.lower() in lower_map:
            chosen_sheet = lower_map[pref.lower()]
            break

    if chosen_sheet is None:
        chosen_sheet = sheet_names[0]

    return pd.read_excel(path, sheet_name=chosen_sheet)


def standardize_feature_columns(df):
    """
    Adapt feature workbook format to runner's expected internal names.
    Supports both:
    - features_calculations
    - features_standardised
    """
    out = df.copy()
    rename_map = {}
    col_lookup = {str(c).strip().lower(): c for c in out.columns}

    candidate_map = {
        "trade_day": ["trade_day", "trade_date", "date"],
        "feature_ts": ["feature_ts", "timestamp", "datetime", "ts"],
        "feature_time": ["feature_time", "time"],

        "DTE": ["dte"],
        "rv_fast": ["rv_fast", "rv fast", "rv_fast_1/2", "rv fast 1/2"],
        "rv_slow": ["rv_slow", "rv slow"],
        "iv": ["iv"],

        "dte_std": ["dte_std", "dte std"],
        "one_std": ["1_std", "one_std", "1 std", "one std"],

        "gap": ["gap"],
        "gap_dur": ["gap_dur", "gap dur"],
        "gap_avg_5d": ["gap_avg_5d", "gap avg 5d"],

        "orb": ["orb", "orb_60m_range", "orb 60m range"],
        "orb_1_std": ["orb_1_std", "orb_60m_range_1_std", "orb 60m range 1 std"],

        "ivp_6m": ["ivp_6m", "ivp 6m"],
        "ivp_4m": ["ivp_4m", "ivp 4m"],
        "ivp_12m": ["ivp_12m", "ivp 12m"],

        "skew": ["skew"],
        "skew_10d": ["skew_10d", "skew 10d"],

        "call_iv_slope": ["call_iv_slope", "call iv slope"],
        "put_iv_slope": ["put_iv_slope", "put iv slope"],

        "rvs_iv": ["rv_slow_over_iv", "rv slow over iv"],
        "rvs_minus_iv": ["rv_slow_minus_iv", "rv slow minus iv"],
        "rv_slow_over_rv_fast_1_3": [
            "rv_slow_over_rv_fast_1/3",
            "rv slow over rv fast 1/3",
            "rv_slow_over_rv_fast_1_3",
        ],

        "move_to_open_1_std": ["move_to_open_1_std", "move to open 1 std"],
        "ema_15_distance_1_std": ["ema_15_distance_1_std"],
        "ema_15_distance_dte_std": ["ema_15_distance_dte_std"],
        "ema_30_distance_1_std": ["ema_30_distance_1_std"],
        "ema_30_distance_dte_std": ["ema_30_distance_dte_std"],
    }

    for target, candidates in candidate_map.items():
        for cand in candidates:
            if cand in col_lookup:
                rename_map[col_lookup[cand]] = target
                break

    out = out.rename(columns=rename_map)

    if "feature_time" not in out.columns and "feature_ts" in out.columns:
        ts = pd.to_datetime(out["feature_ts"], errors="coerce")
        out["feature_time"] = ts.dt.strftime("%H:%M")

    return out


def load_feature_sheet(path):
    feat = load_feature_workbook_auto(path)
    feat = normalize_column_names(feat)
    feat = standardize_feature_columns(feat)

    missing = [c for c in ["trade_day", "feature_ts", "feature_time"] if c not in feat.columns]
    if missing:
        raise ValueError(
            f"Feature sheet missing required columns after normalization: {missing}. "
            f"Available columns: {list(feat.columns)}"
        )

    feat["trade_day"] = pd.to_datetime(feat["trade_day"], errors="coerce").dt.strftime("%Y-%m-%d")
    feat["feature_ts"] = pd.to_datetime(feat["feature_ts"], errors="coerce")
    feat["feature_time"] = feat["feature_time"].astype(str).str.strip().str.slice(0, 5)

    numeric_cols = [
        "DTE", "rv_fast", "rv_slow", "iv",
        "dte_std", "one_std",
        "gap", "gap_dur", "gap_avg_5d",
        "orb", "orb_1_std",
        "ivp_6m", "ivp_4m", "ivp_12m",
        "skew", "skew_10d",
        "call_iv_slope", "put_iv_slope",
        "rvs_iv", "rvs_minus_iv",
        "rv_slow_over_rv_fast_1_3",
        "move_to_open_1_std",
        "ema_15_distance_1_std", "ema_15_distance_dte_std",
        "ema_30_distance_1_std", "ema_30_distance_dte_std",
    ]

    for col in numeric_cols:
        if col in feat.columns:
            feat[col] = pd.to_numeric(feat[col], errors="coerce")

    feat = feat.sort_values(["trade_day", "feature_ts"]).reset_index(drop=True)
    return feat

def attach_strike_features_to_signals(signal_panel, feature_sheet_path):
    """
    Attach base_vix feature columns to signal rows.

    Used by ENTRY_MODE="base_vix".

    This function does NOT apply any entry filter.

    Required for strike construction:
    - dte_std
    - one_std

    Required for REGIME_DRIVEN_ENTRY=True:
    - rv_slow
    - iv
    - move_to_open_1_std

    Important:
    load_feature_sheet() usually loads features_calculations first.
    features_calculations may contain raw move_to_open, while regime classifier
    needs move_to_open_1_std, which is usually in features_standardised.
    Therefore, when regime columns are missing, this function also loads
    features_standardised and merges only the missing columns.
    """
    # ------------------------------------------------------------
    # Step 1: Load default feature sheet.
    # This is usually features_calculations.
    # ------------------------------------------------------------
    feat = load_feature_sheet(feature_sheet_path)

    required_strike_cols = ["dte_std", "one_std"]
    missing_strike_cols = [c for c in required_strike_cols if c not in feat.columns]
    if missing_strike_cols:
        raise ValueError(
            f"Feature sheet must contain strike-construction columns {missing_strike_cols}. "
            f"Available columns: {list(feat.columns)}"
        )

    regime_cols = ["rv_slow", "iv", "move_to_open_1_std"]

    needed_cols = [
        "trade_day",
        "feature_time",
        "feature_ts",
        "dte_std",
        "one_std",
    ]

    needed_cols += [c for c in regime_cols if c in feat.columns]

    feat_join = feat[needed_cols].copy()

    out = signal_panel.copy().merge(
        feat_join,
        left_on=["trade_day", "entry_time", "entry_ts"],
        right_on=["trade_day", "feature_time", "feature_ts"],
        how="left",
    )

    # ------------------------------------------------------------
    # Step 2: If regime-driven entry is ON and any regime input is
    # missing, load features_standardised and merge missing columns.
    # This does not filter any rows.
    # ------------------------------------------------------------
    if getattr(cfg, "REGIME_DRIVEN_ENTRY", False):
        missing_regime_cols = [c for c in regime_cols if c not in out.columns]

        if missing_regime_cols:
            std_feat = load_features_standardised_sheet(feature_sheet_path).copy()

            # Convert features_standardised internal names to the same merge
            # names used by signal_panel/base_vix.
            std_feat = std_feat.rename(
                columns={
                    "tradeday": "trade_day",
                    "featuretime": "feature_time",
                    "featurets": "feature_ts",
                }
            )

            available_missing_cols = [c for c in missing_regime_cols if c in std_feat.columns]

            if available_missing_cols:
                std_join_cols = [
                    "trade_day",
                    "feature_time",
                    "feature_ts",
                ] + available_missing_cols

                std_join = std_feat[std_join_cols].copy()

                out = out.merge(
                    std_join,
                    left_on=["trade_day", "entry_time", "entry_ts"],
                    right_on=["trade_day", "feature_time", "feature_ts"],
                    how="left",
                    suffixes=("", "_std"),
                )

                # Clean helper columns created by second merge.
                for helper_col in ["feature_time_std", "feature_ts_std"]:
                    if helper_col in out.columns:
                        out = out.drop(columns=[helper_col])

    # ------------------------------------------------------------
    # Step 3: Numeric cleanup.
    # ------------------------------------------------------------
    for col in ["dte_std", "one_std", "rv_slow", "iv", "move_to_open_1_std"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def _compact_feature_col_name(col):
    return re.sub(r"[^a-z0-9]+", "", str(col).strip().lower())


def load_features_standardised_sheet(path):
    """
    Load the features_standardised sheet for feature_filter_v1 only.

    Returned columns use the exact internal names required by this mode:
    tradeday, featurets, featuretime, gap, orb_60m_range_1_std,
    strike inputs dte_std/one_std when present, and optional DTE.
    """
    xls = pd.ExcelFile(path)
    sheet_lookup = {str(s).strip().lower(): s for s in xls.sheet_names}
    if "features_standardised" not in sheet_lookup:
        raise ValueError(
            "Feature workbook must contain sheet 'features_standardised' for feature_filter_v1. "
            f"Available sheets: {list(xls.sheet_names)}"
        )

    feat = pd.read_excel(path, sheet_name=sheet_lookup["features_standardised"])
    feat = normalize_column_names(feat)

    alias_map = {
        "tradeday": {"tradeday", "tradedate", "date"},
        "featurets": {"featurets", "timestamp", "datetime", "ts"},
        "featuretime": {"featuretime", "time"},
        "gap": {"gap"},
        "orb_60m_range_1_std": {
            "orb60mrange1std",
            "orb1std",
            "orb60mrange1standarddeviation",
        },
        "dte_std": {"dtestd"},
        "one_std": {"1std", "onestd"},
        "DTE": {"dte"},
        # Patch 3: regime classifier inputs (optional — no error if absent)
        "rv_slow": {"rvslow", "rv_slow"},
        "iv": {"iv"},
        "move_to_open_1_std": {"movetoopen1std", "move_to_open_1_std", "movetoopen1standarddeviation"},
        "rv_slow_over_rv_fast_1_3": {
            "rvslowoverrvfast13",
            "rvslowoverrvfast1_3",
            "rvslowoverrvfast1/3",
            "rv_slow_over_rv_fast_1_3",
            "rv_slow_over_rv_fast_1/3",
        },
    }

    rename_map = {}
    compact_lookup = {_compact_feature_col_name(c): c for c in feat.columns}
    for target, aliases in alias_map.items():
        for alias in aliases:
            if alias in compact_lookup:
                rename_map[compact_lookup[alias]] = target
                break

    feat = feat.rename(columns=rename_map)
    required = ["tradeday", "featurets", "gap", "orb_60m_range_1_std"]
    missing = [c for c in required if c not in feat.columns]
    if missing:
        raise ValueError(
            "features_standardised sheet missing required columns after normalization: "
            f"{missing}. Available columns: {list(feat.columns)}"
        )

    feat["tradeday"] = pd.to_datetime(feat["tradeday"], errors="coerce").dt.strftime("%Y-%m-%d")
    feat["featurets"] = pd.to_datetime(feat["featurets"], errors="coerce")
    if "featuretime" in feat.columns:
        feat["featuretime"] = feat["featuretime"].map(normalize_time_str)
    else:
        feat["featuretime"] = feat["featurets"].dt.strftime("%H:%M")

    feat["gap"] = pd.to_numeric(feat["gap"], errors="coerce")
    feat["orb_60m_range_1_std"] = pd.to_numeric(feat["orb_60m_range_1_std"], errors="coerce")
    for col in ["dte_std", "one_std"]:
        if col in feat.columns:
            feat[col] = pd.to_numeric(feat[col], errors="coerce")
    if "DTE" in feat.columns:
        feat["DTE"] = pd.to_numeric(feat["DTE"], errors="coerce")

    keep_cols = ["tradeday", "featurets", "featuretime", "gap", "orb_60m_range_1_std"]
    keep_cols.extend([c for c in ["dte_std", "one_std"] if c in feat.columns])
    if "DTE" in feat.columns:
        keep_cols.append("DTE")
    # Patch 3: carry regime inputs if present in this sheet
    for _rc_col in ["rv_slow", "iv", "move_to_open_1_std"]:
        if _rc_col in feat.columns:
            feat[_rc_col] = pd.to_numeric(feat[_rc_col], errors="coerce")
            keep_cols.append(_rc_col)

    # feature_filter_v2 input
    if "rv_slow_over_rv_fast_1_3" in feat.columns:
        feat["rv_slow_over_rv_fast_1_3"] = pd.to_numeric(
            feat["rv_slow_over_rv_fast_1_3"],
            errors="coerce",
        )
        keep_cols.append("rv_slow_over_rv_fast_1_3")
    return feat[keep_cols].sort_values(["tradeday", "featurets"]).reset_index(drop=True)


def feature_filter_v1_block_gap(v):
    return (v is None) or pd.isna(v) or (v < -0.4) or (v > 0.4)


def feature_filter_v1_block_orb(v):
    return (v is None) or pd.isna(v) or (v > 0.65)


def feature_filter_v2_block_rvs_rvf(v):
    """
    feature_filter_v2 no-trade rule:
    Block when rv_slow_over_rv_fast_1_3 is missing or below 0.9006.
    Allowed range: rv_slow_over_rv_fast_1_3 >= 0.9006
    """
    return (v is None) or pd.isna(v) or (v < 0.9006)


def feature_filter_v1_time_allowed(entrytime, dte):
    t = normalize_time_str(entrytime)
    if t is None:
        return False
    if t < getattr(cfg, "FEATURE_FILTER_V1_SCAN_START", "09:30"):
        return False
    if t < getattr(cfg, "FEATURE_FILTER_V1_ORB_EARLIEST", "10:15"):
        return False
    dte_val = safe_float(dte)
    if not pd.isna(dte_val) and dte_val == 0 and t > getattr(cfg, "FEATURE_FILTER_V1_DTE0_LAST_ENTRY", "10:00"):
        return False
    return True


def apply_entry_mode(signal_panel, entry_mode, feature_sheet_path=None):
    if entry_mode == "base_vix":
        out = signal_panel.copy()
        if feature_sheet_path is not None:
            out = attach_strike_features_to_signals(out, feature_sheet_path)
        out = out.reset_index(drop=True)
        out["signal_row_id"] = np.arange(len(out))
        return out

    if entry_mode == "feature_filter_v1":
        feat = load_features_standardised_sheet(feature_sheet_path)
        required_cols = ["tradeday", "featuretime", "featurets", "gap", "orb_60m_range_1_std"]
        missing_feature_cols = [c for c in required_cols if c not in feat.columns]
        if missing_feature_cols:
            raise ValueError(
                "feature_filter_v1 requires features_standardised columns missing after normalization: "
                f"{missing_feature_cols}. Available columns: {list(feat.columns)}"
            )
        strike_cols = ["dte_std", "one_std"]
        missing_strike_cols = [c for c in strike_cols if c not in feat.columns]
        if missing_strike_cols:
            raise ValueError(
                "feature_filter_v1 requires strike-construction columns from features_standardised: "
                f"{missing_strike_cols}. Available columns: {list(feat.columns)}"
            )
        feature_join_cols = required_cols + strike_cols + (["DTE"] if "DTE" in feat.columns else [])
        feature_join_cols += [c for c in ["rv_slow", "iv", "move_to_open_1_std"] if c in feat.columns]
        feat_join = feat[feature_join_cols].copy()
        out = signal_panel.copy().merge(
            feat_join,
            left_on=["trade_day", "entry_time", "entry_ts"],
            right_on=["tradeday", "featuretime", "featurets"],
            how="left",
        )
        out["gap_blocked"] = out["gap"].apply(feature_filter_v1_block_gap)
        out["orb_blocked"] = out["orb_60m_range_1_std"].apply(feature_filter_v1_block_orb)
        out["feature_filter_time_allowed"] = [
            feature_filter_v1_time_allowed(et, d)
            for et, d in zip(out["entry_time"], out["dte"])
        ]
        out["feature_filter_allowed"] = (
            (~out["gap_blocked"]) &
            (~out["orb_blocked"]) &
            (out["feature_filter_time_allowed"])
        )
        out = out[out["feature_filter_allowed"]].copy()
        out = out.reset_index(drop=True)
        out["signal_row_id"] = np.arange(len(out))
        return out

    if entry_mode == "feature_filter_v2":
        feat = load_features_standardised_sheet(feature_sheet_path)

        required_cols = [
            "tradeday",
            "featuretime",
            "featurets",
            "gap",
            "rv_slow_over_rv_fast_1_3",
        ]
        missing_feature_cols = [c for c in required_cols if c not in feat.columns]
        if missing_feature_cols:
            raise ValueError(
                "feature_filter_v2 requires features_standardised columns missing after normalization: "
                f"{missing_feature_cols}. Available columns: {list(feat.columns)}"
            )

        strike_cols = ["dte_std", "one_std"]
        missing_strike_cols = [c for c in strike_cols if c not in feat.columns]
        if missing_strike_cols:
            raise ValueError(
                "feature_filter_v2 requires strike-construction columns from features_standardised: "
                f"{missing_strike_cols}. Available columns: {list(feat.columns)}"
            )

        feature_join_cols = required_cols + strike_cols + (["DTE"] if "DTE" in feat.columns else [])
        feature_join_cols += [
            c for c in ["rv_slow", "iv", "move_to_open_1_std"]
            if c in feat.columns
        ]

        feat_join = feat[feature_join_cols].copy()

        out = signal_panel.copy().merge(
            feat_join,
            left_on=["trade_day", "entry_time", "entry_ts"],
            right_on=["tradeday", "featuretime", "featurets"],
            how="left",
        )

        out["gap_blocked"] = out["gap"].apply(feature_filter_v1_block_gap)
        out["rvs_rvf_blocked"] = out["rv_slow_over_rv_fast_1_3"].apply(
            feature_filter_v2_block_rvs_rvf
        )

        out["feature_filter_time_allowed"] = [
            feature_filter_v1_time_allowed(et, d)
            for et, d in zip(out["entry_time"], out["dte"])
        ]

        out["feature_filter_allowed"] = (
            (~out["gap_blocked"]) &
            (~out["rvs_rvf_blocked"]) &
            (out["feature_filter_time_allowed"])
        )

        out = out[out["feature_filter_allowed"]].copy()
        out = out.reset_index(drop=True)
        out["signal_row_id"] = np.arange(len(out))
        return out

    raise ValueError(f"Unsupported ENTRY_MODE: {entry_mode}")


# ============================================================
# PATCH 3 — TIME-WINDOW DEDUP + REGIME-DRIVEN ENTRY
# ============================================================

def apply_time_window_dedup(signal_panel_selected):
    """
    If TIME_WINDOWS is defined in config, keep only the FIRST qualifying
    signal per window per day.  The regime of that first signal determines
    the trade structure and SL for the window entry.

    If TIME_WINDOWS is not defined, fall back to the old bucket-dedup
    logic controlled by ENTRY_WINDOW_MINUTES (or no dedup if None).
    """
    time_windows = getattr(cfg, "TIME_WINDOWS", None)

    # New path: named time windows.
    if time_windows:
        df = signal_panel_selected.copy()
        df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
        df["_entry_time"] = df["entry_ts"].dt.strftime("%H:%M")
        df = df.sort_values("entry_ts")

        kept_rows = []
        for trade_day, day_group in df.groupby("trade_day", sort=False):
            seen_windows = set()
            for _, row in day_group.sort_values("entry_ts").iterrows():
                t = row["_entry_time"]
                for w_label, w_start, w_end in time_windows:
                    if w_label in seen_windows:
                        continue
                    if w_start <= t <= w_end:
                        kept_rows.append(row)
                        seen_windows.add(w_label)
                        break

        if not kept_rows:
            return signal_panel_selected.iloc[0:0].reset_index(drop=True)

        out = pd.DataFrame(kept_rows).drop(columns=["_entry_time"])
        out = out.reset_index(drop=True)
        out["signal_row_id"] = np.arange(len(out))
        return out

    # Legacy path: bucket dedup via ENTRY_WINDOW_MINUTES.
    window_min = getattr(cfg, "ENTRY_WINDOW_MINUTES", None)
    if not window_min:
        return signal_panel_selected
    df = signal_panel_selected.copy()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], errors="coerce")
    df["_wbucket"] = df["entry_ts"].dt.floor(f"{int(window_min)}min")
    df = df.sort_values("entry_ts")
    df = df.drop_duplicates(subset=["trade_day", "_wbucket"], keep="first")
    df = df.drop(columns=["_wbucket"]).reset_index(drop=True)
    df["signal_row_id"] = np.arange(len(df))
    return df


def attach_regime_params(signal_panel_selected, feature_sheet_path=None):
    """
    Attach regime-classifier outputs as rc_* columns to every signal row.

    Sources for rv_slow / iv / move_to_open_1_std (tried in order):
      1. Already present in signal_panel_selected (carried from feature_filter_v1 merge)
      2. Loaded fresh from features_calculations sheet via load_feature_sheet()

    Fallback when values are missing: uses first entry of cfg config lists so
    the runner never drops a signal silently.
    """
    if not getattr(cfg, "REGIME_DRIVEN_ENTRY", True):
        return signal_panel_selected

    try:
        import regime_classifier as rc
    except ImportError:
        import warnings
        warnings.warn("regime_classifier not found — skipping regime attachment", stacklevel=2)
        return signal_panel_selected

    df = signal_panel_selected.copy()

    # ── Step 1: ensure rv_slow / iv / move_to_open_1_std are present ──────
    regime_input_cols = ["rv_slow", "iv", "move_to_open_1_std"]
    missing_inputs = [c for c in regime_input_cols if c not in df.columns]

    if missing_inputs and feature_sheet_path is not None:
        try:
            calc_feat = load_feature_sheet(feature_sheet_path)   # loads features_calculations
            needed = ["trade_day", "feature_time", "feature_ts"] + [
                c for c in regime_input_cols if c in calc_feat.columns
            ]
            calc_join = calc_feat[[c for c in needed if c in calc_feat.columns]].copy()
            df = df.merge(
                calc_join,
                left_on=["trade_day", "entry_time", "entry_ts"],
                right_on=["trade_day", "feature_time", "feature_ts"],
                how="left",
                suffixes=("", "_calc"),
            )
            # drop duplicate helper cols if any
            for _drop in ["feature_time_calc", "feature_ts_calc"]:
                if _drop in df.columns:
                    df = df.drop(columns=[_drop])
        except Exception as e:
            import warnings
            warnings.warn(f"attach_regime_params: could not load calc features — {e}", stacklevel=2)

    # If regime-driven entry is ON, regime inputs must be available.
    # Do not silently fall back to rc_regime=-1, because that makes the run look
    # like regime-driven while actually using fallback structure/multiplier.
    if getattr(cfg, "REGIME_DRIVEN_ENTRY", True):
        missing_regime_cols = [c for c in regime_input_cols if c not in df.columns]
        if missing_regime_cols:
            raise ValueError(
                "REGIME_DRIVEN_ENTRY=True requires regime input columns, but these are missing "
                f"after feature attachment: {missing_regime_cols}. "
                "Required columns are: rv_slow, iv, move_to_open_1_std. "
                "Check FEATURE_SHEET_PATH and feature sheet column names."
            )

        bad_counts = {
            c: int(pd.to_numeric(df[c], errors="coerce").isna().sum())
            for c in regime_input_cols
        }
        total_rows = len(df)
        if total_rows > 0 and any(v == total_rows for v in bad_counts.values()):
            raise ValueError(
                "REGIME_DRIVEN_ENTRY=True but one or more regime input columns are fully blank "
                f"after merge. Blank counts: {bad_counts}, total rows: {total_rows}. "
                "This usually means feature timestamps did not match signal timestamps."
            )

    # ── Step 2: call resolve_trade_params row-by-row ───────────────────────
    fallback_structure   = str(getattr(cfg, "STRATEGY_STRUCTURES", ["IRON_CONDOR"])[0]).upper()
    fallback_multiplier  = float(getattr(cfg, "SHORT_DTE_STD_MULTIPLIERS", [1.0])[0])

    rc_records = []
    for _, row in df.iterrows():
        rv   = safe_float(row.get("rv_slow"))
        iv_v = safe_float(row.get("iv"))
        mv   = safe_float(row.get("move_to_open_1_std"))

        if pd.isna(rv) or pd.isna(iv_v) or pd.isna(mv):
            rc_records.append({
                "rc_regime":      -1,
                "rc_rv_zone":     "Unknown",
                "rc_iv_zone":     "Unknown",
                "rc_move_zone":   "Unknown",
                "rc_structure":   fallback_structure,
                "rc_multiplier":  fallback_multiplier,
                "rc_sl_type":     "SD",
                "rc_sl_value":    np.nan,
                "rc_exit_side":   "one_side",
                "rc_combined_target_pct": np.nan,
            })
        else:
            params = rc.resolve_trade_params(rv, iv_v, mv)
            rc_records.append({
                "rc_regime":      params["regime"],
                "rc_rv_zone":     params["rv_zone"],
                "rc_iv_zone":     params["iv_zone"],
                "rc_move_zone":   params["move_zone"],
                "rc_structure":   str(params["structure"]).upper(),
                "rc_multiplier":  float(params["multiplier"]),
                "rc_sl_type":     params["sl_type"],
                "rc_sl_value":    float(params["sl_value"]),
                "rc_exit_side":   params["exit_side"],
                "rc_combined_target_pct": safe_float(
                    getattr(cfg, "REGIME_TO_COMBINED_TARGET", {})
                    .get(int(params["regime"]), {})
                    .get("target_pct")
                ),
            })

    rc_df = pd.DataFrame(rc_records, index=df.index)
    for col in rc_df.columns:
        df[col] = rc_df[col]

    return df


def sanitize_variant_tag(value):
    """Return a compact token safe for filenames and variant identifiers."""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    token = re.sub(r"_+", "_", token).strip("._-")
    return token or "variant"


def format_multiplier_tag(mult):
    """Return multiplier tags such as mult1.0 and mult0.8."""
    try:
        mult_text = str(float(mult))
    except Exception:
        mult_text = str(mult)
    return f"mult{sanitize_variant_tag(mult_text)}"


def build_variant_tag(structure, multiplier):
    """Return a stable variant tag such as IRON_CONDOR_mult1.0."""
    return f"{sanitize_variant_tag(structure)}_{format_multiplier_tag(multiplier)}"


def round_atm_boundary_from_spot(spot, optiontype):
    """
    Round the spot to the ATM stopping boundary using the same directional
    midpoint rules used for strike construction.
    """
    ot = normalize_option_type(optiontype)
    if ot == "CE":
        return round_to_step_directional(spot, cfg.STRIKE_STEP, "UP")
    if ot == "PE":
        return round_to_step_directional(spot, cfg.STRIKE_STEP, "DOWN")
    raise ValueError(f"Unsupported option type for ATM boundary: {optiontype}")


def generate_itm_candidate_strikes(short_strike, atm_boundary, optiontype, step):
    """
    Return strictly ITM Batman inner-long candidate strikes between short and ATM.

    Batman rules:
    - ATM is only a stopping boundary.
    - Never select ATM itself.
    - Never include the short strike itself in the candidate set.
    - This helper returns strike integers only; contract bars are resolved later
      on same-day data.
    """
    ot = normalize_option_type(optiontype)
    short = safe_int(short_strike)
    atm = safe_int(atm_boundary)
    step = safe_int(step)
    if pd.isna(short) or pd.isna(atm) or pd.isna(step) or step <= 0:
        return []

    if ot == "CE":
        candidates = list(range(int(atm) + int(step), int(short), int(step)))
        candidates = [s for s in candidates if int(atm) < s < int(short)]
        candidates.sort()
    elif ot == "PE":
        candidates = list(range(int(short) + int(step), int(atm), int(step)))
        candidates = [s for s in candidates if int(short) < s < int(atm)]
        candidates.sort(reverse=True)
    else:
        raise ValueError(f"Unsupported option type for Batman candidates: {optiontype}")

    if int(short) in candidates or int(atm) in candidates:
        raise AssertionError("Batman candidate set must exclude short strike and ATM boundary")
    return candidates


def _candidate_price_col(candidates_df):
    for col in ["candidate_price", "price", "resolved_price"]:
        if col in candidates_df.columns:
            return col
    raise ValueError("candidates_df must contain candidate_price, price, or resolved_price")


def choose_candidate_by_target_price(candidates_df, target_price, short_strike):
    """
    Choose the candidate closest to target price.

    The observed candidate min and max bound the search. If target_price is
    outside that range, return None. Ties prefer cheaper price, then strike
    closer to the short strike.
    """
    if candidates_df is None or candidates_df.empty or pd.isna(target_price):
        return None
    price_col = _candidate_price_col(candidates_df)
    df = candidates_df.copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    df = df.dropna(subset=[price_col, "strike"]).copy()
    if df.empty:
        return None
    min_price = df[price_col].min()
    max_price = df[price_col].max()
    if target_price < min_price or target_price > max_price:
        return None
    df["_abs_target_diff"] = (df[price_col] - target_price).abs()
    df["_short_distance"] = (df["strike"] - safe_float(short_strike)).abs()
    df = df.sort_values(["_abs_target_diff", price_col, "_short_distance", "strike"]).reset_index(drop=True)
    return df.iloc[0].drop(labels=["_abs_target_diff", "_short_distance"])


def resolve_price_same_day_exact_or_nearest(tradeday, expirydate, strike, optiontype, requestedts, optionsource, maxfallbackmin):
    """
    Resolve one same-day option price via exact bar, then nearest within fallback.

    The trade day passed in is always used as the contract trade day, so Batman
    inner-long selection does not drift into expiry-day or cross-day bars.
    """
    return resolve_option_bar(
        trade_day=tradeday,
        expiry_date=expirydate,
        strike=strike,
        option_type=optiontype,
        requested_ts=requestedts,
        option_source=optionsource,
        allow_nearest=True,
        max_fallback_min=maxfallbackmin,
    )


def _extract_resolved_price(resolved):
    return extract_bar_price(resolved, cfg.ENTRY_PRICE_FIELD)


def _first_itm_from_short(candidates, side):
    if not candidates:
        return np.nan
    if normalize_option_type(side) == "CE":
        return max(candidates)
    return min(candidates)


def resolve_batman_inner_long_for_side(intent_row, side, optionsource):
    """
    Select a Batman inner-long strike for CE or PE.

    Batman rules:
    - ATM is only a stopping boundary.
    - Never select ATM itself.
    - Never include the short strike itself in the candidate set.
    - Candidate contract bars are resolved on the same trade day only.
    - If target premium is outside the observed candidate range, fallback to
      the first ITM strike from the short strike, then price-resolve it again
      within cfg.BATMAN_NEAREST_PREMIUM_FALLBACK_MIN minutes.
    """
    side = normalize_option_type(side)
    if side not in {"CE", "PE"}:
        raise ValueError(f"Unsupported Batman side: {side}")

    short_col = "short_ce_strike" if side == "CE" else "short_pe_strike"
    outer_col = "outer_long_ce_strike" if side == "CE" else "outer_long_pe_strike"
    atm_col = "atm_ce_boundary" if side == "CE" else "atm_pe_boundary"
    short_strike = safe_int(intent_row[short_col])
    outer_strike = safe_int(intent_row[outer_col])
    atm_boundary = safe_int(intent_row[atm_col])
    candidates = generate_itm_candidate_strikes(short_strike, atm_boundary, side, cfg.STRIKE_STEP)

    result = {
        "selected_strike": np.nan,
        "selected_price": np.nan,
        "selection_mode": "NO_ITM_CANDIDATES",
        "target_price": np.nan,
        "min_candidate_price": np.nan,
        "max_candidate_price": np.nan,
        "atm_boundary": atm_boundary,
        "candidate_count": 0,
    }
    if not candidates:
        return result

    max_fb = getattr(cfg, "BATMAN_NEAREST_PREMIUM_FALLBACK_MIN", 5)
    trade_day = intent_row["trade_day"]
    expiry_date = intent_row["expiry_date"]
    entry_ts = intent_row["entry_ts"]

    short_res = resolve_price_same_day_exact_or_nearest(trade_day, expiry_date, short_strike, side, entry_ts, optionsource, max_fb)
    outer_res = resolve_price_same_day_exact_or_nearest(trade_day, expiry_date, outer_strike, side, entry_ts, optionsource, max_fb)
    short_price = _extract_resolved_price(short_res)
    outer_price = _extract_resolved_price(outer_res)
    if pd.isna(short_price) or pd.isna(outer_price):
        result["selection_mode"] = "MISSING_SHORT_OR_OUTER_PRICE"
        return result

    side_net_credit = (
        short_price * safe_float(getattr(cfg, "BATMAN_SHORT_QTY", 3))
        - outer_price * safe_float(getattr(cfg, "BATMAN_OUTER_LONG_QTY", 2))
    )
    target_price = safe_float(getattr(cfg, "BATMAN_TARGET_PREMIUM_FRACTION", 0.70)) * side_net_credit
    result["target_price"] = target_price

    candidate_rows = []
    for strike in candidates:
        resolved = resolve_price_same_day_exact_or_nearest(trade_day, expiry_date, strike, side, entry_ts, optionsource, max_fb)
        price = _extract_resolved_price(resolved)
        if pd.isna(price):
            continue
        candidate_rows.append({"strike": strike, "candidate_price": price, "fill_mode": resolved["fill_mode"]})

    candidates_df = pd.DataFrame(candidate_rows)
    if not candidates_df.empty:
        result["candidate_count"] = len(candidates_df)
        result["min_candidate_price"] = candidates_df["candidate_price"].min()
        result["max_candidate_price"] = candidates_df["candidate_price"].max()
        chosen = choose_candidate_by_target_price(candidates_df, target_price, short_strike)
        if chosen is not None:
            result.update({
                "selected_strike": safe_int(chosen["strike"]),
                "selected_price": safe_float(chosen["candidate_price"]),
                "selection_mode": "TARGET_PREMIUM",
            })
            return result

    fallback_strike = _first_itm_from_short(candidates, side)
    fallback_res = resolve_price_same_day_exact_or_nearest(trade_day, expiry_date, fallback_strike, side, entry_ts, optionsource, max_fb)
    fallback_price = _extract_resolved_price(fallback_res)
    result.update({
        "selected_strike": fallback_strike,
        "selected_price": fallback_price,
        "selection_mode": "FALLBACK_FIRST_ITM",
    })
    return result


def _validate_config_variants():
    structures = [str(x).strip().upper() for x in getattr(cfg, "STRATEGY_STRUCTURES", ["IRON_CONDOR"])]
    multipliers = list(getattr(cfg, "SHORT_DTE_STD_MULTIPLIERS", [1.0]))
    if not multipliers:
        raise ValueError("SHORT_DTE_STD_MULTIPLIERS must contain at least one value")
    allowed = {"IRON_CONDOR", "BATMAN"}
    unknown = [s for s in structures if s not in allowed]
    if unknown:
        raise ValueError(f"Unsupported strategy structure(s): {unknown}")
    return structures, multipliers


def build_trade_intents_base(signal_df):
    rows = []
    structures, multipliers = _validate_config_variants()
    signal_df = signal_df.copy().reset_index(drop=True)

    for _, row in signal_df.iterrows():
        spot = safe_float(row.get("spot_open"))
        dte_std = safe_float(row.get("dte_std"))
        one_std = safe_float(row.get("one_std"))
        valid = not pd.isna(spot) and not pd.isna(dte_std) and not pd.isna(one_std)

        # Patch 3: regime-driven single pair vs config grid
        _rc_structure  = row.get("rc_structure")
        _rc_multiplier = row.get("rc_multiplier")
        _regime_driven = (
            getattr(cfg, "REGIME_DRIVEN_ENTRY", True)
            and _rc_structure is not None
            and not (isinstance(_rc_structure, float) and pd.isna(_rc_structure))
            and _rc_multiplier is not None
            and not (isinstance(_rc_multiplier, float) and pd.isna(_rc_multiplier))
        )
        iter_pairs = (
            [(str(_rc_structure).upper(), safe_float(_rc_multiplier))]
            if _regime_driven
            else [(s, safe_float(m)) for s in structures for m in multipliers]
        )

        for structure, multiplier in iter_pairs:
            multiplier = safe_float(multiplier)
            variant_tag = build_variant_tag(structure, multiplier)
            effective_short_move = dte_std * multiplier if valid and not pd.isna(multiplier) else np.nan
            short_ce = round_to_step_directional(spot + effective_short_move, cfg.STRIKE_STEP, "UP") if valid else np.nan
            short_pe = round_to_step_directional(spot - effective_short_move, cfg.STRIKE_STEP, "DOWN") if valid else np.nan
            outer_ce = round_to_step_directional(short_ce + one_std, cfg.STRIKE_STEP, "UP") if valid else np.nan
            outer_pe = round_to_step_directional(short_pe - one_std, cfg.STRIKE_STEP, "DOWN") if valid else np.nan
            atm_ce = round_atm_boundary_from_spot(spot, "CE") if valid else np.nan
            atm_pe = round_atm_boundary_from_spot(spot, "PE") if valid else np.nan
            trade_id = (
                "T_"
                + pd.Timestamp(row["entry_ts"]).strftime("%Y%m%d_%H%M")
                + "_"
                + str(int(row["signal_row_id"])).zfill(7)
                + "_"
                + variant_tag
            )
            rows.append({
                    "trade_id": trade_id,
                    "strategy_name": cfg.STRATEGY_NAME,
                    "entry_mode": cfg.ENTRY_MODE,
                    "entry_family": structure,
                    "structure": structure,
                    "short_dte_std_multiplier": multiplier,
                    "variant_tag": variant_tag,
                    "signal_row_id": row["signal_row_id"],
                    "trade_day": row["trade_day"],
                    "entry_ts": row["entry_ts"],
                    "entry_time": row["entry_time"],
                    "expiry_date": row["expiry_date"],
                    "dte": row["dte"],
                    "day_name": row["Day"] if "Day" in row.index else None,
                    "spot_at_entry": row["spot_open"],
                    "dte_std": dte_std,
                    "one_std": one_std,
                    "roundvalue": dte_std,
                    "hedge_move": one_std,
                    "effective_short_move": effective_short_move,
                    "atm_ce_boundary": atm_ce,
                    "atm_pe_boundary": atm_pe,
                    "short_ce_strike": short_ce,
                    "short_pe_strike": short_pe,
                    "outer_long_ce_strike": outer_ce,
                    "outer_long_pe_strike": outer_pe,
                    "long_ce_strike": outer_ce,
                    "long_pe_strike": outer_pe,
                    "inner_long_ce_strike": np.nan,
                    "inner_long_pe_strike": np.nan,
                    "ce_selection_mode": "NA",
                    "pe_selection_mode": "NA",
                    "ce_target_price": np.nan,
                    "pe_target_price": np.nan,
                    "ce_candidate_count": 0,
                    "pe_candidate_count": 0,
                    "ce_min_candidate_price": np.nan,
                    "ce_max_candidate_price": np.nan,
                    "pe_min_candidate_price": np.nan,
                    "pe_max_candidate_price": np.nan,
                    "intent_status": "READY" if valid else "INVALID_FORMULA",
                    # Patch 3: carry regime metadata into intent row
                    "rc_regime":     row.get("rc_regime",    np.nan),
                    "rc_rv_zone":    row.get("rc_rv_zone",   None),
                    "rc_iv_zone":    row.get("rc_iv_zone",   None),
                    "rc_move_zone":  row.get("rc_move_zone", None),
                    "rc_sl_type":    row.get("rc_sl_type",   None),
                    "rc_sl_value":   row.get("rc_sl_value",  np.nan),
                    "rc_exit_side":  row.get("rc_exit_side", "one_side"),
                    "rc_combined_target_pct": safe_float(row.get("rc_combined_target_pct", np.nan)),
                })

    return pd.DataFrame(rows)


def finalize_trade_intents_with_batman_selection(trade_intents_df, option_source):
    out = trade_intents_df.copy().reset_index(drop=True)
    if out.empty or "structure" not in out.columns:
        return out
    for idx, row in out.iterrows():
        if str(row.get("structure", "")).upper() != "BATMAN" or row.get("intent_status") != "READY":
            continue
        for side, prefix in [("CE", "ce"), ("PE", "pe")]:
            selected = resolve_batman_inner_long_for_side(row, side, option_source)
            out.at[idx, f"inner_long_{prefix}_strike"] = selected["selected_strike"]
            out.at[idx, f"{prefix}_selection_mode"] = selected["selection_mode"]
            out.at[idx, f"{prefix}_target_price"] = selected["target_price"]
            out.at[idx, f"{prefix}_candidate_count"] = selected["candidate_count"]
            out.at[idx, f"{prefix}_min_candidate_price"] = selected["min_candidate_price"]
            out.at[idx, f"{prefix}_max_candidate_price"] = selected["max_candidate_price"]
        ce_bad = pd.isna(out.at[idx, "inner_long_ce_strike"])
        pe_bad = pd.isna(out.at[idx, "inner_long_pe_strike"])
        if ce_bad or pe_bad:
            out.at[idx, "intent_status"] = "INVALID_BATMAN_SELECTION"
    return out


def build_trade_intents(signal_df):
    return build_trade_intents_base(signal_df)


# ============================================================
# OPTION ACCESS MODES
# ============================================================
# ─────────────────────────────────────────────────────────
# OPTION PRICE FILL  (forward-fill / back-fill NaN bars)
# ─────────────────────────────────────────────────────────
def _apply_option_price_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaN values in OHLC columns within a single contract DataFrame.
    The DataFrame must already be sorted by timestamp and deduplicated.

    Controlled by two module-level constants:
        OPTION_PRICE_FILL_METHOD : "ffill" | "bfill" | "ffill+bfill" | None
        OPTION_PRICE_FILL_LIMIT  : int or None (max consecutive bars to fill)

    "ffill+bfill" applies forward-fill first, then back-fill on any remaining
    NaNs (useful for gaps at the very start of the session).
    Setting OPTION_PRICE_FILL_METHOD = None skips filling entirely.
    """
    if not OPTION_PRICE_FILL_METHOD or df.empty:
        return df

    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if not price_cols:
        return df

    out = df.copy()
    method = str(OPTION_PRICE_FILL_METHOD).strip().lower()
    limit = int(OPTION_PRICE_FILL_LIMIT) if OPTION_PRICE_FILL_LIMIT is not None else None

    if method == "ffill":
        out[price_cols] = out[price_cols].ffill(limit=limit)
    elif method == "bfill":
        out[price_cols] = out[price_cols].bfill(limit=limit)
    elif method in ("ffill+bfill", "ffill_bfill"):
        out[price_cols] = out[price_cols].ffill(limit=limit).bfill(limit=limit)
    else:
        raise ValueError(
            f"OPTION_PRICE_FILL_METHOD='{OPTION_PRICE_FILL_METHOD}' is not supported. "
            "Use 'ffill', 'bfill', 'ffill+bfill', or None."
        )

    return out


class ContractCache:
    def __init__(self, max_items=5000):
        self.max_items = max_items
        self.store = OrderedDict()

    def get(self, key):
        if key not in self.store:
            return None
        self.store.move_to_end(key)
        return self.store[key]

    def set(self, key, value):
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.max_items:
            self.store.popitem(last=False)

    def __len__(self):
        return len(self.store)


def _get_pkl_path(folder, trade_day):
    """
    Returns the .pkl file path for the YYYY-MM of trade_day.
    Pattern: <folder>/NIFTY_YYYYMM.pkl
    """
    d = pd.Timestamp(trade_day)
    fname = f"NIFTY_{d.year}{d.month:02d}.pkl"
    return os.path.join(folder, fname)


def _load_pkl_month(folder, trade_day):
    """
    Load a monthly PKL WITHOUT caching - load, use, discard.
    Avoids accumulating all months in RAM simultaneously.
    Uses pd.read_pickle which is more memory-efficient than pickle.load.
    Returns a pd.DataFrame or empty DataFrame if file not found.
    """
    path = _get_pkl_path(folder, trade_day)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_pickle(path)
    except Exception as e:
        print(f"  [PKL load error] {path}: {e}")
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df


def open_options_dataset(folder):
    """
    For PKL mode the 'dataset' object is just the folder path string.
    Kept with the same signature so the rest of the code is unchanged.
    """
    return folder


def load_single_contract_from_pkl(dataset, trade_day, expiry_date, strike, option_type):
    """
    `dataset` is now the PKL folder path (string).
    Loads the correct monthly PKL, filters to the requested contract,
    and returns a clean contract DataFrame.
    """
    trade_day = normalize_date_str(trade_day)
    expiry_date = normalize_date_str(expiry_date)
    option_type = normalize_option_type(option_type)
    strike = safe_int(strike)
    if trade_day is None or expiry_date is None or option_type is None or pd.isna(strike):
        return pd.DataFrame()

    folder = dataset
    raw = _load_pkl_month(folder, trade_day)
    if raw.empty:
        return pd.DataFrame()

    raw = normalize_column_names(raw)

    required_cols = ["Date", "Time", "ExpiryDate", "StrikePrice", "Type", "Open", "High", "Low", "Close"]
    missing_required = [c for c in required_cols if c not in raw.columns]
    if missing_required:
        raise ValueError(f"PKL file missing columns {missing_required}")

    df = raw[
        (raw["Date"].astype(str).str.strip().str[:10] == trade_day)
        & (raw["ExpiryDate"].astype(str).str.strip().str[:10] == expiry_date)
        & (pd.to_numeric(raw["StrikePrice"], errors="coerce").fillna(-1).astype(int) == int(strike))
        & (raw["Type"].map(normalize_option_type) == option_type)
    ].copy()

    if df.empty:
        return pd.DataFrame()
    df["Date"] = df["Date"].astype("string").str.strip().str.slice(0, 10)
    df["ExpiryDate"] = df["ExpiryDate"].astype("string").str.strip().str.slice(0, 10)
    df["Time"] = df["Time"].astype("string").str.strip().str.slice(0, 5)
    df["Type"] = df["Type"].map(normalize_option_type)
    df["StrikePrice"] = pd.to_numeric(df["StrikePrice"], errors="coerce")
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["Date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False) &
        df["ExpiryDate"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False) &
        df["Time"].str.match(r"^\d{2}:\d{2}$", na=False)
    ].copy()
    df = df.dropna(subset=["Date", "Time", "ExpiryDate", "StrikePrice", "Type", "Open", "High", "Low", "Close"]).copy()
    if df.empty:
        return df

    df["StrikePrice"] = df["StrikePrice"].astype(np.int32)
    df = df[(df["Time"] >= cfg.SESSION_START) & (df["Time"] <= cfg.SESSION_END)].copy()
    df["trade_day"] = df["Date"]
    df["expiry_date"] = df["ExpiryDate"]
    df["option_type"] = df["Type"]
    df["strike"] = df["StrikePrice"].astype(np.int32)
    df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df = df.sort_values(["timestamp"]).drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    # Fill missing OHLC values within this contract's day before timestamp lookups.
    df = _apply_option_price_fill(df)
    # ── set timestamp as index for O(log n) bar lookups
    df = df.set_index("timestamp", drop=False)
    df["contract_key"] = df.apply(
        lambda r: make_contract_key(r["trade_day"], r["expiry_date"], r["strike"], r["option_type"]),
        axis=1
    )
    return df


def get_contract_df_lazy(trade_day, expiry_date, strike, option_type, dataset, cache):
    key = make_contract_key(trade_day, expiry_date, strike, option_type)
    if key is None:
        return pd.DataFrame()
    cached = cache.get(key)
    if cached is not None:
        return cached
    df = load_single_contract_from_pkl(dataset, trade_day, expiry_date, strike, option_type)
    cache.set(key, df)
    return df


def _fallback_offsets():
    step = int(cfg.STRIKE_STEP)
    return [0] + [
        direction * i * step
        for i in range(1, MAX_STRIKE_FALLBACK_STEPS + 1)
        for direction in (1, -1)
    ]


def _base_days_for_preload(row):
    expiry_date = row["expiry_date"]
    base_days = {normalize_date_str(row["trade_day"]), normalize_date_str(expiry_date)}
    if getattr(cfg, "SAVE_INTRATRADE_TIMELINE", False):
        entry_ts = pd.Timestamp(row["entry_ts"])
        trade_end_ts = combine_date_time(expiry_date, cfg.SESSION_END)
        timeline_days = {
            pd.Timestamp(ts).strftime("%Y-%m-%d")
            for ts in build_intratrade_mark_grid(
                entry_ts=entry_ts,
                end_ts=trade_end_ts,
                timeframe=getattr(cfg, "INTRATRADE_MARK_TIMEFRAME", "15min"),
                include_final_exit_row=getattr(cfg, "INTRATRADE_INCLUDE_FINAL_EXIT_ROW", True),
            )
        }
        base_days |= timeline_days
    return {d for d in base_days if d is not None}


def _strike_pairs_for_intent(row):
    structure = str(row.get("structure", "IRON_CONDOR")).upper()
    if structure == "BATMAN":
        cols = [
            ("inner_long_ce_strike", "CE"),
            ("short_ce_strike", "CE"),
            ("outer_long_ce_strike", "CE"),
            ("inner_long_pe_strike", "PE"),
            ("short_pe_strike", "PE"),
            ("outer_long_pe_strike", "PE"),
        ]
    elif structure == "IRON_CONDOR":
        cols = [
            ("short_ce_strike", "CE"),
            ("short_pe_strike", "PE"),
            ("outer_long_ce_strike", "CE"),
            ("outer_long_pe_strike", "PE"),
        ]
    else:
        raise ValueError(f"Unsupported strategy structure for preload: {structure}")
    pairs = []
    for col, option_type in cols:
        if col in row.index:
            strike = safe_int(row.get(col))
            if not pd.isna(strike):
                pairs.append((strike, option_type))
    return pairs


def _add_contract_keys_for_pairs(keys, days, expiry_date, strike_pairs, fallback_offsets):
    for d in days:
        for strike, option_type in strike_pairs:
            base_strike = safe_int(strike)
            if pd.isna(base_strike):
                continue
            for offset in fallback_offsets:
                key = make_contract_key(d, expiry_date, base_strike + offset, option_type)
                if key is not None:
                    keys.add(key)


def build_batman_candidate_contract_keys(trade_intents_df):
    """
    Include same-day Batman candidate strikes so pre-finalization preload can
    resolve inner-long prices without needing a second preload.
    """
    keys = set()
    fallback_offsets = _fallback_offsets() if NEAREST_STRIKE_FALLBACK_ENABLED else [0]
    for _, row in trade_intents_df.iterrows():
        if str(row.get("structure", "")).upper() != "BATMAN":
            continue
        expiry_date = row["expiry_date"]
        trade_day = normalize_date_str(row["trade_day"])
        if trade_day is None:
            continue
        strike_pairs = []
        for side in ["CE", "PE"]:
            short_col = "short_ce_strike" if side == "CE" else "short_pe_strike"
            atm_col = "atm_ce_boundary" if side == "CE" else "atm_pe_boundary"
            candidates = generate_itm_candidate_strikes(row.get(short_col), row.get(atm_col), side, cfg.STRIKE_STEP)
            strike_pairs.extend((strike, side) for strike in candidates)
        _add_contract_keys_for_pairs(keys, {trade_day}, expiry_date, strike_pairs, fallback_offsets)
    return keys


def build_required_contract_keys(trade_intents_df):
    keys = set()
    # Include exact strike +/- fallback steps so preload covers contracts the
    # nearest-strike bar search may need at runtime.
    fallback_offsets = _fallback_offsets()
    for _, row in trade_intents_df.iterrows():
        expiry_date = row["expiry_date"]
        base_days = _base_days_for_preload(row)
        _add_contract_keys_for_pairs(keys, base_days, expiry_date, _strike_pairs_for_intent(row), fallback_offsets)
    keys |= build_batman_candidate_contract_keys(trade_intents_df)
    return keys


def preload_required_contracts(dataset, trade_intents_df):
    """
    PKL-aware preload.
    `dataset` is the folder path string (set by open_options_dataset).
    Loads each required monthly PKL once, filters to needed contracts,
    and builds the same lookup dict as the old parquet path.
    """
    needed_keys = build_required_contract_keys(trade_intents_df)
    if not needed_keys:
        return {}

    needed_trade_days = set()
    needed_expiries = set()
    needed_types = set()
    needed_strikes = set()
    for key in needed_keys:
        d, e, ot, s = key.split("|")
        needed_trade_days.add(d)
        needed_expiries.add(e)
        needed_types.add(ot)
        needed_strikes.add(str(int(float(s))))

    folder = dataset
    lookup_parts = {}
    t0 = perf_counter()
    scanned_rows = 0
    kept_rows = 0

    days_by_month = {}
    for d in needed_trade_days:
        ym = d[:7]
        days_by_month.setdefault(ym, set()).add(d)

    required_cols = ["Date", "Time", "ExpiryDate", "StrikePrice", "Type", "Open", "High", "Low", "Close"]

    for ym, days_in_month in sorted(days_by_month.items()):
        year, month = ym.split("-")
        fname = f"NIFTY_{year}{month}.pkl"
        fpath = os.path.join(folder, fname)
        if not os.path.exists(fpath):
            print(f"  [preload] PKL not found, skipping: {fpath}")
            continue

        raw = _load_pkl_month(folder, list(days_in_month)[0])   # load fresh, no cache
        if raw.empty:
            continue
        raw = normalize_column_names(raw)

        missing = [c for c in required_cols if c not in raw.columns]
        if missing:
            raise ValueError(f"PKL {fname} missing columns {missing}")

        df = raw.copy()
        scanned_rows += len(df)

        df["Date"] = df["Date"].astype(str).str.strip().str[:10]
        df["ExpiryDate"] = df["ExpiryDate"].astype(str).str.strip().str[:10]
        df["Time"] = df["Time"].astype(str).str.strip().str[:5]
        df["Type"] = df["Type"].astype(str).str.strip().str.upper()
        df["Type"] = df["Type"].replace({"CALL": "CE", "C": "CE", "PUT": "PE", "P": "PE"})
        df["StrikePrice"] = pd.to_numeric(df["StrikePrice"], errors="coerce")
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=required_cols)
        if df.empty:
            continue

        df["StrikePrice"] = df["StrikePrice"].astype(np.int32)

        df = df[
            df["Date"].isin(days_in_month)
            & df["ExpiryDate"].isin(needed_expiries)
            & df["Type"].isin(needed_types)
            & df["StrikePrice"].astype(str).isin(needed_strikes)
        ]
        if df.empty:
            continue
        df = df[df["Time"].between(cfg.SESSION_START, cfg.SESSION_END)]
        if df.empty:
            continue

        df["contract_key"] = (
            df["Date"].astype(str) + "|" +
            df["ExpiryDate"].astype(str) + "|" +
            df["Type"].astype(str) + "|" +
            df["StrikePrice"].astype(str)
        )
        df = df[df["contract_key"].isin(needed_keys)]
        if df.empty:
            continue

        df["trade_day"] = df["Date"]
        df["expiry_date"] = df["ExpiryDate"]
        df["option_type"] = df["Type"]
        df["strike"] = df["StrikePrice"].astype(np.int32)
        df["timestamp"] = pd.to_datetime(df["Date"] + " " + df["Time"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            continue

        kept_rows += len(df)
        for key, g in df.groupby("contract_key", sort=False):
            cols = [
                c for c in
                ["Date", "Time", "ExpiryDate", "StrikePrice", "Type", "Open", "High", "Low", "Close",
                 "Ticker", "trade_day", "expiry_date", "option_type", "strike", "timestamp", "contract_key"]
                if c in g.columns
            ]
            if key in lookup_parts:
                lookup_parts[key].append(g[cols].copy())
            else:
                lookup_parts[key] = [g[cols].copy()]

        print(
            f"  [preload] {fname}: days={len(days_in_month)}, "
            f"keys_so_far={len(lookup_parts)}, kept={kept_rows}, "
            f"elapsed={perf_counter()-t0:.1f}s"
        )
        del df

    lookup = {}
    for key, parts in lookup_parts.items():
        x = pd.concat(parts, ignore_index=True)
        x = (
            x.sort_values("timestamp")
            .drop_duplicates(subset="timestamp", keep="first")
            .reset_index(drop=True)
        )
        # Always normalize to canonical column names regardless of source path.
        if "Date" in x.columns:
            x["trade_day"] = x["Date"].astype(str).str.strip().str.slice(0, 10)
        if "ExpiryDate" in x.columns:
            x["expiry_date"] = x["ExpiryDate"].astype(str).str.strip().str.slice(0, 10)
        if "Type" in x.columns:
            x["option_type"] = x["Type"].map(normalize_option_type)
        if "StrikePrice" in x.columns:
            x["strike"] = pd.to_numeric(x["StrikePrice"], errors="coerce").astype("Int32")
        # Guarantee canonical columns are present; raise early if still missing.
        for _req_col in ("trade_day", "expiry_date", "option_type", "strike", "timestamp"):
            if _req_col not in x.columns:
                raise ValueError(
                    f"preload_required_contracts: contract key '{key}' is missing "
                    f"required column '{_req_col}' after normalization. "
                    f"Available columns: {list(x.columns)}"
                )
        x = _apply_option_price_fill(x)
        x = x.set_index("timestamp", drop=False)
        lookup[key] = x

    print(
        f"preload done scanned={scanned_rows}, lookup_keys={len(lookup)}, "
        f"kept={kept_rows}, elapsed={perf_counter()-t0:.1f}s"
    )
    return lookup

def get_contract_df_preloaded(trade_day, expiry_date, strike, option_type, lookup):
    key = make_contract_key(trade_day, expiry_date, strike, option_type)
    if key is None:
        return pd.DataFrame()
    return lookup.get(key, pd.DataFrame())


# ── bar-lookup helpers: exploit timestamp index when present ─────────────────
def _fetch_contract(trade_day, expiry_date, strike, option_type, option_source):
    """Fetch contract df regardless of access mode."""
    if option_source["mode"] == "lazy":
        return get_contract_df_lazy(
            trade_day, expiry_date, strike, option_type,
            option_source["dataset"], option_source["cache"]
        )
    return get_contract_df_preloaded(
        trade_day, expiry_date, strike, option_type,
        option_source["lookup"]
    )


def _get_nearest_bar_in_contract(contract_df, requested_ts, max_offset_min):
    """
    Within a contract, search +/- max_offset_min candle by candle
    in alternating +/- order. Returns the first bar found, or None.
    """
    if contract_df is None or contract_df.empty or pd.isna(requested_ts):
        return None
    requested_ts = pd.Timestamp(requested_ts)
    for offset in range(0, int(max_offset_min) + 1):
        for sign in ([0] if offset == 0 else [1, -1]):
            ts = requested_ts + pd.Timedelta(minutes=offset * sign)
            bar = get_option_bar_exact(contract_df, ts)
            if bar is not None:
                return bar
    return None


def _resolve_nearest_strike(trade_day, expiry_date, strike, option_type,
                            option_source, max_steps,
                            requested_ts=None, max_bar_offset_min=None):
    """
    Step 1 - try exact strike, scan bars +/-1 min, +/-2 min up to max_bar_offset_min.
    Step 2 - try +STRIKE_STEP, scan its bars the same way.
    Step 3 - try -STRIKE_STEP, then +/- 2*STRIKE_STEP, and so on.

    If requested_ts is None, falls back to contract-level existence only.
    Returns (contract_df, bar_or_None, actual_strike_used).
    """
    base = safe_int(strike)
    if pd.isna(base):
        return pd.DataFrame(), None, base

    step = int(cfg.STRIKE_STEP)
    bar_scan = max_bar_offset_min is not None and requested_ts is not None

    strike_offsets = [0] + [
        direction * i * step
        for i in range(1, max_steps + 1)
        for direction in (1, -1)
    ]

    for offset in strike_offsets:
        candidate = base + offset
        df = _fetch_contract(trade_day, expiry_date, candidate, option_type, option_source)
        if df is None or df.empty:
            continue

        if not bar_scan:
            return df, None, candidate

        bar = _get_nearest_bar_in_contract(df, requested_ts, max_bar_offset_min)
        if bar is not None:
            return df, bar, candidate

    return pd.DataFrame(), None, base


def get_option_bar_exact(contract_df, requested_ts):
    if contract_df is None or contract_df.empty or pd.isna(requested_ts):
        return None
    requested_ts = pd.Timestamp(requested_ts)
    if contract_df.index.name == "timestamp":
        if requested_ts in contract_df.index:
            return contract_df.loc[requested_ts]
        return None
    match = contract_df[contract_df["timestamp"] == requested_ts]
    return None if match.empty else match.iloc[0]


def get_option_bar_nearest_within(contract_df, requested_ts, max_offset_min):
    """
    Find the closest bar within +/- max_offset_min minutes.
    Tie-break rule:
    1) smaller absolute offset
    2) earlier timestamp if exactly tied
    """
    if contract_df is None or contract_df.empty or pd.isna(requested_ts):
        return None

    requested_ts = pd.Timestamp(requested_ts)
    lower_ts = requested_ts - pd.Timedelta(minutes=max_offset_min)
    upper_ts = requested_ts + pd.Timedelta(minutes=max_offset_min)

    if contract_df.index.name == "timestamp":
        x = contract_df.loc[lower_ts:upper_ts].copy()
        if x.empty:
            return None
        x = x.reset_index(drop=True)
    else:
        x = contract_df[
            (contract_df["timestamp"] >= lower_ts) &
            (contract_df["timestamp"] <= upper_ts)
        ].copy()
        if x.empty:
            return None

    x["abs_offset_min"] = (
        (pd.to_datetime(x["timestamp"]) - requested_ts).abs().dt.total_seconds() / 60.0
    )
    x = x.sort_values(["abs_offset_min", "timestamp"]).reset_index(drop=True)
    return x.iloc[0]


def resolve_option_bar(
    trade_day,
    expiry_date,
    strike,
    option_type,
    requested_ts,
    option_source,
    allow_nearest,
    max_fallback_min
):
    # ── primary contract fetch ───────────────────────────────────────────────
    strike_used = safe_int(strike)
    contract_df = _fetch_contract(trade_day, expiry_date, strike, option_type, option_source)

    # ── nearest-strike fallback if contract is completely absent ─────────────
    exact_bar = get_option_bar_exact(contract_df, requested_ts)
    if exact_bar is not None:
        fill_mode = "EXACT" if strike_used == safe_int(strike) else f"NEAREST_STRIKE:{strike_used}"
        return {
            "status": "FOUND",
            "requested_ts": pd.Timestamp(requested_ts),
            "actual_ts": pd.Timestamp(exact_bar["timestamp"]),
            "fill_mode": fill_mode,
            "fill_offset_min": 0.0,
            "bar": exact_bar,
            "contract_rows": len(contract_df),
            "strike_used": strike_used,
        }

    if allow_nearest:
        max_steps = MAX_STRIKE_FALLBACK_STEPS if NEAREST_STRIKE_FALLBACK_ENABLED else 0
        resolved_df, nearest_bar, resolved_strike = _resolve_nearest_strike(
            trade_day, expiry_date, strike, option_type,
            option_source,
            max_steps=max_steps,
            requested_ts=requested_ts,
            max_bar_offset_min=max_fallback_min,
        )
        if nearest_bar is not None:
            contract_df = resolved_df
            strike_used = resolved_strike
            actual_ts = pd.Timestamp(nearest_bar["timestamp"])
            fill_mode = "NEAREST_PM" if strike_used == safe_int(strike) else f"NEAREST_STRIKE:{strike_used}:NEAREST_PM"
            return {
                "status": "FOUND",
                "requested_ts": pd.Timestamp(requested_ts),
                "actual_ts": actual_ts,
                "fill_mode": fill_mode,
                "fill_offset_min": minutes_diff(requested_ts, actual_ts),
                "bar": nearest_bar,
                "contract_rows": len(contract_df),
                "strike_used": strike_used,
            }

    if (contract_df is None or contract_df.empty) and NEAREST_STRIKE_FALLBACK_ENABLED:
        contract_df, _, strike_used = _resolve_nearest_strike(
            trade_day, expiry_date, strike, option_type,
            option_source,
            max_steps=MAX_STRIKE_FALLBACK_STEPS,
        )

    if contract_df is None or contract_df.empty:
        return {
            "status": "MISSING_CONTRACT",
            "requested_ts": pd.Timestamp(requested_ts),
            "actual_ts": pd.NaT,
            "fill_mode": "MISSING",
            "fill_offset_min": np.nan,
            "bar": None,
            "contract_rows": 0,
            "strike_used": safe_int(strike),
        }

    return {
        "status": "MISSING_BAR",
        "requested_ts": pd.Timestamp(requested_ts),
        "actual_ts": pd.NaT,
        "fill_mode": "MISSING",
        "fill_offset_min": np.nan,
        "bar": None,
        "contract_rows": len(contract_df),
        "strike_used": strike_used,
    }


def extract_bar_price(resolved_bar_dict, price_field):
    if resolved_bar_dict is None:
        return np.nan
    bar = resolved_bar_dict.get("bar", None)
    if bar is None or price_field not in bar.index:
        return np.nan
    return safe_float(bar[price_field])


# ============================================================
# SIMULATOR CORE
# ============================================================
def build_legs_from_trade_intent(trade_row):
    common = {
        "trade_id": trade_row["trade_id"],
        "trade_day": trade_row["trade_day"],
        "entry_ts": trade_row["entry_ts"],
        "expiry_date": trade_row["expiry_date"],
        "dte": trade_row["dte"],
        "spot_at_entry": trade_row["spot_at_entry"],
        "roundvalue": trade_row["roundvalue"],
        "lot_size": cfg.LOT_SIZE,
        "intent_status": trade_row["intent_status"],
        "structure": trade_row.get("structure", "IRON_CONDOR"),
        "short_dte_std_multiplier": trade_row.get("short_dte_std_multiplier", 1.0),
        "variant_tag": trade_row.get("variant_tag", build_variant_tag("IRON_CONDOR", 1.0)),
        # Bug 1 fix: SD SL fields were absent; _sl_check_*_side needs all four.
        "dte_std": safe_float(trade_row.get("dte_std", trade_row.get("roundvalue"))),
        "short_ce_strike": safe_int(trade_row.get("short_ce_strike")),
        "short_pe_strike": safe_int(trade_row.get("short_pe_strike")),
        "rc_sl_type": trade_row.get("rc_sl_type"),
        "rc_sl_value": safe_float(trade_row.get("rc_sl_value")),
        "rc_exit_side": trade_row.get("rc_exit_side", "one_side"),
        "rc_combined_target_pct": safe_float(trade_row.get("rc_combined_target_pct", np.nan)),
    }
    structure = str(common["structure"]).upper()
    if structure == "IRON_CONDOR":
        qty = getattr(cfg, "IRON_CONDOR_LEG_QTY", cfg.QTY_PER_LEG)
        return [
            {**common, "leg_name": "SHORTCE", "option_type": "CE", "side": "SELL", "strike": safe_int(trade_row["short_ce_strike"]), "qty": qty},
            {**common, "leg_name": "SHORTPE", "option_type": "PE", "side": "SELL", "strike": safe_int(trade_row["short_pe_strike"]), "qty": qty},
            {**common, "leg_name": "LONGCE", "option_type": "CE", "side": "BUY", "strike": safe_int(trade_row["outer_long_ce_strike"]), "qty": qty},
            {**common, "leg_name": "LONGPE", "option_type": "PE", "side": "BUY", "strike": safe_int(trade_row["outer_long_pe_strike"]), "qty": qty},
        ]
    if structure == "BATMAN":
        return [
            {**common, "leg_name": "INNERLONGCE", "option_type": "CE", "side": "BUY", "strike": safe_int(trade_row["inner_long_ce_strike"]), "qty": getattr(cfg, "BATMAN_INNER_LONG_QTY", 1)},
            {**common, "leg_name": "SHORTCE", "option_type": "CE", "side": "SELL", "strike": safe_int(trade_row["short_ce_strike"]), "qty": getattr(cfg, "BATMAN_SHORT_QTY", 3)},
            {**common, "leg_name": "OUTERLONGCE", "option_type": "CE", "side": "BUY", "strike": safe_int(trade_row["outer_long_ce_strike"]), "qty": getattr(cfg, "BATMAN_OUTER_LONG_QTY", 2)},
            {**common, "leg_name": "INNERLONGPE", "option_type": "PE", "side": "BUY", "strike": safe_int(trade_row["inner_long_pe_strike"]), "qty": getattr(cfg, "BATMAN_INNER_LONG_QTY", 1)},
            {**common, "leg_name": "SHORTPE", "option_type": "PE", "side": "SELL", "strike": safe_int(trade_row["short_pe_strike"]), "qty": getattr(cfg, "BATMAN_SHORT_QTY", 3)},
            {**common, "leg_name": "OUTERLONGPE", "option_type": "PE", "side": "BUY", "strike": safe_int(trade_row["outer_long_pe_strike"]), "qty": getattr(cfg, "BATMAN_OUTER_LONG_QTY", 2)},
        ]
    raise ValueError(f"Unsupported strategy structure for legs: {structure}")


def resolve_entry_for_leg(leg_row, option_source):
    out = dict(leg_row)
    out.update({
        "entry_requested_ts": pd.Timestamp(leg_row["entry_ts"]),
        "entry_exec_ts": pd.NaT,
        "entry_price": np.nan,
        "entry_fill_mode": "MISSING",
        "entry_fill_offset_min": np.nan,
        "entry_status": "PENDING",
        "entry_skip_reason": None,
    })

    if leg_row.get("intent_status") != "READY":
        out["entry_status"] = "SKIPPED"
        out["entry_skip_reason"] = f"INTENT_{leg_row.get('intent_status')}"
        return out

    res = resolve_option_bar(
        trade_day=leg_row["trade_day"],
        expiry_date=leg_row["expiry_date"],
        strike=leg_row["strike"],
        option_type=leg_row["option_type"],
        requested_ts=leg_row["entry_ts"],
        option_source=option_source,
        allow_nearest=cfg.ALLOW_NEAREST_FILL,
        max_fallback_min=cfg.MAX_ENTRY_FALLBACK_MIN,
    )
    if res["status"] != "FOUND":
        forced_price = get_forced_option_price(leg_row["entry_ts"], leg_row["expiry_date"])
        if not pd.isna(forced_price):
            out["entry_exec_ts"] = pd.Timestamp(leg_row["entry_ts"])
            out["entry_price"] = forced_price
            out["entry_fill_mode"] = f"FORCED_DEFAULT:{res['status']}"
            out["entry_fill_offset_min"] = np.nan
            out["entry_status"] = "FILLED"
            out["entry_skip_reason"] = None
            return out
        out["entry_status"] = "FAILED"
        out["entry_fill_mode"] = res["fill_mode"]
        out["entry_fill_offset_min"] = res["fill_offset_min"]
        out["entry_skip_reason"] = res["status"]
        return out

    entry_price = extract_bar_price(res, cfg.ENTRY_PRICE_FIELD)
    if pd.isna(entry_price):
        forced_price = get_forced_option_price(res["actual_ts"], leg_row["expiry_date"])
        if not pd.isna(forced_price):
            out["entry_exec_ts"] = pd.Timestamp(res["actual_ts"])
            out["entry_price"] = forced_price
            out["entry_fill_mode"] = f"{res['fill_mode']}:FORCED_DEFAULT"
            out["entry_fill_offset_min"] = res["fill_offset_min"]
            out["entry_status"] = "FILLED"
            out["entry_skip_reason"] = None
            return out
        out["entry_status"] = "FAILED"
        out["entry_fill_mode"] = res["fill_mode"]
        out["entry_fill_offset_min"] = res["fill_offset_min"]
        out["entry_skip_reason"] = f"MISSING_{cfg.ENTRY_PRICE_FIELD.upper()}"
        return out

    out["entry_exec_ts"] = pd.Timestamp(res["actual_ts"])
    out["entry_price"] = entry_price
    out["entry_fill_mode"] = res["fill_mode"]
    out["entry_fill_offset_min"] = res["fill_offset_min"]
    out["entry_status"] = "FILLED"
    # Printed strike stays as the originally requested leg strike.
    # The actual strike used for price fetch is carried in fill_mode only.
    return out


def resolve_expiry_exit_for_leg(leg_row, option_source):
    out = dict(leg_row)
    req_ts = combine_date_time(leg_row["expiry_date"], cfg.EXPIRY_EXIT_TIME)
    expiry_after_only = getattr(cfg, "EXPIRY_AFTER_MINUTES_ONLY", False)
    out.update({
        "exit_requested_ts": req_ts,
        "exit_exec_ts": pd.NaT,
        "exit_price": np.nan,
        "exit_fill_mode": "MISSING",
        "exit_fill_offset_min": np.nan,
        "exit_status": "PENDING",
        "exit_skip_reason": None,
        "exit_reason": cfg.EXIT_REASON_DEFAULT,
    })

    if out.get("entry_status") != "FILLED":
        out["exit_status"] = "SKIPPED"
        out["exit_skip_reason"] = "ENTRY_NOT_FILLED"
        return out

    res = resolve_option_bar(
        trade_day=leg_row["expiry_date"],
        expiry_date=leg_row["expiry_date"],
        strike=leg_row["strike"],
        option_type=leg_row["option_type"],
        requested_ts=req_ts,
        option_source=option_source,
        allow_nearest=True,
        max_fallback_min=cfg.MAX_EXIT_FALLBACK_MIN,
    )
    if (
        expiry_after_only
        and res["status"] == "FOUND"
        and not pd.isna(res.get("actual_ts"))
        and pd.Timestamp(res["actual_ts"]) < pd.Timestamp(req_ts)
    ):
        res = {
            "status": "MISSING_BAR",
            "requested_ts": pd.Timestamp(req_ts),
            "actual_ts": pd.NaT,
            "fill_mode": "REJECTED_BEFORE_EXIT_TIME",
            "fill_offset_min": np.nan,
            "bar": None,
            "contract_rows": res.get("contract_rows", 0),
            "strike_used": res.get("strike_used", safe_int(leg_row["strike"])),
        }
    if res["status"] != "FOUND":
        forced_price = get_forced_option_price(req_ts, leg_row["expiry_date"])
        if not pd.isna(forced_price):
            out["exit_exec_ts"] = pd.Timestamp(req_ts)
            out["exit_price"] = forced_price
            out["exit_fill_mode"] = f"FORCED_DEFAULT:{res['status']}"
            out["exit_fill_offset_min"] = np.nan
            out["exit_status"] = "FILLED"
            out["exit_skip_reason"] = None
            return out
        out["exit_status"] = "FAILED"
        out["exit_fill_mode"] = res["fill_mode"]
        out["exit_fill_offset_min"] = res["fill_offset_min"]
        out["exit_skip_reason"] = res["status"]
        return out

    exit_price = extract_bar_price(res, cfg.EXIT_PRICE_FIELD)
    if pd.isna(exit_price):
        forced_price = get_forced_option_price(res["actual_ts"], leg_row["expiry_date"])
        if not pd.isna(forced_price):
            out["exit_exec_ts"] = pd.Timestamp(res["actual_ts"])
            out["exit_price"] = forced_price
            out["exit_fill_mode"] = f"{res['fill_mode']}:FORCED_DEFAULT"
            out["exit_fill_offset_min"] = res["fill_offset_min"]
            out["exit_status"] = "FILLED"
            out["exit_skip_reason"] = None
            return out
        out["exit_status"] = "FAILED"
        out["exit_fill_mode"] = res["fill_mode"]
        out["exit_fill_offset_min"] = res["fill_offset_min"]
        out["exit_skip_reason"] = f"MISSING_{cfg.EXIT_PRICE_FIELD.upper()}"
        return out

    out["exit_exec_ts"] = pd.Timestamp(res["actual_ts"])
    out["exit_price"] = exit_price
    out["exit_fill_mode"] = res["fill_mode"]
    out["exit_fill_offset_min"] = res["fill_offset_min"]
    out["exit_status"] = "FILLED"
    # Printed strike stays as the originally requested leg strike.
    # The actual strike used for price fetch is carried in fill_mode only.
    return out


def compute_leg_points_pnl(side, entry_price, exit_price):
    entry_price = safe_float(entry_price)
    exit_price = safe_float(exit_price)
    if pd.isna(entry_price) or pd.isna(exit_price):
        return np.nan
    return entry_price - exit_price if side == "SELL" else exit_price - entry_price


def compute_leg_total_pnl(points_pnl, qty, lot_size):
    points_pnl = safe_float(points_pnl)
    qty = safe_float(qty)
    lot_size = safe_float(lot_size)
    if pd.isna(points_pnl) or pd.isna(qty) or pd.isna(lot_size):
        return np.nan
    return points_pnl * qty * lot_size


def compute_leg_cashflow(side, price, qty, lot_size):
    price = safe_float(price)
    qty = safe_float(qty)
    lot_size = safe_float(lot_size)
    if pd.isna(price) or pd.isna(qty) or pd.isna(lot_size):
        return np.nan
    gross = price * qty * lot_size
    return gross if side == "SELL" else -gross


def compute_running_leg_pnl(side, entry_price, mark_price, qty, lot_size):
    entry_price = safe_float(entry_price)
    mark_price = safe_float(mark_price)
    qty = safe_float(qty)
    lot_size = safe_float(lot_size)
    if pd.isna(entry_price) or pd.isna(mark_price) or pd.isna(qty) or pd.isna(lot_size):
        return np.nan
    points = (entry_price - mark_price) if side == "SELL" else (mark_price - entry_price)
    return points * qty * lot_size


def _finalize_leg_rows(entry_legs, exit_legs):
    final_legs = []
    for leg in exit_legs:
        x = dict(leg)
        x["points_pnl"] = compute_leg_points_pnl(x["side"], x.get("entry_price"), x.get("exit_price"))
        x["gross_pnl_total"] = compute_leg_total_pnl(x["points_pnl"], x.get("qty"), x.get("lot_size"))
        final_legs.append(x)
    return final_legs


def _row_get(row, key, default=np.nan):
    if isinstance(key, (list, tuple)):
        for k in key:
            val = _row_get(row, k, np.nan)
            if _is_missing_value(val):
                continue
            if isinstance(val, str) and val.strip() == "":
                continue
            return val
        return default
    if isinstance(row, pd.Series):
        return row.get(key, default)
    if isinstance(row, dict):
        return row.get(key, default)
    return default


def _is_missing_value(value):
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _first_present(row, keys, default=np.nan):
    for key in keys:
        val = _row_get(row, key, np.nan)
        if _is_missing_value(val):
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        return val
    return default


def _infer_structure_name(trade_row):
    raw = _first_present(
        trade_row,
        ["structure", "Structure", "variant_tag", "VariantTag", "strategy_name", "Strategy_Name"],
        default="",
    )
    s = str(raw).strip().upper()
    if "BATMAN" in s:
        return "BATMAN"
    if ("IRON" in s) and ("CONDOR" in s):
        return "IRON_CONDOR"
    return str(getattr(cfg, "STRUCTURE_NAME", "IRON_CONDOR")).strip().upper()


def _canonical_trade_output_fields(trade_row):
    structure = _first_present(trade_row, ["structure", "Structure"], default=np.nan)
    if _is_missing_value(structure) or str(structure).strip() == "":
        structure = _infer_structure_name(trade_row)
    structure = str(structure).strip().upper()

    variant_tag = _first_present(
        trade_row,
        ["variant_tag", "VariantTag"],
        default=getattr(cfg, "VARIANT_TAG", structure),
    )
    if _is_missing_value(variant_tag) or str(variant_tag).strip() == "":
        variant_tag = structure

    ordered_leg_names = _first_present(
        trade_row,
        ["ordered_leg_names", "Ordered_Leg_Names"],
        default="SHORTCE|SHORTPE|LONGCE|LONGPE",
    )

    return {
        "Structure": structure,
        "VariantTag": str(variant_tag).strip(),
        "Ordered_Leg_Names": str(ordered_leg_names).strip(),
        "InnerLongCE_Strike": _first_present(
            trade_row,
            ["inner_long_ce_strike", "InnerLongCE_Strike", "InnerLongCEStrike"],
            default=np.nan,
        ),
        "InnerLongPE_Strike": _first_present(
            trade_row,
            ["inner_long_pe_strike", "InnerLongPE_Strike", "InnerLongPEStrike"],
            default=np.nan,
        ),
        "OuterLongCE_Strike": _first_present(
            trade_row,
            ["outer_long_ce_strike", "OuterLongCE_Strike", "OuterLongCEStrike"],
            default=np.nan,
        ),
        "OuterLongPE_Strike": _first_present(
            trade_row,
            ["outer_long_pe_strike", "OuterLongPE_Strike", "OuterLongPEStrike"],
            default=np.nan,
        ),
    }


def signed_sum_or_nan(values):
    vals = [safe_float(v) for v in values]
    if all(pd.isna(v) for v in vals):
        return np.nan
    return float(np.nansum(vals))


def _leg_output_prefix(leg_name):
    return str(leg_name).replace("_", "")


def _leg_order_for_structure(structure, leg_rows=None):
    structure = str(structure or "IRON_CONDOR").upper()
    if structure == "BATMAN":
        return ["INNERLONGCE", "SHORTCE", "OUTERLONGCE", "INNERLONGPE", "SHORTPE", "OUTERLONGPE"]
    if structure == "IRON_CONDOR":
        return ["SHORTCE", "SHORTPE", "LONGCE", "LONGPE"]
    if leg_rows is not None:
        return [x["leg_name"] for x in leg_rows]
    raise ValueError(f"Unsupported strategy structure: {structure}")


def _exit_reason_label(exit_reason):
    if exit_reason is None:
        return ""
    try:
        if pd.isna(exit_reason):
            return ""
    except Exception:
        pass
    r = str(exit_reason).upper()
    # COMBINED_TP and COMBINED_TP_AT_EXPIRY must be checked before "EXPIRY"
    # because "EXPIRY" is a substring of "COMBINED_TP_AT_EXPIRY".
    if "COMBINED_TP" in r:
        return str(exit_reason)
    if "SL" in r:
        return "SL"
    if "TGT" in r:
        return "TGT"
    if "EXPIRY" in r:
        return "EXPIRY"
    return str(exit_reason)


def _format_sl_params(trade_row):
    sl_type  = trade_row.get("rc_sl_type",  None)
    sl_value = safe_float(trade_row.get("rc_sl_value", np.nan))
    if sl_type is None or (isinstance(sl_type, float) and pd.isna(sl_type)):
        return np.nan
    if pd.isna(sl_value):
        return str(sl_type)
    return f"{sl_type}:{sl_value}"


def build_trade_rows(trade_row, leg_rows, trade_status, skip_reason):
    canonical_meta = _canonical_trade_output_fields(trade_row)
    structure = canonical_meta["Structure"]
    is_batman = structure == "BATMAN"
    leg_map = {x["leg_name"]: x for x in leg_rows}
    ordered_leg_names = [name for name in _leg_order_for_structure(structure, leg_rows) if name in leg_map]
    canonical_meta["Ordered_Leg_Names"] = "|".join(ordered_leg_names)

    def g(leg, field, default=np.nan):
        return leg_map.get(leg, {}).get(field, default)

    def leg_dict(leg):
        return leg_map.get(leg, {})

    def entry_ts(leg):
        return _first_present(leg_dict(leg), ["entry_exec_ts", "entry_ts"], default=np.nan)

    def entry_px(leg):
        return g(leg, "entry_price")

    def exit_ts(leg):
        return _first_present(leg_dict(leg), ["exit_exec_ts", "exit_ts"], default=np.nan)

    def exit_px(leg):
        return g(leg, "exit_price")

    feature_cols = [
        "rv_slow", "rv_slow_lag1", "rv_slow_lag2", "rv_slow_lag3",
        "iv", "skew", "skew_10d", "call_iv_slope", "put_iv_slope",
        "one_std", "dte_std", "gap", "orb_1_std",
        "ivp_12m", "rvs_minus_iv", "rv_slow_over_rv_fast_1_3",
        "VIX_At_Entry",
    ]
    feature_vals = {col: _row_get(trade_row, col, np.nan) for col in feature_cols}
    feature_vals["VIX_At_Entry"] = _first_present(
        trade_row,
        ["VIX_At_Entry", "vix_at_entry"],
        default=feature_vals["VIX_At_Entry"],
    )

    short_ce_strike = _first_present(trade_row, ["short_ce_strike", "Short_CE_Strike"], default=np.nan)
    short_pe_strike = _first_present(trade_row, ["short_pe_strike", "Short_PE_Strike"], default=np.nan)
    if is_batman:
        long_ce_strike = _first_present(
            trade_row,
            ["outer_long_ce_strike", "OuterLongCE_Strike", "OuterLongCEStrike", "long_ce_strike", "Long_CE_Strike"],
            default=np.nan,
        )
        long_pe_strike = _first_present(
            trade_row,
            ["outer_long_pe_strike", "OuterLongPE_Strike", "OuterLongPEStrike", "long_pe_strike", "Long_PE_Strike"],
            default=np.nan,
        )
    else:
        long_ce_strike = _first_present(
            trade_row,
            ["long_ce_strike", "Long_CE_Strike", "outer_long_ce_strike"],
            default=np.nan,
        )
        long_pe_strike = _first_present(
            trade_row,
            ["long_pe_strike", "Long_PE_Strike", "outer_long_pe_strike"],
            default=np.nan,
        )

    entry_cashflows = []
    exit_cashflows = []
    pnl_values = []
    for leg_name in ordered_leg_names:
        leg = leg_map[leg_name]
        entry_cashflows.append(compute_leg_cashflow(leg.get("side"), leg.get("entry_price"), leg.get("qty"), leg.get("lot_size")))
        exit_side = "BUY" if leg.get("side") == "SELL" else "SELL"
        exit_cashflows.append(compute_leg_cashflow(exit_side, leg.get("exit_price"), leg.get("qty"), leg.get("lot_size")))
        pnl_values.append(leg.get("gross_pnl_total"))

    def signed_sum_or_nan(values):
        vals = [safe_float(v) for v in values]
        if all(pd.isna(v) for v in vals):
            return np.nan
        return float(np.nansum(vals))

    total_entry_value = signed_sum_or_nan(entry_cashflows)
    total_exit_value = signed_sum_or_nan(exit_cashflows)
    total_pnl = signed_sum_or_nan(pnl_values)

    trade_details = {
        "Trade_ID": trade_row["trade_id"],
        **canonical_meta,
        "Trade_Day": trade_row["trade_day"],
        "Signal_Entry_TS": trade_row["entry_ts"],
        "DTE": trade_row["dte"],
        "Expiry_Date": trade_row["expiry_date"],
        "VIX_At_Entry": feature_vals["VIX_At_Entry"],
        "Spot_At_Entry": trade_row["spot_at_entry"],
        "Regime": _row_get(trade_row, "rc_regime", np.nan),
        "RoundValue": trade_row["roundvalue"],
        "ShortDteStdMultiplier": trade_row.get("short_dte_std_multiplier"),
        "CESelectionMode": trade_row.get("ce_selection_mode"),
        "PESelectionMode": trade_row.get("pe_selection_mode"),
        "CECandidateCount": trade_row.get("ce_candidate_count"),
        "PECandidateCount": trade_row.get("pe_candidate_count"),
        "CETargetPrice": trade_row.get("ce_target_price"),
        "PETargetPrice": trade_row.get("pe_target_price"),
        "CEObservedMinPrice": trade_row.get("ce_min_candidate_price"),
        "CEObservedMaxPrice": trade_row.get("ce_max_candidate_price"),
        "PEObservedMinPrice": trade_row.get("pe_min_candidate_price"),
        "PEObservedMaxPrice": trade_row.get("pe_max_candidate_price"),
        "CEAtmBoundary": trade_row.get("atm_ce_boundary"),
        "PEAtmBoundary": trade_row.get("atm_pe_boundary"),
        "Short_CE_Strike": short_ce_strike,
        "Short_PE_Strike": short_pe_strike,
        "Long_CE_Strike": long_ce_strike,
        "Long_PE_Strike": long_pe_strike,
        "Short_CE_Entry_TS": entry_ts("SHORTCE"),
        "Short_CE_Entry_Price": entry_px("SHORTCE"),
        "Short_CE_Exit_TS": exit_ts("SHORTCE"),
        "Short_CE_Exit_Price": exit_px("SHORTCE"),
        "Short_PE_Entry_TS": entry_ts("SHORTPE"),
        "Short_PE_Entry_Price": entry_px("SHORTPE"),
        "Short_PE_Exit_TS": exit_ts("SHORTPE"),
        "Short_PE_Exit_Price": exit_px("SHORTPE"),
        "Long_CE_Entry_TS": entry_ts("OUTERLONGCE" if is_batman else "LONGCE"),
        "Long_CE_Entry_Price": entry_px("OUTERLONGCE" if is_batman else "LONGCE"),
        "Long_CE_Exit_TS": exit_ts("OUTERLONGCE" if is_batman else "LONGCE"),
        "Long_CE_Exit_Price": exit_px("OUTERLONGCE" if is_batman else "LONGCE"),
        "Long_PE_Entry_TS": entry_ts("OUTERLONGPE" if is_batman else "LONGPE"),
        "Long_PE_Entry_Price": entry_px("OUTERLONGPE" if is_batman else "LONGPE"),
        "Long_PE_Exit_TS": exit_ts("OUTERLONGPE" if is_batman else "LONGPE"),
        "Long_PE_Exit_Price": exit_px("OUTERLONGPE" if is_batman else "LONGPE"),
        "CE_Exit_Reason": _exit_reason_label(g("SHORTCE", "exit_reason", None)),
        "PE_Exit_Reason": _exit_reason_label(g("SHORTPE", "exit_reason", None)),
        "SecondLegSide": None,
        "SecondLegEntryTS": pd.NaT,
        "SecondLegEntryCredit": np.nan,
        "SecondLegExitTS": pd.NaT,
        "SecondLegExitReason": None,
        "SecondLegPnL": np.nan,
        "SL_Params": _format_sl_params(trade_row),
        "TGT_Params": (
            f"{safe_float(trade_row.get('rc_combined_target_pct')):.0f}%"
            if not pd.isna(safe_float(trade_row.get("rc_combined_target_pct")))
            else np.nan
        ),
        "TGT_Amount": (
            total_entry_value * safe_float(trade_row.get("rc_combined_target_pct")) / 100.0
            if (
                not pd.isna(total_entry_value)
                and not pd.isna(safe_float(trade_row.get("rc_combined_target_pct")))
            )
            else np.nan
        ),
        "Total_Entry_Value": total_entry_value,
        "Total_Exit_Value": total_exit_value,
        "Total_PnL": total_pnl,
        "Trade_Status": trade_status,
        "Skip_Reason": skip_reason if skip_reason else "",
        **{k: v for k, v in feature_vals.items() if k != "VIX_At_Entry"},
    }
    if structure == "IRON_CONDOR":
        trade_details.update({
            "Short_CE_Strike": short_ce_strike,
            "Short_PE_Strike": short_pe_strike,
            "Long_CE_Strike": long_ce_strike,
            "Long_PE_Strike": long_pe_strike,
        })
    if structure == "BATMAN":
        trade_details.update({
            "InnerLongCEStrike": trade_row.get("inner_long_ce_strike"),
            "ShortCEStrike": trade_row.get("short_ce_strike"),
            "OuterLongCEStrike": trade_row.get("outer_long_ce_strike"),
            "InnerLongPEStrike": trade_row.get("inner_long_pe_strike"),
            "ShortPEStrike": trade_row.get("short_pe_strike"),
            "OuterLongPEStrike": trade_row.get("outer_long_pe_strike"),
            "Long_CE_Strike": long_ce_strike,
            "Long_PE_Strike": long_pe_strike,
        })
    for leg_name in ordered_leg_names:
        leg = leg_map[leg_name]
        prefix = _leg_output_prefix(leg_name)
        trade_details.update({
            f"{prefix}Strike": leg.get("strike"),
            f"{prefix}Qty": leg.get("qty"),
            f"{prefix}Side": leg.get("side"),
            f"{prefix}EntryTS": leg.get("entry_exec_ts"),
            f"{prefix}EntryPrice": leg.get("entry_price"),
            f"{prefix}ExitTS": leg.get("exit_exec_ts"),
            f"{prefix}ExitPrice": leg.get("exit_price"),
            f"{prefix}PnL": leg.get("gross_pnl_total"),
        })

    _2l_exit_reasons = {"2L_SL_EXIT", "2L_TGT_EXIT"}
    _2l_legs = [leg for leg in leg_rows if leg.get("exit_reason") in _2l_exit_reasons]
    if not _2l_legs:
        _sl_sides = {
            leg.get("option_type")
            for leg in leg_rows
            if "SL_REGIME" in str(leg.get("exit_reason", ""))
            and leg.get("entry_status") == "FILLED"
        }
        _expiry_sides = {
            leg.get("option_type")
            for leg in leg_rows
            if leg.get("exit_reason") == cfg.EXIT_REASON_DEFAULT
            and leg.get("entry_status") == "FILLED"
        }
        _2l_side_candidate = _expiry_sides - _sl_sides
        if _sl_sides and _2l_side_candidate:
            _2l_side = _2l_side_candidate.pop()
            _2l_legs = [
                leg for leg in leg_rows
                if leg.get("option_type") == _2l_side
                and leg.get("entry_status") == "FILLED"
            ]
            if _2l_legs:
                trade_details["SecondLegSide"] = _2l_side
                trade_details["SecondLegExitTS"] = _2l_legs[0].get("exit_exec_ts", pd.NaT)
                trade_details["SecondLegExitReason"] = cfg.EXIT_REASON_DEFAULT
    else:
        _2l_side = _2l_legs[0].get("option_type")
        trade_details["SecondLegSide"] = _2l_side
        trade_details["SecondLegExitTS"] = _2l_legs[0].get("exit_exec_ts", pd.NaT)
        trade_details["SecondLegExitReason"] = _2l_legs[0].get("exit_reason")

    if trade_details.get("SecondLegSide"):
        _2l_side = trade_details["SecondLegSide"]
        _sl_leg = next(
            (
                l for l in leg_rows
                if "SL_REGIME" in str(l.get("exit_reason", ""))
                and l.get("entry_status") == "FILLED"
            ),
            None,
        )
        if _sl_leg:
            trade_details["SecondLegEntryTS"] = _sl_leg.get("exit_exec_ts", pd.NaT)

        _2l_credit_vals = [
            safe_float(leg.get("second_leg_entry_credit"))
            for leg in leg_rows
            if leg.get("option_type") == _2l_side
            and leg.get("entry_status") == "FILLED"
            and not pd.isna(safe_float(leg.get("second_leg_entry_credit")))
        ]
        if _2l_credit_vals:
            trade_details["SecondLegEntryCredit"] = _2l_credit_vals[0]

        _2l_pnl = 0.0
        _2l_pnl_valid = True
        for leg in leg_rows:
            if leg.get("option_type") != _2l_side or leg.get("entry_status") != "FILLED":
                continue
            ep = safe_float(leg.get("second_leg_entry_price"))
            if pd.isna(ep):
                ep = safe_float(leg.get("entry_price"))
            xp = safe_float(leg.get("exit_price"))
            qty = safe_float(leg.get("qty", 1))
            lot = safe_float(leg.get("lot_size", cfg.LOT_SIZE))
            if pd.isna(ep) or pd.isna(xp):
                _2l_pnl_valid = False
                break
            if leg["side"] == "SELL":
                _2l_pnl += (ep - xp) * qty * lot
            else:
                _2l_pnl += (xp - ep) * qty * lot
        if _2l_pnl_valid:
            trade_details["SecondLegPnL"] = _2l_pnl

    trade_pnl = {
        "Trade_ID": trade_row["trade_id"],
        **canonical_meta,
        "Trade_Day": trade_row["trade_day"],
        "Signal_Entry_TS": trade_row["entry_ts"],
        "Expiry_Date": trade_row["expiry_date"],
        "DTE": trade_row["dte"],
        "Spot_At_Entry": trade_row["spot_at_entry"],
        "RoundValue": trade_row["roundvalue"],
        "ShortDteStdMultiplier": trade_row.get("short_dte_std_multiplier"),
        "Total_Entry_Value": total_entry_value,
        "Total_Exit_Value": total_exit_value,
        "Total_PnL": total_pnl,
        "Trade_Status": trade_status,
        "Skip_Reason": skip_reason,
    }
    for leg_name in ordered_leg_names:
        trade_pnl[f"{_leg_output_prefix(leg_name)}PnL"] = leg_map[leg_name].get("gross_pnl_total")
    return trade_details, trade_pnl


# ============================================================
# EXIT RULES ENGINE
# Inserted as a self-contained layer.  Nothing else is changed.
# ============================================================

def _er_side_net_credit(entry_legs, option_type):
    """
    CE-side or PE-side net credit at entry.
    = sum(cashflow of each filled leg on this option_type side)
    Cashflow: +price*qty*lot for SELL, -price*qty*lot for BUY.
    Returns float (positive = net credit) or np.nan if no filled legs.
    """
    vals = []
    for leg in entry_legs:
        if leg.get("entry_status") != "FILLED":
            continue
        if leg.get("option_type") != option_type:
            continue
        cf = compute_leg_cashflow(
            leg["side"], leg.get("entry_price"), leg.get("qty"), leg.get("lot_size")
        )
        if not pd.isna(cf):
            vals.append(cf)
    if not vals:
        return np.nan
    return float(np.nansum(vals))


def _er_short_price_for_side(entry_legs, option_type, mark_ts, option_source):
    """
    Resolve the current mark price of the SHORT leg on one option_type side.
    Returns (leg_dict, mark_price) for the first SELL leg found on that side.
    Returns (None, np.nan) if not available.
    """
    for leg in entry_legs:
        if leg.get("entry_status") != "FILLED":
            continue
        if leg.get("option_type") != option_type:
            continue
        if leg.get("side") != "SELL":
            continue
        mark = resolve_mark_for_leg(leg, mark_ts, option_source)
        px = mark.get("mark_price", np.nan) if mark else np.nan
        return leg, px
    return None, np.nan


def _er_remaining_value(entry_legs, mark_ts, option_source):
    """
    Sum of current mark prices of ALL legs, weighted by qty*lot_size,
    using the sign convention of what it would cost to CLOSE the position.
    SELL leg  => cost to close = +mark_price * qty * lot  (we buy back)
    BUY  leg  => cost to close = -mark_price * qty * lot  (we sell back)
    Returns float or np.nan if any filled leg is missing its mark.
    """
    total = 0.0
    for leg in entry_legs:
        if leg.get("entry_status") != "FILLED":
            continue
        mark = resolve_mark_for_leg(leg, mark_ts, option_source)
        px = mark.get("mark_price", np.nan) if mark else np.nan
        if pd.isna(px):
            return np.nan
        qty = safe_float(leg.get("qty", 1))
        lot = safe_float(leg.get("lot_size", 1))
        if leg["side"] == "SELL":
            total += px * qty * lot
        else:
            total -= px * qty * lot
    return total


def _er_exit_one_side(entry_legs, option_type, triggered_ts, exit_reason, option_source):
    """
    Build exit dicts for ALL legs on the given option_type side (both
    SELL and BUY legs on that side are exited).  Legs on the OTHER side
    are left with exit_status=PENDING so the normal expiry-exit path
    handles them.

    Returns list of leg dicts (same structure as finalize_leg_rows input).
    """
    out_legs = []
    max_fb = getattr(cfg, "SLTOP_MAX_FALLBACK_MIN", 15)
    for leg in entry_legs:
        out = dict(leg)
        if leg.get("entry_status") != "FILLED":
            out.update(
                exit_requested_ts=pd.NaT,
                exit_exec_ts=pd.NaT,
                exit_price=np.nan,
                exit_fill_mode="MISSING",
                exit_fill_offset_min=np.nan,
                exit_status="SKIPPED",
                exit_skip_reason="ENTRY_NOT_FILLED",
                exit_reason=exit_reason,
            )
            out_legs.append(out)
            continue

        if leg.get("option_type") != option_type:
            out.update(
                exit_requested_ts=pd.NaT,
                exit_exec_ts=pd.NaT,
                exit_price=np.nan,
                exit_fill_mode="MISSING",
                exit_fill_offset_min=np.nan,
                exit_status="PENDING",
                exit_skip_reason=None,
                exit_reason=None,
            )
            out_legs.append(out)
            continue

        res = resolve_option_bar(
            trade_day=pd.Timestamp(triggered_ts).strftime("%Y-%m-%d"),
            expiry_date=leg["expiry_date"],
            strike=leg["strike"],
            option_type=leg["option_type"],
            requested_ts=triggered_ts,
            option_source=option_source,
            allow_nearest=True,
            max_fallback_min=max_fb,
        )
        exit_px = (
            extract_bar_price(res, cfg.EXIT_PRICE_FIELD)
            if res.get("status") == "FOUND" else np.nan
        )
        if pd.isna(exit_px):
            exit_px = get_forced_option_price(triggered_ts, leg["expiry_date"])

        out.update(
            exit_requested_ts=triggered_ts,
            exit_exec_ts=res.get("actual_ts", triggered_ts)
            if res.get("status") == "FOUND" else triggered_ts,
            exit_price=exit_px,
            exit_fill_mode=res.get("fill_mode", "MISSING"),
            exit_fill_offset_min=res.get("fill_offset_min", np.nan),
            exit_status="FILLED" if not pd.isna(exit_px) else "FAILED",
            exit_skip_reason=None,
            exit_reason=exit_reason,
        )
        out_legs.append(out)
    return out_legs


def _er_exit_all(entry_legs, triggered_ts, exit_reason, option_source):
    """
    Build exit dicts for ALL legs at triggered_ts.
    """
    out_legs = []
    max_fb = getattr(cfg, "SLTOP_MAX_FALLBACK_MIN", 15)
    for leg in entry_legs:
        out = dict(leg)
        if leg.get("entry_status") != "FILLED":
            out.update(
                exit_requested_ts=pd.NaT,
                exit_exec_ts=pd.NaT,
                exit_price=np.nan,
                exit_fill_mode="MISSING",
                exit_fill_offset_min=np.nan,
                exit_status="SKIPPED",
                exit_skip_reason="ENTRY_NOT_FILLED",
                exit_reason=exit_reason,
            )
            out_legs.append(out)
            continue

        res = resolve_option_bar(
            trade_day=pd.Timestamp(triggered_ts).strftime("%Y-%m-%d"),
            expiry_date=leg["expiry_date"],
            strike=leg["strike"],
            option_type=leg["option_type"],
            requested_ts=triggered_ts,
            option_source=option_source,
            allow_nearest=True,
            max_fallback_min=max_fb,
        )
        exit_px = (
            extract_bar_price(res, cfg.EXIT_PRICE_FIELD)
            if res.get("status") == "FOUND" else np.nan
        )
        if pd.isna(exit_px):
            exit_px = get_forced_option_price(triggered_ts, leg["expiry_date"])

        out.update(
            exit_requested_ts=triggered_ts,
            exit_exec_ts=res.get("actual_ts", triggered_ts)
            if res.get("status") == "FOUND" else triggered_ts,
            exit_price=exit_px,
            exit_fill_mode=res.get("fill_mode", "MISSING"),
            exit_fill_offset_min=res.get("fill_offset_min", np.nan),
            exit_status="FILLED" if not pd.isna(exit_px) else "FAILED",
            exit_skip_reason=None,
            exit_reason=exit_reason,
        )
        out_legs.append(out)
    return out_legs


# ============================================================
# PATCH 4 — SPOT BAR HELPERS (for SD-type SL)
# ============================================================

_SPOT_CACHE: dict = {}   # trade_day -> timestamp-indexed DataFrame
_INDEX_DF_CACHE: dict = {}   # "__full__" -> full index CSV DataFrame


def _load_index_csv_for_day(trade_day: str) -> pd.DataFrame:
    """
    Load NIFTY index CSV (cfg.INDEX_DATA_PATH) filtered to trade_day.
    Returns a timestamp-indexed DataFrame with a Close column, or empty.
    """
    index_path = getattr(cfg, "INDEX_DATA_PATH", None)
    if not index_path or not os.path.isfile(str(index_path)):
        return pd.DataFrame()

    # Load full CSV once and cache it globally.
    if "__full__" not in _INDEX_DF_CACHE:
        try:
            raw = pd.read_csv(index_path)
            raw = normalize_column_names(raw)

            lower_cols = {c.lower(): c for c in raw.columns}
            timestamp_col = lower_cols.get("timestamp")
            date_col = lower_cols.get("date")
            time_col = lower_cols.get("time")

            if timestamp_col:
                raw["timestamp"] = pd.to_datetime(raw[timestamp_col], errors="coerce")
            elif date_col and time_col:
                raw["timestamp"] = pd.to_datetime(
                    raw[date_col].astype(str).str.strip().str[:10]
                    + " "
                    + raw[time_col].astype(str).str.strip().str[:5],
                    errors="coerce",
                )
            else:
                return pd.DataFrame()

            # Standardise close column name.
            close_col = next(
                (c for c in raw.columns if c.lower() in {"close", "ltp", "last"}),
                None,
            )
            if close_col is None:
                return pd.DataFrame()

            raw["Close"] = pd.to_numeric(raw[close_col], errors="coerce")
            raw = raw.dropna(subset=["timestamp", "Close"]).copy()
            raw["_trade_day"] = raw["timestamp"].dt.strftime("%Y-%m-%d")
            raw = raw.set_index("timestamp", drop=False).sort_index()
            _INDEX_DF_CACHE["__full__"] = raw
        except Exception as exc:
            warnings.warn(f"_load_index_csv_for_day: failed to load {index_path} - {exc}")
            _INDEX_DF_CACHE["__full__"] = pd.DataFrame()

    full = _INDEX_DF_CACHE["__full__"]
    if full.empty:
        return pd.DataFrame()

    return full[full["_trade_day"] == trade_day][["Close"]].copy()


def _get_spot_bars_for_day(trade_day: str, option_source: dict) -> pd.DataFrame:
    """
    Lazy-load spot bars for trade_day.
    Priority:
      1. cfg.INDEX_DATA_PATH  (NIFTY index CSV)
      2. PKL folder IDX rows  (legacy fallback)
    """
    if trade_day in _SPOT_CACHE:
        return _SPOT_CACHE[trade_day]

    # Priority 1: index CSV.
    df = _load_index_csv_for_day(trade_day)
    if not df.empty:
        _SPOT_CACHE[trade_day] = df
        return df

    # Priority 2: PKL folder IDX rows.
    folder = (
        option_source.get("dataset")
        if option_source.get("mode") == "lazy"
        else getattr(cfg, "PKL_FOLDER", None)
    )

    if not folder or not os.path.isdir(str(folder)):
        _SPOT_CACHE[trade_day] = pd.DataFrame()
        return pd.DataFrame()

    raw = _load_pkl_month(folder, trade_day)
    if raw.empty:
        _SPOT_CACHE[trade_day] = pd.DataFrame()
        return pd.DataFrame()

    raw = normalize_column_names(raw)
    required = ["Date", "Time", "Type", "Close"]
    if any(c not in raw.columns for c in required):
        _SPOT_CACHE[trade_day] = pd.DataFrame()
        return pd.DataFrame()

    df = raw[
        (raw["Date"].astype(str).str.strip().str[:10] == trade_day)
        & (raw["Type"].astype(str).str.strip().str.upper() == "IDX")
    ].copy()

    if df.empty:
        _SPOT_CACHE[trade_day] = pd.DataFrame()
        return pd.DataFrame()

    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str).str.strip().str[:10]
        + " "
        + df["Time"].astype(str).str.strip().str[:5],
        errors="coerce",
    )
    df = (
        df.dropna(subset=["timestamp", "Close"])
        .sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )
    df = df.set_index("timestamp", drop=False)[["Close"]]
    _SPOT_CACHE[trade_day] = df
    return df


def _resolve_spot_at_ts(
    trade_day: str,
    ts: pd.Timestamp,
    option_source: dict,
    max_fallback_min: int = 3,
) -> float:
    """Return NIFTY spot close at ts ± max_fallback_min minutes."""
    spot_df = _get_spot_bars_for_day(trade_day, option_source)
    if spot_df.empty:
        return np.nan
    ts = pd.Timestamp(ts)
    if ts in spot_df.index:
        return safe_float(spot_df.loc[ts, "Close"])
    bar = get_option_bar_nearest_within(spot_df, ts, max_fallback_min)
    return safe_float(bar.get("Close", np.nan)) if bar is not None else np.nan


# ============================================================
# PATCH 4 — SL-BY-REGIME EXIT ENGINE
# ============================================================

def _sl_check_ce_side(
    mark_ts: pd.Timestamp,
    trade_row: dict,
    ce_short_leg: dict,
    sl_type: str,
    sl_value: float,
    option_source: dict,
    max_fallback_min: int,
) -> bool:
    """True if CE-side SL is triggered at mark_ts."""
    if sl_type == "PCT_CREDIT":
        ce_entry = safe_float(ce_short_leg.get("entry_price"))
        if pd.isna(ce_entry) or ce_entry <= 0:
            return False
        mark   = resolve_mark_for_leg(ce_short_leg, mark_ts, option_source)
        ce_px  = safe_float(mark.get("mark_price", np.nan)) if mark else np.nan
        return (not pd.isna(ce_px)) and ce_px >= ce_entry * sl_value

    if sl_type == "SD":
        dte_std   = safe_float(trade_row.get("dte_std"))
        ce_strike = safe_float(trade_row.get("short_ce_strike"))
        if any(pd.isna(v) for v in [dte_std, ce_strike, sl_value]):
            return False
        threshold = ce_strike + sl_value * dte_std
        trade_day = normalize_date_str(trade_row.get("trade_day"))
        spot = _resolve_spot_at_ts(trade_day, mark_ts, option_source, max_fallback_min)
        return (not pd.isna(spot)) and spot >= threshold

    return False


def _sl_check_pe_side(
    mark_ts: pd.Timestamp,
    trade_row: dict,
    pe_short_leg: dict,
    sl_type: str,
    sl_value: float,
    option_source: dict,
    max_fallback_min: int,
) -> bool:
    """True if PE-side SL is triggered at mark_ts."""
    if sl_type == "PCT_CREDIT":
        pe_entry = safe_float(pe_short_leg.get("entry_price"))
        if pd.isna(pe_entry) or pe_entry <= 0:
            return False
        mark  = resolve_mark_for_leg(pe_short_leg, mark_ts, option_source)
        pe_px = safe_float(mark.get("mark_price", np.nan)) if mark else np.nan
        return (not pd.isna(pe_px)) and pe_px >= pe_entry * sl_value

    if sl_type == "SD":
        dte_std   = safe_float(trade_row.get("dte_std"))
        pe_strike = safe_float(trade_row.get("short_pe_strike"))
        if any(pd.isna(v) for v in [dte_std, pe_strike, sl_value]):
            return False
        threshold = pe_strike - sl_value * dte_std
        trade_day = normalize_date_str(trade_row.get("trade_day"))
        spot = _resolve_spot_at_ts(trade_day, mark_ts, option_source, max_fallback_min)
        return (not pd.isna(spot)) and spot <= threshold

    return False


def _both_sides_fully_open(filled_legs):
    """
    Returns True only when at least one FILLED+PENDING CE leg and at least
    one FILLED+PENDING PE leg are both still open.
    """
    ce_open = any(
        leg.get("entry_status") == "FILLED"
        and leg.get("exit_status", "PENDING") == "PENDING"
        and normalize_option_type(leg.get("option_type")) == "CE"
        for leg in filled_legs
    )
    pe_open = any(
        leg.get("entry_status") == "FILLED"
        and leg.get("exit_status", "PENDING") == "PENDING"
        and normalize_option_type(leg.get("option_type")) == "PE"
        for leg in filled_legs
    )
    return ce_open and pe_open


def run_sl_by_regime_engine(entry_legs: list, trade_row: dict, option_source: dict):
    """
    Patch 4 — SL-by-regime one-side exit engine.

    Scans the SL_EVAL_TIMEFRAME (default 1min) grid from entry to expiry.
    On first SL trigger: cuts triggered side at that bar, other side holds to expiry.
    Returns (exit_legs | None, exit_reason | None, trigger_ts | NaT).
    """
    filled = [l for l in entry_legs if l.get("entry_status") == "FILLED"]
    if not filled:
        return None, None, pd.NaT

    sl_type   = str(trade_row.get("rc_sl_type",  "PCT_CREDIT")).upper()
    sl_value  = safe_float(trade_row.get("rc_sl_value"))

    # Guard: nothing to do if SL value is invalid
    if pd.isna(sl_value) or sl_value <= 0:
        return None, None, pd.NaT

    ce_short = next(
        (l for l in filled if l.get("option_type") == "CE" and l.get("side") == "SELL"),
        None,
    )
    pe_short = next(
        (l for l in filled if l.get("option_type") == "PE" and l.get("side") == "SELL"),
        None,
    )
    if ce_short is None and pe_short is None:
        return None, None, pd.NaT

    timeframe  = getattr(cfg, "SL_EVAL_TIMEFRAME",       "1min")
    max_fb_min = int(getattr(cfg, "SL_EVAL_MAX_FALLBACK_MIN", 3))
    entry_ts   = pd.Timestamp(trade_row["entry_ts"])
    expiry_ts  = combine_date_time(trade_row["expiry_date"], cfg.EXPIRY_EXIT_TIME)

    mark_grid = build_intratrade_mark_grid(
        entry_ts=entry_ts,
        end_ts=expiry_ts,
        timeframe=timeframe,
        include_final_exit_row=True,
    )
    mark_grid = [ts for ts in mark_grid if ts > entry_ts]

    # Combined Profit Target setup.
    enable_cpt = getattr(cfg, "ENABLE_COMBINED_PROFIT_TARGET", False)
    cpt_pct = safe_float(trade_row.get("rc_combined_target_pct"))
    cpt_abs_target = np.nan

    if enable_cpt and not pd.isna(cpt_pct) and cpt_pct > 0:
        entry_nc_vals = []
        for leg in filled:
            if leg.get("entry_status") != "FILLED" or pd.isna(leg.get("entry_price")):
                continue
            cf = compute_leg_cashflow(
                leg["side"],
                leg["entry_price"],
                leg.get("qty", 1),
                leg.get("lot_size", cfg.LOT_SIZE),
            )
            if not pd.isna(cf):
                entry_nc_vals.append(cf)
        entry_nc = float(np.nansum(entry_nc_vals)) if entry_nc_vals else np.nan
        if not pd.isna(entry_nc) and entry_nc > 0:
            cpt_abs_target = entry_nc * (cpt_pct / 100.0)

    def _check_combined_profit_target(mark_ts):
        if not (
            enable_cpt
            and not pd.isna(cpt_abs_target)
            and _both_sides_fully_open(filled)
        ):
            return None

        current_pnl = 0.0
        pnl_valid = True
        for leg in filled:
            if leg.get("exit_status", "PENDING") != "PENDING":
                entry_cf = compute_leg_cashflow(
                    leg["side"],
                    leg["entry_price"],
                    leg.get("qty", 1),
                    leg.get("lot_size", cfg.LOT_SIZE),
                )
                exit_cf = compute_leg_cashflow(
                    "BUY" if leg["side"] == "SELL" else "SELL",
                    leg.get("exit_price"),
                    leg.get("qty", 1),
                    leg.get("lot_size", cfg.LOT_SIZE),
                )
                if pd.isna(entry_cf) or pd.isna(exit_cf):
                    pnl_valid = False
                    break
                current_pnl += entry_cf + exit_cf
                continue

            mark_res = resolve_option_bar(
                trade_day=pd.Timestamp(mark_ts).strftime("%Y-%m-%d"),  # use bar day, not entry day
                expiry_date=leg["expiry_date"],
                strike=leg["strike"],
                option_type=leg["option_type"],
                requested_ts=mark_ts,
                option_source=option_source,
                allow_nearest=True,
                max_fallback_min=max_fb_min,
            )
            mark_price = extract_bar_price(mark_res, cfg.EXIT_PRICE_FIELD)
            if pd.isna(mark_price):
                mark_price = get_forced_option_price(mark_ts, leg["expiry_date"])
            if pd.isna(mark_price):
                pnl_valid = False
                break

            entry_cf = compute_leg_cashflow(
                leg["side"],
                leg["entry_price"],
                leg.get("qty", 1),
                leg.get("lot_size", cfg.LOT_SIZE),
            )
            close_cf = compute_leg_cashflow(
                "BUY" if leg["side"] == "SELL" else "SELL",
                mark_price,
                leg.get("qty", 1),
                leg.get("lot_size", cfg.LOT_SIZE),
            )
            if pd.isna(entry_cf) or pd.isna(close_cf):
                pnl_valid = False
                break
            current_pnl += entry_cf + close_cf

        if pnl_valid and current_pnl >= cpt_abs_target:
            exit_legs = _er_exit_all(entry_legs, mark_ts, "COMBINED_TP", option_source)
            return exit_legs, "COMBINED_TP", mark_ts

        return None

    # PCT_CREDIT: trigger on total position close value, then cut the
    # losing-short side. All other sl_type values keep the original
    # per-side short-premium check unchanged.
    if sl_type == "PCT_CREDIT":
        # Total entry net credit: sum cashflows of every filled leg at entry.
        # compute_leg_cashflow returns +ve for SELL legs and -ve for BUY legs.
        entry_cf_vals = []
        for leg in filled:
            cf = compute_leg_cashflow(
                leg.get("side"),
                leg.get("entry_price"),
                leg.get("qty", 1),
                leg.get("lot_size", cfg.LOT_SIZE),
            )
            if not pd.isna(cf):
                entry_cf_vals.append(cf)

        if not entry_cf_vals:
            return None, None, pd.NaT

        total_entry_credit = float(np.nansum(entry_cf_vals))
        if total_entry_credit <= 0:
            # Net debit strategy - PCT_CREDIT SL is not meaningful.
            return None, None, pd.NaT

        sl_threshold = total_entry_credit * sl_value

        for mark_ts in mark_grid:
            cpt_result = _check_combined_profit_target(mark_ts)
            if cpt_result is not None:
                return cpt_result

            # Step 1: compute current cost-to-close the full position.
            current_close_value = _er_remaining_value(entry_legs, mark_ts, option_source)
            if pd.isna(current_close_value):
                continue
            if current_close_value < sl_threshold:
                continue

            # Step 2: total threshold breached - find the losing short side.
            _, ce_mark = _er_short_price_for_side(entry_legs, "CE", mark_ts, option_source)
            _, pe_mark = _er_short_price_for_side(entry_legs, "PE", mark_ts, option_source)

            ce_entry = safe_float(ce_short.get("entry_price")) if ce_short else np.nan
            pe_entry = safe_float(pe_short.get("entry_price")) if pe_short else np.nan

            ce_short_loss = (
                safe_float(ce_mark) - ce_entry
                if not pd.isna(ce_mark) and not pd.isna(ce_entry)
                else np.nan
            )
            pe_short_loss = (
                safe_float(pe_mark) - pe_entry
                if not pd.isna(pe_mark) and not pd.isna(pe_entry)
                else np.nan
            )

            # Higher loss gets cut. Tie-break, equal, or both NaN: PE first.
            if pd.isna(ce_short_loss) and pd.isna(pe_short_loss):
                cut_side = "PE"
            elif pd.isna(ce_short_loss):
                cut_side = "PE"
            elif pd.isna(pe_short_loss):
                cut_side = "CE"
            elif ce_short_loss >= pe_short_loss:
                cut_side = "CE"
            else:
                cut_side = "PE"

            reason = "PE_SL_REGIME" if cut_side == "PE" else "CE_SL_REGIME"
            surviving_side = "CE" if cut_side == "PE" else "PE"

            partial = _er_exit_one_side(entry_legs, cut_side, mark_ts, reason, option_source)
            surviving_legs = [l for l in partial if l.get("exit_status") == "PENDING"]
            exit_legs = run_second_leg_engine(
                surviving_side=surviving_side,
                surviving_legs=surviving_legs,
                all_legs_after_sl=partial,
                trade_row=trade_row,
                first_sl_ts=mark_ts,
                option_source=option_source,
            )
            return exit_legs, reason, mark_ts

        return None, None, pd.NaT

    # Original per-side short-premium check for SD and any other sl_type.
    for mark_ts in mark_grid:
        cpt_result = _check_combined_profit_target(mark_ts)
        if cpt_result is not None:
            return cpt_result

        ce_hit = (
            ce_short is not None
            and _sl_check_ce_side(
                mark_ts, trade_row, ce_short, sl_type, sl_value, option_source, max_fb_min
            )
        )
        pe_hit = (
            pe_short is not None
            and _sl_check_pe_side(
                mark_ts, trade_row, pe_short, sl_type, sl_value, option_source, max_fb_min
            )
        )

        if not ce_hit and not pe_hit:
            continue

        # Tie-break: both triggered on same bar -> cut PE first.
        cut_side = "PE" if pe_hit else "CE"
        reason = "PE_SL_REGIME" if cut_side == "PE" else "CE_SL_REGIME"
        surviving_side = "CE" if cut_side == "PE" else "PE"

        partial = _er_exit_one_side(entry_legs, cut_side, mark_ts, reason, option_source)
        surviving_legs = [l for l in partial if l.get("exit_status") == "PENDING"]
        exit_legs = run_second_leg_engine(
            surviving_side=surviving_side,
            surviving_legs=surviving_legs,
            all_legs_after_sl=partial,
            trade_row=trade_row,
            first_sl_ts=mark_ts,
            option_source=option_source,
        )
        return exit_legs, reason, mark_ts

    # No trigger found — caller will fall through to standard expiry exit
    return None, None, pd.NaT


def _resolve_one_std_at_ts(trade_day, mark_ts, option_source, max_fb_min=3):
    """
    Re-read one_std context from the live index bars at mark_ts.
    Returns np.nan if unavailable so callers can fall back gracefully.
    """
    spot_now = _resolve_spot_at_ts(
        trade_day,
        mark_ts,
        option_source,
        max_fallback_min=max_fb_min,
    )
    return safe_float(spot_now)


def run_second_leg_engine(
    surviving_side: str,
    surviving_legs: list,
    all_legs_after_sl: list,
    trade_row: dict,
    first_sl_ts: pd.Timestamp,
    option_source: dict,
) -> list:
    """
    Manage the surviving side independently after the first side is cut by SL.
    Checks the second-leg target first, then the second-leg SL, on each 1min bar.
    """
    def _resolve_pending_to_expiry(legs):
        expiry_resolved = []
        for leg in legs:
            if leg.get("exit_status") == "PENDING":
                expiry_resolved.append(resolve_expiry_exit_for_leg(leg, option_source))
            else:
                expiry_resolved.append(leg)
        return expiry_resolved

    def _exit_pending_at(legs, mark_ts, exit_reason):
        bar_day = pd.Timestamp(mark_ts).strftime("%Y-%m-%d")
        exit_legs_updated = []
        for leg in legs:
            if leg.get("exit_status") == "PENDING":
                res2 = resolve_option_bar(
                    trade_day=bar_day,
                    expiry_date=leg["expiry_date"],
                    strike=leg["strike"],
                    option_type=leg["option_type"],
                    requested_ts=mark_ts,
                    option_source=option_source,
                    allow_nearest=True,
                    max_fallback_min=max_fb_min,
                )
                xpx = extract_bar_price(res2, cfg.EXIT_PRICE_FIELD)
                if pd.isna(xpx):
                    xpx = get_forced_option_price(mark_ts, leg["expiry_date"])
                out = dict(leg)
                out.update(
                    exit_requested_ts=mark_ts,
                    exit_exec_ts=res2.get("actual_ts", mark_ts)
                    if res2.get("status") == "FOUND" else mark_ts,
                    exit_price=xpx,
                    exit_fill_mode=res2.get("fill_mode", "MISSING"),
                    exit_fill_offset_min=res2.get("fill_offset_min", np.nan),
                    exit_status="FILLED" if not pd.isna(xpx) else "FAILED",
                    exit_skip_reason=None,
                    exit_reason=exit_reason,
                )
                exit_legs_updated.append(out)
            else:
                exit_legs_updated.append(leg)
        return exit_legs_updated

    if not getattr(cfg, "ENABLE_SECOND_LEG_MANAGEMENT", False):
        return _resolve_pending_to_expiry(all_legs_after_sl)

    regime = safe_float(trade_row.get("rc_regime"))
    regime = int(regime) if not pd.isna(regime) else -1
    second_leg_cfg = getattr(cfg, "REGIME_TO_SECOND_LEG", {}).get(regime, {})
    sl_value_2 = safe_float(second_leg_cfg.get("sl_value_2"))
    target_pct_2 = safe_float(second_leg_cfg.get("target_pct_2"))

    if pd.isna(sl_value_2) or pd.isna(target_pct_2):
        return _resolve_pending_to_expiry(all_legs_after_sl)

    sl_type = str(trade_row.get("rc_sl_type", "PCT_CREDIT")).upper()
    max_fb_min = int(getattr(cfg, "SL_EVAL_MAX_FALLBACK_MIN", 3))
    timeframe = getattr(cfg, "SL_EVAL_TIMEFRAME", "1min")
    expiry_ts = combine_date_time(trade_row["expiry_date"], cfg.EXPIRY_EXIT_TIME)

    surviving_short = next(
        (
            l for l in surviving_legs
            if l.get("option_type") == surviving_side
            and l.get("side") == "SELL"
            and l.get("entry_status") == "FILLED"
        ),
        None,
    )
    if surviving_short is None:
        return _resolve_pending_to_expiry(all_legs_after_sl)

    second_leg_entry_credit = np.nan
    total_2l_credit = 0.0
    credit_valid = True
    leg_2l_entry_px = {}
    first_sl_day = pd.Timestamp(first_sl_ts).strftime("%Y-%m-%d")

    for leg in surviving_legs:
        if leg.get("entry_status") != "FILLED":
            continue
        mark_res = resolve_option_bar(
            trade_day=first_sl_day,
            expiry_date=leg["expiry_date"],
            strike=leg["strike"],
            option_type=leg["option_type"],
            requested_ts=first_sl_ts,
            option_source=option_source,
            allow_nearest=True,
            max_fallback_min=max_fb_min,
        )
        px = extract_bar_price(mark_res, cfg.EXIT_PRICE_FIELD)
        if pd.isna(px):
            px = get_forced_option_price(first_sl_ts, leg["expiry_date"])
        if pd.isna(px):
            credit_valid = False
            break
        leg_2l_entry_px[id(leg)] = px
        qty = safe_float(leg.get("qty", 1))
        lot = safe_float(leg.get("lot_size", cfg.LOT_SIZE))
        cf = compute_leg_cashflow(leg["side"], px, qty, lot)
        if pd.isna(cf):
            credit_valid = False
            break
        total_2l_credit += cf

    if credit_valid and total_2l_credit > 0:
        second_leg_entry_credit = total_2l_credit

    short_2l_entry_px = leg_2l_entry_px.get(id(surviving_short), np.nan)

    enriched_legs_after_sl = []
    for leg in all_legs_after_sl:
        out = dict(leg)
        if out.get("exit_status") == "PENDING":
            out["second_leg_entry_ts"] = first_sl_ts
            out["second_leg_entry_credit"] = second_leg_entry_credit
            out["second_leg_entry_price"] = leg_2l_entry_px.get(id(leg), np.nan)
        enriched_legs_after_sl.append(out)
    all_legs_after_sl = enriched_legs_after_sl
    surviving_legs = [l for l in all_legs_after_sl if l.get("exit_status") == "PENDING"]
    surviving_short = next(
        (
            l for l in surviving_legs
            if l.get("option_type") == surviving_side
            and l.get("side") == "SELL"
            and l.get("entry_status") == "FILLED"
        ),
        surviving_short,
    )

    mark_grid = build_intratrade_mark_grid(
        entry_ts=first_sl_ts,
        end_ts=expiry_ts,
        timeframe=timeframe,
        include_final_exit_row=True,
    )
    mark_grid = [ts for ts in mark_grid if ts > first_sl_ts]

    trade_day = normalize_date_str(trade_row["trade_day"])

    for mark_ts in mark_grid:
        bar_day = pd.Timestamp(mark_ts).strftime("%Y-%m-%d")

        if (
            not pd.isna(second_leg_entry_credit)
            and second_leg_entry_credit > 0
            and not pd.isna(target_pct_2)
        ):
            tgt_abs = second_leg_entry_credit * (target_pct_2 / 100.0)
            running_pnl = 0.0
            pnl_ok = True
            for leg in surviving_legs:
                if leg.get("entry_status") != "FILLED":
                    continue
                mark_res = resolve_option_bar(
                    trade_day=bar_day,
                    expiry_date=leg["expiry_date"],
                    strike=leg["strike"],
                    option_type=leg["option_type"],
                    requested_ts=mark_ts,
                    option_source=option_source,
                    allow_nearest=True,
                    max_fallback_min=max_fb_min,
                )
                px = extract_bar_price(mark_res, cfg.EXIT_PRICE_FIELD)
                if pd.isna(px):
                    px = get_forced_option_price(mark_ts, leg["expiry_date"])
                entry_px_2l = safe_float(leg.get("second_leg_entry_price"))
                if pd.isna(px) or pd.isna(entry_px_2l):
                    pnl_ok = False
                    break
                qty = safe_float(leg.get("qty", 1))
                lot = safe_float(leg.get("lot_size", cfg.LOT_SIZE))
                if leg["side"] == "SELL":
                    running_pnl += (entry_px_2l - px) * qty * lot
                else:
                    running_pnl += (px - entry_px_2l) * qty * lot

            if pnl_ok and running_pnl >= tgt_abs:
                return _exit_pending_at(all_legs_after_sl, mark_ts, "2L_TGT_EXIT")

        sl_triggered = False
        if sl_type == "PCT_CREDIT":
            if not pd.isna(short_2l_entry_px) and short_2l_entry_px > 0:
                mark_res = resolve_option_bar(
                    trade_day=bar_day,
                    expiry_date=surviving_short["expiry_date"],
                    strike=surviving_short["strike"],
                    option_type=surviving_short["option_type"],
                    requested_ts=mark_ts,
                    option_source=option_source,
                    allow_nearest=True,
                    max_fallback_min=max_fb_min,
                )
                cur_px = extract_bar_price(mark_res, cfg.EXIT_PRICE_FIELD)
                if not pd.isna(cur_px) and cur_px >= short_2l_entry_px * sl_value_2:
                    sl_triggered = True

        elif sl_type == "SD":
            spot_at_bar = _resolve_spot_at_ts(
                bar_day,
                mark_ts,
                option_source,
                max_fallback_min=max_fb_min,
            )
            one_std_live = safe_float(trade_row.get("dte_std"))
            spot_at_entry = safe_float(trade_row.get("spot_at_entry"))
            if (
                not pd.isna(spot_at_bar)
                and not pd.isna(spot_at_entry)
                and spot_at_entry > 0
                and not pd.isna(one_std_live)
            ):
                one_std_live = one_std_live * (spot_at_bar / spot_at_entry)

            short_strike = safe_float(surviving_short["strike"])
            if not pd.isna(spot_at_bar) and not pd.isna(short_strike) and not pd.isna(one_std_live):
                if surviving_side == "CE":
                    threshold = short_strike + sl_value_2 * one_std_live
                    sl_triggered = spot_at_bar >= threshold
                else:
                    threshold = short_strike - sl_value_2 * one_std_live
                    sl_triggered = spot_at_bar <= threshold

        if sl_triggered:
            return _exit_pending_at(all_legs_after_sl, mark_ts, "2L_SL_EXIT")

    return _resolve_pending_to_expiry(all_legs_after_sl)


def run_exit_rules_engine(entry_legs, trade_row, option_source):
    """
    Route to run_sl_by_regime_engine (PCT or SD SL types only).
    Returns (exit_legs | None, exit_reason | None, trigger_ts | NaT).
    None means no SL triggered - caller holds all legs to expiry.
    """
    return run_sl_by_regime_engine(entry_legs, trade_row, option_source)


# ============================================================
# END OF EXIT RULES ENGINE
# ============================================================


def _maybe_reclassify_as_combined_tp_at_expiry(exit_legs, entry_legs, trade_row):
    """
    Called only when the SL/CPT engine found no intraday trigger (all legs held
    to expiry).  If ENABLE_COMBINED_PROFIT_TARGET is True AND both sides are
    still marked PENDING at the point the expiry exits were resolved AND the
    final Net_PnL >= cpt_abs_target, reclassify every filled exit leg's
    exit_reason from EXPIRY_EXIT to COMBINED_TP_AT_EXPIRY.

    Returns the (possibly mutated) exit_legs list.
    """
    if not getattr(cfg, "ENABLE_COMBINED_PROFIT_TARGET", False):
        return exit_legs

    cpt_pct = safe_float(trade_row.get("rc_combined_target_pct"))
    if pd.isna(cpt_pct) or cpt_pct <= 0:
        return exit_legs

    # If any leg was cut intraday by SL, do not reclassify
    if any("SL" in str(leg.get("exit_reason") or "").upper() for leg in exit_legs):
        return exit_legs

    # Compute entry net credit across ALL filled legs.
    entry_nc_vals = []
    for leg in entry_legs:
        if leg.get("entry_status") != "FILLED" or pd.isna(leg.get("entry_price")):
            continue
        cf = compute_leg_cashflow(
            leg["side"],
            leg["entry_price"],
            leg.get("qty", 1),
            leg.get("lot_size", cfg.LOT_SIZE),
        )
        if not pd.isna(cf):
            entry_nc_vals.append(cf)
    if not entry_nc_vals:
        return exit_legs
    entry_nc = float(np.nansum(entry_nc_vals))
    if entry_nc <= 0:
        return exit_legs

    cpt_abs_target = entry_nc * (cpt_pct / 100.0)

    # Compute final Net PnL from the expiry exit prices.
    net_pnl = 0.0
    pnl_valid = True
    for leg in exit_legs:
        if leg.get("entry_status") != "FILLED":
            continue
        ep  = safe_float(leg.get("entry_price"))
        xp  = safe_float(leg.get("exit_price"))
        qty = safe_float(leg.get("qty", 1))
        lot = safe_float(leg.get("lot_size", cfg.LOT_SIZE))
        if pd.isna(ep) or pd.isna(xp) or pd.isna(qty) or pd.isna(lot):
            pnl_valid = False
            break
        net_pnl += (ep - xp) * qty * lot if leg["side"] == "SELL" else (xp - ep) * qty * lot

    if not pnl_valid or net_pnl < cpt_abs_target:
        return exit_legs

    # Reclassify all filled exits to COMBINED_TP_AT_EXPIRY.
    out = []
    for leg in exit_legs:
        l = dict(leg)
        if l.get("entry_status") == "FILLED" and l.get("exit_status") == "FILLED":
            l["exit_reason"] = "COMBINED_TP_AT_EXPIRY"
        out.append(l)
    return out


def simulate_one_trade_with_legs(trade_row, option_source):
    legs = build_legs_from_trade_intent(trade_row)
    entry_legs = [resolve_entry_for_leg(leg, option_source) for leg in legs]

    filled = [l for l in entry_legs if l.get("entry_status") == "FILLED"]
    if not filled:
        reason = "; ".join(
            sorted({
                l.get("entry_skip_reason", "UNKNOWN")
                for l in entry_legs
                if l.get("entry_skip_reason")
            })
        )
        final_legs = _finalize_leg_rows(entry_legs, entry_legs)
        details, pnl = build_trade_rows(
            trade_row,
            final_legs,
            trade_status="SKIPPED",
            skip_reason=reason,
        )
        return details, pnl, final_legs

    # Try SL (PCT or SD).
    er_exit_legs, er_reason, er_ts = run_exit_rules_engine(
        entry_legs, trade_row, option_source
    )

    if er_exit_legs is not None:
        # SL/TP triggered; any remaining side is handled inside the regime engine.
        exit_legs = er_exit_legs
    else:
        # No SL triggered - hold ALL legs to expiry.
        exit_legs = [resolve_expiry_exit_for_leg(leg, option_source) for leg in entry_legs]
        # If final Net_PnL >= combined target with both sides open, mark as COMBINED_TP_AT_EXPIRY.
        exit_legs = _maybe_reclassify_as_combined_tp_at_expiry(exit_legs, entry_legs, trade_row)

    failed_exit = [x for x in exit_legs if x["exit_status"] not in ("FILLED", "SKIPPED")]
    final_legs = _finalize_leg_rows(entry_legs, exit_legs)

    if failed_exit and cfg.SKIP_TRADE_IF_ANY_LEG_MISSING:
        reason = "; ".join(sorted({x["exit_skip_reason"] for x in failed_exit if x["exit_skip_reason"]}))
        details, pnl = build_trade_rows(trade_row, final_legs, trade_status="SKIPPED", skip_reason=reason)
        return details, pnl, final_legs

    details, pnl = build_trade_rows(trade_row, final_legs, trade_status="FILLED", skip_reason=None)
    return details, pnl, final_legs


def simulate_one_trade(trade_row, option_source):
    details, pnl, _ = simulate_one_trade_with_legs(trade_row, option_source)
    return details, pnl


# ============================================================
# OPTIONAL INTRATRADE TIMELINE LAYER
# ============================================================
def ceil_timestamp_to_grid(ts, step_min):
    ts = pd.Timestamp(ts)
    floored = ts.floor(f"{step_min}min")
    return floored if floored == ts else (floored + pd.Timedelta(minutes=step_min))


def build_intratrade_mark_grid(entry_ts, end_ts, timeframe, include_final_exit_row=True):
    if pd.isna(entry_ts) or pd.isna(end_ts):
        return []

    entry_ts = pd.Timestamp(entry_ts)
    end_ts = pd.Timestamp(end_ts)
    if end_ts < entry_ts:
        return []

    step_min = parse_timeframe_to_minutes(timeframe)
    start_day = entry_ts.normalize()
    end_day = end_ts.normalize()
    days = pd.date_range(start_day, end_day, freq="D")

    marks = []
    for day in days:
        day_str = day.strftime("%Y-%m-%d")
        sess_start_ts = combine_date_time(day_str, cfg.SESSION_START)
        sess_end_ts = combine_date_time(day_str, cfg.SESSION_END)
        if pd.isna(sess_start_ts) or pd.isna(sess_end_ts):
            continue

        lower = max(sess_start_ts, entry_ts if day == start_day else sess_start_ts)
        upper = min(sess_end_ts, end_ts if day == end_day else sess_end_ts)
        if upper < lower:
            continue

        first_mark = ceil_timestamp_to_grid(lower, step_min)
        if first_mark <= entry_ts:
            first_mark = first_mark + pd.Timedelta(minutes=step_min)

        cur = first_mark
        while cur <= upper:
            marks.append(cur)
            cur += pd.Timedelta(minutes=step_min)

    if include_final_exit_row and end_ts > entry_ts and end_ts not in marks:
        marks.append(end_ts)

    marks = sorted(pd.to_datetime(pd.Series(marks)).drop_duplicates().tolist())
    return marks


def resolve_mark_for_leg(leg_row, mark_ts, option_source):
    strict_after_only = getattr(cfg, "INTRATRADE_STRICT_AFTER_ONLY", False)
    max_fb = getattr(
        cfg,
        "INTRATRADE_MARK_FALLBACK_MIN",
        getattr(cfg, "INTRATRADE_MAX_FALLBACK_MIN", cfg.MAX_EXIT_FALLBACK_MIN),
    )
    allow_nearest_marks = True
    mark_trade_day = pd.Timestamp(mark_ts).strftime("%Y-%m-%d")

    res = resolve_option_bar(
        trade_day=mark_trade_day,
        expiry_date=leg_row["expiry_date"],
        strike=leg_row["strike"],
        option_type=leg_row["option_type"],
        requested_ts=mark_ts,
        option_source=option_source,
        allow_nearest=allow_nearest_marks,
        max_fallback_min=max_fb,
    )
    if (
        strict_after_only
        and res["status"] == "FOUND"
        and not pd.isna(res.get("actual_ts"))
        and pd.Timestamp(res["actual_ts"]) < pd.Timestamp(mark_ts)
    ):
        res = {
            "status": "MISSING_BAR",
            "requested_ts": pd.Timestamp(mark_ts),
            "actual_ts": pd.NaT,
            "fill_mode": "REJECTED_BEFORE_MARK_TS",
            "fill_offset_min": np.nan,
            "bar": None,
            "contract_rows": res.get("contract_rows", 0),
            "strike_used": res.get("strike_used", safe_int(leg_row["strike"])),
        }

    if res["status"] != "FOUND":
        forced_price = get_forced_option_price(mark_ts, leg_row["expiry_date"])
        if not pd.isna(forced_price):
            return {
                "mark_price": forced_price,
                "mark_exec_ts": pd.Timestamp(mark_ts),
                "mark_fill_mode": f"FORCED_DEFAULT:{res['status']}",
                "mark_fill_offset_min": np.nan,
                "mark_status": "FOUND",
            }
        return {
            "mark_price": np.nan,
            "mark_exec_ts": pd.NaT,
            "mark_fill_mode": "MISSING",
            "mark_fill_offset_min": np.nan,
            "mark_status": res["status"],
        }

    mark_price = extract_bar_price(res, cfg.EXIT_PRICE_FIELD)
    if pd.isna(mark_price):
        forced_price = get_forced_option_price(res["actual_ts"], leg_row["expiry_date"])
        if not pd.isna(forced_price):
            return {
                "mark_price": forced_price,
                "mark_exec_ts": pd.Timestamp(res["actual_ts"]),
                "mark_fill_mode": f"{res['fill_mode']}:FORCED_DEFAULT",
                "mark_fill_offset_min": res["fill_offset_min"],
                "mark_status": "FOUND",
            }
    status = "FOUND" if not pd.isna(mark_price) else f"MISSING_{cfg.EXIT_PRICE_FIELD.upper()}"
    return {
        "mark_price": mark_price,
        "mark_exec_ts": pd.Timestamp(res["actual_ts"]),
        "mark_fill_mode": res["fill_mode"],
        "mark_fill_offset_min": res["fill_offset_min"],
        "mark_status": status,
    }


def build_intratrade_timeline_rows(trade_row, leg_rows, option_source):
    if not leg_rows:
        return []

    filled_exit_ts = [pd.Timestamp(x["exit_exec_ts"]) for x in leg_rows if not pd.isna(x.get("exit_exec_ts", pd.NaT))]
    if not filled_exit_ts:
        return []

    trade_end_ts = max(filled_exit_ts)
    mark_grid = build_intratrade_mark_grid(
        entry_ts=trade_row["entry_ts"],
        end_ts=trade_end_ts,
        timeframe=cfg.INTRATRADE_MARK_TIMEFRAME,
        include_final_exit_row=getattr(cfg, "INTRATRADE_INCLUDE_FINAL_EXIT_ROW", True),
    )
    if not mark_grid:
        return []

    canonical_meta = _canonical_trade_output_fields(trade_row)
    structure = canonical_meta["Structure"]
    leg_map = {x["leg_name"]: x for x in leg_rows}
    ordered_leg_names = _leg_order_for_structure(structure, leg_rows)
    canonical_meta["Ordered_Leg_Names"] = "|".join([name for name in ordered_leg_names if name in leg_map])

    # ── pre-convert per-leg timestamps once, outside the mark-grid loop ──────
    _leg_meta = {}
    for leg_name in ordered_leg_names:
        leg = leg_map.get(leg_name)
        if leg is None:
            _leg_meta[leg_name] = None
            continue
        entry_ts_val = leg.get("entry_exec_ts", pd.NaT)
        exit_ts_val  = leg.get("exit_exec_ts",  pd.NaT)
        _leg_meta[leg_name] = {
            "leg": leg,
            "entry_ts": pd.Timestamp(entry_ts_val) if not pd.isna(entry_ts_val) else pd.NaT,
            "exit_ts":  pd.Timestamp(exit_ts_val)  if not pd.isna(exit_ts_val)  else pd.NaT,
            "exit_price":        safe_float(leg.get("exit_price")),
            "realized_if_closed": safe_float(leg.get("gross_pnl_total")),
        }

    # ── common header values shared across all mark rows ────────────────────
    _common_header = {
        "Trade_ID":        trade_row["trade_id"],
        **canonical_meta,
        "Trade_Day":       trade_row["trade_day"],
        "Signal_Entry_TS": trade_row["entry_ts"],
        "Expiry_Date":     trade_row["expiry_date"],
        "DTE":             trade_row["dte"],
        "Spot_At_Entry":   trade_row["spot_at_entry"],
        "RoundValue":      trade_row["roundvalue"],
        "ShortDteStdMultiplier": trade_row.get("short_dte_std_multiplier"),
        "Short_CE_Strike": trade_row.get("short_ce_strike"),
        "Short_PE_Strike": trade_row.get("short_pe_strike"),
        "Long_CE_Strike":  trade_row.get("outer_long_ce_strike", trade_row.get("long_ce_strike")),
        "Long_PE_Strike":  trade_row.get("outer_long_pe_strike", trade_row.get("long_pe_strike")),
        "InnerLongCEStrike": trade_row.get("inner_long_ce_strike"),
        "OuterLongCEStrike": trade_row.get("outer_long_ce_strike"),
        "InnerLongPEStrike": trade_row.get("inner_long_pe_strike"),
        "OuterLongPEStrike": trade_row.get("outer_long_pe_strike"),
    }

    timeline_rows = []

    for mark_ts in mark_grid:
        row = dict(_common_header)
        row["Mark_TS"] = mark_ts
        row["Open_Legs_Count"] = 0

        trade_realized = 0.0
        trade_unrealized = 0.0
        open_leg_missing_mark = False

        for leg_name in ordered_leg_names:
            meta = _leg_meta[leg_name]
            prefix = leg_name

            if meta is None:
                row[f"{prefix}_State"] = "MISSING_LEG"
                row[f"{prefix}_Mark_TS"] = pd.NaT
                row[f"{prefix}_Mark_Price"] = np.nan
                row[f"{prefix}_Mark_Fill_Mode"] = "MISSING"
                row[f"{prefix}_Mark_Fill_Offset_Min"] = np.nan
                row[f"{prefix}_Realized_PnL"] = np.nan
                row[f"{prefix}_Unrealized_PnL"] = np.nan
                row[f"{prefix}_Total_PnL"] = np.nan
                row[f"{prefix}_Exit_TS"] = pd.NaT
                row[f"{prefix}_Exit_Price"] = np.nan
                continue

            leg        = meta["leg"]
            entry_ts   = meta["entry_ts"]
            exit_ts    = meta["exit_ts"]
            exit_price = meta["exit_price"]
            realized_if_closed = meta["realized_if_closed"]

            row[f"{prefix}_Exit_TS"]    = exit_ts
            row[f"{prefix}_Exit_Price"] = exit_price

            if pd.isna(entry_ts) or mark_ts < entry_ts:
                row[f"{prefix}_State"] = "BEFORE_ENTRY"
                row[f"{prefix}_Mark_TS"] = pd.NaT
                row[f"{prefix}_Mark_Price"] = np.nan
                row[f"{prefix}_Mark_Fill_Mode"] = "NA"
                row[f"{prefix}_Mark_Fill_Offset_Min"] = np.nan
                row[f"{prefix}_Realized_PnL"] = 0.0
                row[f"{prefix}_Unrealized_PnL"] = np.nan
                row[f"{prefix}_Total_PnL"] = np.nan
                continue

            if not pd.isna(exit_ts) and mark_ts >= exit_ts:
                row[f"{prefix}_State"] = "CLOSED"
                row[f"{prefix}_Mark_TS"] = exit_ts
                row[f"{prefix}_Mark_Price"] = exit_price
                row[f"{prefix}_Mark_Fill_Mode"] = "FROZEN_EXIT"
                row[f"{prefix}_Mark_Fill_Offset_Min"] = 0.0
                row[f"{prefix}_Realized_PnL"] = realized_if_closed
                row[f"{prefix}_Unrealized_PnL"] = 0.0
                row[f"{prefix}_Total_PnL"] = realized_if_closed
                if not pd.isna(realized_if_closed):
                    trade_realized += realized_if_closed
                continue

            row["Open_Legs_Count"] += 1
            live_mark = resolve_mark_for_leg(leg, mark_ts, option_source)
            running_unreal = compute_running_leg_pnl(
                side=leg["side"],
                entry_price=leg.get("entry_price"),
                mark_price=live_mark["mark_price"],
                qty=leg.get("qty"),
                lot_size=leg.get("lot_size"),
            )

            row[f"{prefix}_State"] = "OPEN"
            row[f"{prefix}_Mark_TS"] = live_mark["mark_exec_ts"]
            row[f"{prefix}_Mark_Price"] = live_mark["mark_price"]
            row[f"{prefix}_Mark_Fill_Mode"] = live_mark["mark_fill_mode"]
            row[f"{prefix}_Mark_Fill_Offset_Min"] = live_mark["mark_fill_offset_min"]
            row[f"{prefix}_Realized_PnL"] = 0.0
            row[f"{prefix}_Unrealized_PnL"] = running_unreal
            row[f"{prefix}_Total_PnL"] = running_unreal

            if pd.isna(running_unreal):
                open_leg_missing_mark = True
            else:
                trade_unrealized += running_unreal

        row["Trade_Realized_Running_PnL"] = trade_realized
        row["Trade_Unrealized_Running_PnL"] = np.nan if open_leg_missing_mark else trade_unrealized
        row["Trade_Total_Running_PnL"] = np.nan if open_leg_missing_mark else (trade_realized + trade_unrealized)
        row["Trade_Mark_Status"] = "PARTIAL_MISSING_OPEN_LEG_MARK" if open_leg_missing_mark else "OK"

        timeline_rows.append(row)

    return timeline_rows


# ============================================================
# BACKTEST LOOP
# ============================================================
def _simulate_batch(batch_rows, option_source, use_timeline):
    details_rows, pnl_rows, timeline_rows = [], [], []
    for trade_row_dict in batch_rows:
        trade_row = pd.Series(trade_row_dict)
        if use_timeline:
            details, pnl, final_legs = simulate_one_trade_with_legs(trade_row, option_source)
            if details.get("Trade_Status") == "FILLED":
                timeline_rows.extend(build_intratrade_timeline_rows(trade_row, final_legs, option_source))
        else:
            details, pnl = simulate_one_trade(trade_row, option_source)
        details_rows.append(details)
        pnl_rows.append(pnl)
    return details_rows, pnl_rows, timeline_rows


def _build_lookup_slice_for_batch(batch_df, full_lookup):
    needed_keys = build_required_contract_keys(batch_df)
    return {k: full_lookup[k] for k in needed_keys if k in full_lookup}


def run_full_backtest(trade_intents_df, option_source):
    n = len(trade_intents_df)
    use_timeline = getattr(cfg, "SAVE_INTRATRADE_TIMELINE", False)
    num_workers = getattr(cfg, "NUM_WORKERS", 1)
    if num_workers == 0:
        num_workers = os.cpu_count() or 1

    t0 = perf_counter()

    if num_workers == 1:
        details_rows = []
        pnl_rows = []
        timeline_rows = []

        for i, (_, trade_row) in enumerate(trade_intents_df.iterrows(), start=1):
            if use_timeline:
                details, pnl, final_legs = simulate_one_trade_with_legs(trade_row, option_source)
                if details.get("Trade_Status") == "FILLED":
                    timeline_rows.extend(build_intratrade_timeline_rows(trade_row, final_legs, option_source))
            else:
                details, pnl = simulate_one_trade(trade_row, option_source)

            details_rows.append(details)
            pnl_rows.append(pnl)

            if (i % cfg.PRINT_PROGRESS_EVERY == 0) or (i == 1) or (i == n):
                elapsed = perf_counter() - t0
                rate = i / elapsed if elapsed > 0 else np.nan
                eta = (n - i) / rate / 60 if rate and rate > 0 else np.nan
                print(f"[{i:>6}/{n}] elapsed={elapsed:8.2f}s rate={rate:8.2f} trades/s eta={eta:8.2f} min")

        return pd.DataFrame(details_rows), pd.DataFrame(pnl_rows), pd.DataFrame(timeline_rows)

    print(f"[parallel] using {num_workers} workers for {n} trades")

    all_details = []
    all_pnl = []
    all_timeline = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        if option_source["mode"] == "lazy":
            all_rows = [row.to_dict() for _, row in trade_intents_df.iterrows()]
            chunk_size = max(1, (n + num_workers - 1) // num_workers)
            chunks = [all_rows[i:i + chunk_size] for i in range(0, n, chunk_size)]
            futures = {
                executor.submit(_simulate_batch, chunk, option_source, use_timeline): idx
                for idx, chunk in enumerate(chunks)
            }
        else:
            grouped = []
            for idx, (trade_day, batch_df) in enumerate(trade_intents_df.groupby("trade_day", sort=True), start=0):
                batch_df = batch_df.reset_index(drop=True)
                batch_rows = batch_df.to_dict("records")
                batch_lookup = _build_lookup_slice_for_batch(batch_df, option_source["lookup"])
                batch_option_source = {"mode": option_source["mode"], "lookup": batch_lookup}
                grouped.append((idx, batch_rows, batch_option_source))

            futures = {
                executor.submit(_simulate_batch, batch_rows, batch_option_source, use_timeline): idx
                for idx, batch_rows, batch_option_source in grouped
            }

        completed = 0
        for future in as_completed(futures):
            chunk_idx = futures[future]
            det_rows, pnl_rows, tl_rows = future.result()
            all_details.append((chunk_idx, det_rows))
            all_pnl.append((chunk_idx, pnl_rows))
            all_timeline.extend(tl_rows)

            completed += len(det_rows)
            elapsed = perf_counter() - t0
            rate = completed / elapsed if elapsed > 0 else np.nan
            print(f"[parallel] {completed}/{n} trades done, elapsed={elapsed:.1f}s, rate={rate:.1f} trades/s")

    all_details.sort(key=lambda x: x[0])
    all_pnl.sort(key=lambda x: x[0])

    details_rows = [r for _, rows in all_details for r in rows]
    pnl_rows = [r for _, rows in all_pnl for r in rows]

    return pd.DataFrame(details_rows), pd.DataFrame(pnl_rows), pd.DataFrame(all_timeline)


# ============================================================
# ORCHESTRATION
# ============================================================
def validate_trade_intents_schema(trade_intents_df):
    required_cols = [
        "trade_id", "structure", "variant_tag", "short_dte_std_multiplier",
        "trade_day", "entry_ts", "expiry_date", "dte", "spot_at_entry",
        "dte_std", "one_std", "roundvalue",
        "short_ce_strike", "short_pe_strike",
        "outer_long_ce_strike", "outer_long_pe_strike",
        "inner_long_ce_strike", "inner_long_pe_strike",
        "atm_ce_boundary", "atm_pe_boundary", "intent_status",
    ]
    missing = [c for c in required_cols if c not in trade_intents_df.columns]
    if missing:
        raise ValueError(f"trade_intents_df missing required columns: {missing}")
    structures = set(trade_intents_df["structure"].dropna().astype(str).str.upper())
    unknown = structures - {"IRON_CONDOR", "BATMAN"}
    if unknown:
        raise ValueError(f"Unsupported strategy structure(s) in trade intents: {sorted(unknown)}")
    return True


def build_option_source(trade_intents_df):
    dataset = open_options_dataset(cfg.OPTIONS_DATA_FOLDER)
    if cfg.OPTION_ACCESS_MODE == "lazy":
        return {"mode": "lazy", "dataset": dataset, "cache": ContractCache(cfg.CONTRACT_CACHE_MAX_ITEMS)}
    if cfg.OPTION_ACCESS_MODE == "preload_required_contracts":
        lookup = preload_required_contracts(dataset, trade_intents_df)
        return {"mode": "preload_required_contracts", "lookup": lookup}
    raise ValueError(f"Unsupported OPTION_ACCESS_MODE: {cfg.OPTION_ACCESS_MODE}")


def _coalesce_prefer_left(df, left_col, right_col, out_col):
    has_left = left_col in df.columns
    has_right = right_col in df.columns

    if not has_left and not has_right:
        return df

    if out_col in df.columns:
        if has_left:
            df[out_col] = df[out_col].combine_first(df[left_col])
        if has_right:
            df[out_col] = df[out_col].combine_first(df[right_col])
    else:
        if has_left and has_right:
            df[out_col] = df[left_col].combine_first(df[right_col])
        elif has_left:
            df[out_col] = df[left_col]
        else:
            df[out_col] = df[right_col]

    drop_cols = [c for c in [left_col, right_col] if c in df.columns and c != out_col]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def simplify_trade_details_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Slim the full trade-details DataFrame down to only the columns needed
    in the final output CSV:

        Trade_ID, VariantTag,
        Entry_Date, Entry_Time, DTE, Spot_At_Entry, Regime,
        6 strike cols  (InnerLong* are NaN for Iron Condor),
        6 entry-price cols,
        CE_SelectionMode, PE_SelectionMode  (NaN / "NA" for Iron Condor),
        Net_Credit,
        Exit_Date, Exit_Time,
        6 exit-price cols,
        Net_PnL,
        <FEATURE_COLUMNS_TO_PRINT>
    """
    if df is None or df.empty:
        return df if df is not None else pd.DataFrame()

    out = df.copy()

    # ── 1. Entry date / time ──────────────────────────────────────────────
    if "Signal_Entry_TS" in out.columns:
        _ets = pd.to_datetime(out["Signal_Entry_TS"], errors="coerce")
        out["Entry_Date"] = _ets.dt.strftime("%Y-%m-%d")
        out["Entry_Time"] = _ets.dt.strftime("%H:%M")
    else:
        out["Entry_Date"] = out.get("Trade_Day", np.nan)
        out["Entry_Time"] = np.nan

    # ── 2. Regime (already written as "Regime" in build_trade_rows) ───────
    if "Regime" not in out.columns:
        out["Regime"] = np.nan
    if "Structure" not in out.columns:
        out["Structure"] = np.nan
    out["RegimeLabel"] = out.apply(
        lambda r: (
            f"R{int(float(r['Regime']))}_{r['Structure']}"
            if not pd.isna(r["Regime"])
            else f"RUnknown_{r['Structure']}"
        ),
        axis=1,
    )
    if getattr(cfg, "REGIME_DRIVEN_ENTRY", False):
        out["VariantTag"] = out["RegimeLabel"]

    # ── 3. 6 unified strike cols ──────────────────────────────────────────
    _strike_map = [
        ("InnerLongCE_Strike", ["InnerLongCE_Strike", "inner_long_ce_strike"]),
        ("ShortCE_Strike",     ["ShortCE_Strike",     "Short_CE_Strike",
                                "ShortCEStrike",      "short_ce_strike"]),
        ("OuterLongCE_Strike", ["OuterLongCE_Strike", "outer_long_ce_strike",
                                "LongCE_Strike",      "Long_CE_Strike",
                                "LongCEStrike",       "long_ce_strike"]),
        ("InnerLongPE_Strike", ["InnerLongPE_Strike", "inner_long_pe_strike"]),
        ("ShortPE_Strike",     ["ShortPE_Strike",     "Short_PE_Strike",
                                "ShortPEStrike",      "short_pe_strike"]),
        ("OuterLongPE_Strike", ["OuterLongPE_Strike", "outer_long_pe_strike",
                                "LongPE_Strike",      "Long_PE_Strike",
                                "LongPEStrike",       "long_pe_strike"]),
    ]
    for target, sources in _strike_map:
        if target not in out.columns:
            for src in sources:
                if src in out.columns:
                    out[target] = out[src]
                    break
            else:
                out[target] = np.nan

    # ── 4. 6 unified entry-price cols ─────────────────────────────────────
    # leg-prefix form:  "INNERLONGCE" / "SHORTCE" / "OUTERLONGCE" (Batman)
    #                   "SHORTCE"     / "SHORTPE"  / "LONGCE"      (IC)
    _ep_map = [
        ("InnerLongCE_EntryPrice", ["INNERLONGCEEntryPrice"]),
        ("ShortCE_EntryPrice",     ["SHORTCEEntryPrice"]),
        ("OuterLongCE_EntryPrice", ["OUTERLONGCEEntryPrice", "LONGCEEntryPrice"]),
        ("InnerLongPE_EntryPrice", ["INNERLONGPEEntryPrice"]),
        ("ShortPE_EntryPrice",     ["SHORTPEEntryPrice"]),
        ("OuterLongPE_EntryPrice", ["OUTERLONGPEEntryPrice", "LONGPEEntryPrice"]),
    ]
    # Coalesce: merge all matching source columns so IC (LONGCEEntryPrice) and
    # Batman (OUTERLONGCEEntryPrice) rows both get populated even when both
    # column names exist in the combined DataFrame.
    for target, sources in _ep_map:
        merged = None
        for src in sources:
            if src in out.columns:
                if merged is None:
                    merged = out[src].copy()
                else:
                    merged = merged.where(merged.notna(), out[src])
        out[target] = merged if merged is not None else np.nan

    # ── 5. CE / PE selection mode ─────────────────────────────────────────
    if "CE_SelectionMode" not in out.columns:
        out["CE_SelectionMode"] = out.get("CESelectionMode", np.nan)
    if "PE_SelectionMode" not in out.columns:
        out["PE_SelectionMode"] = out.get("PESelectionMode", np.nan)

    # ── 6. Net credit ─────────────────────────────────────────────────────
    if "TotalEntryValue" in out.columns:
        out["Net_Credit"] = out["TotalEntryValue"]
    elif "Total_Entry_Value" in out.columns:
        out["Net_Credit"] = out["Total_Entry_Value"]
    else:
        out["Net_Credit"] = np.nan

    # ── 7. Exit date / time  (use SHORTCE exit as reference bar) ─────────
    _exit_ts_candidates = [
        "SHORTCEExitTS", "SHORTPEExitTS",
        "OUTERLONGCEExitTS", "LONGCEExitTS",
    ]
    _exit_ts_col = next((c for c in _exit_ts_candidates if c in out.columns), None)
    if _exit_ts_col:
        _xts = pd.to_datetime(out[_exit_ts_col], errors="coerce")
    else:
        _xts = pd.Series(pd.NaT, index=out.index)
    out["Exit_Date"] = _xts.dt.strftime("%Y-%m-%d")
    out["Exit_Time"] = _xts.dt.strftime("%H:%M")

    # ── 7b. Per-side exit date / time ─────────────────────────────────────
    _ce_ets_col = "SHORTCEExitTS" if "SHORTCEExitTS" in out.columns else None
    _ce_ets = pd.to_datetime(out[_ce_ets_col], errors="coerce") if _ce_ets_col else pd.Series(pd.NaT, index=out.index)
    out["CE_Exit_Date"] = _ce_ets.dt.strftime("%Y-%m-%d")
    out["CE_Exit_Time"] = _ce_ets.dt.strftime("%H:%M")

    _pe_ets_col = "SHORTPEExitTS" if "SHORTPEExitTS" in out.columns else None
    _pe_ets = pd.to_datetime(out[_pe_ets_col], errors="coerce") if _pe_ets_col else pd.Series(pd.NaT, index=out.index)
    out["PE_Exit_Date"] = _pe_ets.dt.strftime("%Y-%m-%d")
    out["PE_Exit_Time"] = _pe_ets.dt.strftime("%H:%M")

    # ── 8. 6 unified exit-price cols ──────────────────────────────────────
    _xp_map = [
        ("InnerLongCE_ExitPrice", ["INNERLONGCEExitPrice"]),
        ("ShortCE_ExitPrice",     ["SHORTCEExitPrice"]),
        ("OuterLongCE_ExitPrice", ["OUTERLONGCEExitPrice", "LONGCEExitPrice"]),
        ("InnerLongPE_ExitPrice", ["INNERLONGPEExitPrice"]),
        ("ShortPE_ExitPrice",     ["SHORTPEExitPrice"]),
        ("OuterLongPE_ExitPrice", ["OUTERLONGPEExitPrice", "LONGPEExitPrice"]),
    ]
    # Same coalesce logic as _ep_map: IC uses LONGCEExitPrice, Batman uses OUTERLONGCEExitPrice.
    for target, sources in _xp_map:
        merged = None
        for src in sources:
            if src in out.columns:
                if merged is None:
                    merged = out[src].copy()
                else:
                    merged = merged.where(merged.notna(), out[src])
        out[target] = merged if merged is not None else np.nan

    # ── 9. Net PnL ────────────────────────────────────────────────────────
    if "TotalPnL" in out.columns:
        out["Net_PnL"] = out["TotalPnL"]
    elif "Total_PnL" in out.columns:
        out["Net_PnL"] = out["Total_PnL"]
    else:
        out["Net_PnL"] = np.nan

    # ── 10. Feature columns from config ──────────────────────────────────
    if "rv_slow" not in out.columns and "rvslow" in out.columns:
        out["rv_slow"] = out["rvslow"]
    _feat_cols = [
        c for c in getattr(cfg, "FEATURE_COLUMNS_TO_PRINT", [])
        if c in out.columns
    ]

    # ── 10b. CPT diagnostic columns ───────────────────────────────────────
    # CPT_Target: same value as TGT_Amount but explicit for clarity.
    out["CPT_Target"] = out["TGT_Amount"] if "TGT_Amount" in out.columns else np.nan

    # CPT_Hit: True when both legs exited via combined-target (intraday or at expiry).
    _ce_reason = out["CE_Exit_Reason"] if "CE_Exit_Reason" in out.columns else pd.Series("", index=out.index)
    _pe_reason = out["PE_Exit_Reason"] if "PE_Exit_Reason" in out.columns else pd.Series("", index=out.index)
    out["CPT_Hit"] = (
        _ce_reason.astype(str).str.contains("COMBINED_TP", na=False) &
        _pe_reason.astype(str).str.contains("COMBINED_TP", na=False)
    )

    # Final_Net_PnL_GTE_Target: whether the final Net_PnL exceeded the CPT threshold.
    if "Net_PnL" in out.columns and "CPT_Target" in out.columns:
        _pnl = pd.to_numeric(out["Net_PnL"], errors="coerce")
        _tgt = pd.to_numeric(out["CPT_Target"], errors="coerce")
        out["Final_Net_PnL_GTE_Target"] = (_pnl >= _tgt) & _pnl.notna() & _tgt.notna()
    else:
        out["Final_Net_PnL_GTE_Target"] = np.nan

    # ── 11. Final slim column list ────────────────────────────────────────
    _slim = [
        "Trade_ID",
        "VariantTag",
        "Entry_Date",
        "Entry_Time",
        "DTE",
        "Expiry_Date",
        "Spot_At_Entry",
        "Regime",
        "RegimeLabel",
        # strikes (6)
        "InnerLongCE_Strike",
        "ShortCE_Strike",
        "OuterLongCE_Strike",
        "InnerLongPE_Strike",
        "ShortPE_Strike",
        "OuterLongPE_Strike",
        # entry prices (6)
        "InnerLongCE_EntryPrice",
        "ShortCE_EntryPrice",
        "OuterLongCE_EntryPrice",
        "InnerLongPE_EntryPrice",
        "ShortPE_EntryPrice",
        "OuterLongPE_EntryPrice",
        # Batman inner-long selection diagnostics (NaN/"NA" for IC)
        "CE_SelectionMode",
        "PE_SelectionMode",
        # net credit
        "Net_Credit",
        # exit
        "Exit_Date",
        "Exit_Time",
        "CE_Exit_Date",
        "CE_Exit_Time",
        "PE_Exit_Date",
        "PE_Exit_Time",
        # exit prices (6)
        "InnerLongCE_ExitPrice",
        "ShortCE_ExitPrice",
        "OuterLongCE_ExitPrice",
        "InnerLongPE_ExitPrice",
        "ShortPE_ExitPrice",
        "OuterLongPE_ExitPrice",
        # exit reasons and regime parameters
        "CE_Exit_Reason",
        "PE_Exit_Reason",
        "SL_Params",
        "TGT_Params",
        "TGT_Amount",
        "SecondLegSide",
        "SecondLegEntryTS",
        "SecondLegEntryCredit",
        "SecondLegExitTS",
        "SecondLegExitReason",
        "SecondLegPnL",
        # CPT diagnostics
        "CPT_Target",
        "CPT_Hit",
        "Final_Net_PnL_GTE_Target",
        # PnL
        "Net_PnL",
    ] + _feat_cols

    # Ensure every listed column exists (fill NaN if not produced above)
    for _c in _slim:
        if _c not in out.columns:
            out[_c] = np.nan

    # Deduplicate while preserving order
    _seen: set = set()
    _final: list = []
    for _c in _slim:
        if _c not in _seen:
            _seen.add(_c)
            _final.append(_c)

    return out[_final].copy()


def _save_variant_outputs(trade_details_df, trade_timeline_df, variant_tag, base_tag):
    """Save one variant's files immediately after its backtest completes."""
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    if not cfg.SAVE_CSV:
        return

    v = sanitize_variant_tag(variant_tag)

    # --- trade details ---
    if trade_details_df is not None and not trade_details_df.empty:
        simplified = simplify_trade_details_output(trade_details_df)
        details_csv = os.path.join(cfg.OUTPUT_DIR, f"{base_tag}_{v}_trade_details.csv")
        simplified.to_csv(details_csv, index=False)
        print(f"[checkpoint] saved -> {details_csv}")

    # --- timeline ---
    if (
        getattr(cfg, "SAVE_INTRATRADE_TIMELINE", False)
        and getattr(cfg, "SAVE_INTRATRADE_TIMELINE_CSV", False)
    ):
        timeline_csv = os.path.join(cfg.OUTPUT_DIR, f"{base_tag}_{v}_trade_timeline.csv")
        if trade_timeline_df is not None and not trade_timeline_df.empty:
            trade_timeline_df.to_csv(timeline_csv, index=False)
        else:
            pd.DataFrame().to_csv(timeline_csv, index=False)
        print(f"[checkpoint] saved -> {timeline_csv}")


def save_outputs(tradedetailsdf, tradepnldf, tradetimelinedf=None):
    """
    Save trade details and optional timeline.
    tradepnl is intentionally not saved.
    """
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    outputfiletag = getattr(
        cfg,
        "OUTPUT_FILE_TAG",
        (
            f"{cfg.STRATEGY_NAME}_"
            f"{getattr(cfg, 'ENTRYMODE', getattr(cfg, 'ENTRY_MODE', 'entry'))}_"
            f"{getattr(cfg, 'STARTDATE', getattr(cfg, 'START_DATE', 'start'))}_to_"
            f"{getattr(cfg, 'ENDDATE', getattr(cfg, 'END_DATE', 'end'))}"
        ),
    )
    if getattr(cfg, "REGIME_DRIVEN_ENTRY", False) and not hasattr(cfg, "OUTPUT_FILE_TAG"):
        outputfiletag = (
            f"{cfg.STRATEGY_NAME}_REGIME_DRIVEN_"
            f"{getattr(cfg, 'STARTDATE', getattr(cfg, 'START_DATE', 'start'))}_to_"
            f"{getattr(cfg, 'ENDDATE', getattr(cfg, 'END_DATE', 'end'))}"
        )
    basetag = f"{outputfiletag}"

    if not cfg.SAVE_CSV:
        return

    if tradetimelinedf is None:
        tradetimelinedf = pd.DataFrame()

    # Clean/simplify only the final trade-details output
    simplified_all = simplify_trade_details_output(tradedetailsdf)

    if (
        getattr(cfg, "SAVE_OUTPUTS_BY_VARIANT", False)
        and not getattr(cfg, "REGIME_DRIVEN_ENTRY", False)
        and simplified_all is not None
        and not simplified_all.empty
        and "VariantTag" in simplified_all.columns
    ):
        print("Saved")
        for varianttag, detailspart in simplified_all.groupby("VariantTag", dropna=False):
            variant = sanitize_variant_tag(varianttag)

            detailscsv = os.path.join(
                cfg.OUTPUT_DIR,
                f"{basetag}_{variant}_trade_details.csv"
            )
            detailspart.to_csv(detailscsv, index=False)
            print(detailscsv)

            if (
                getattr(cfg, "SAVE_INTRATRADE_TIMELINE", False)
                and getattr(cfg, "SAVE_INTRATRADE_TIMELINE_CSV", False)
            ):
                timelinecsv = os.path.join(
                    cfg.OUTPUT_DIR,
                    f"{basetag}_{variant}_trade_timeline.csv"
                )
                if (
                    tradetimelinedf is not None
                    and not tradetimelinedf.empty
                    and "VariantTag" in tradetimelinedf.columns
                ):
                    tradetimelinedf[
                        tradetimelinedf["VariantTag"] == varianttag
                    ].to_csv(timelinecsv, index=False)
                else:
                    pd.DataFrame().to_csv(timelinecsv, index=False)
                print(timelinecsv)
        return

    detailscsv = os.path.join(cfg.OUTPUT_DIR, f"{basetag}_trade_details.csv")
    simplified_all.to_csv(detailscsv, index=False)
    print("Saved")
    print(detailscsv)

    if (
        getattr(cfg, "SAVE_INTRATRADE_TIMELINE", False)
        and getattr(cfg, "SAVE_INTRATRADE_TIMELINE_CSV", False)
    ):
        timelinecsv = os.path.join(cfg.OUTPUT_DIR, f"{basetag}_trade_timeline.csv")
        tradetimelinedf.to_csv(timelinecsv, index=False)
        print(timelinecsv)


# ============================================================
# OUTPUT ENRICHMENT (FEATURES IN TRADE DETAILS)
# ============================================================

def load_features_for_output(feature_sheet_path):
    """
    Output enrichment requires:
    - from features_calculations : one_std, dte_std
    - from features_standardised : gap, orb_1_std, ivp_12m, skew,
                                   rvs_minus_iv, rv_slow_over_rv_fast_1_3
    Merge both on (trade_day, feature_ts, feature_time).
    """
    xls = pd.ExcelFile(feature_sheet_path)
    _lower = {s.strip().lower(): s for s in xls.sheet_names}
    calc_raw = pd.read_excel(feature_sheet_path, sheet_name=_lower.get("features_calculations", xls.sheet_names[0]))
    std_raw = pd.read_excel(feature_sheet_path, sheet_name=_lower.get("features_standardised", xls.sheet_names[0]))

    calc_df = normalize_column_names(calc_raw)
    calc_df = standardize_feature_columns(calc_df)
    std_df = normalize_column_names(std_raw)
    std_df = standardize_feature_columns(std_df)

    def _prep(df):
        if "trade_day" not in df.columns or "feature_ts" not in df.columns:
            raise ValueError(f"Feature sheet missing trade_day/feature_ts after normalization. Columns: {list(df.columns)}")
        df["trade_day"] = pd.to_datetime(df["trade_day"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["feature_ts"] = pd.to_datetime(df["feature_ts"], errors="coerce")
        if "feature_time" not in df.columns:
            df["feature_time"] = df["feature_ts"].dt.strftime("%H:%M")
        else:
            df["feature_time"] = df["feature_time"].astype(str).str.strip().str.slice(0, 5)
        return df

    calc_df = _prep(calc_df)
    std_df = _prep(std_df)

    calc_keep = [c for c in [
        "trade_day", "feature_ts", "feature_time",
        "one_std", "dte_std", "move_to_open_1_std", "rv_slow", "iv",
    ] if c in calc_df.columns]
    std_keep = [c for c in [
        "trade_day", "feature_ts", "feature_time",
        "gap", "orb_1_std", "ivp_12m", "skew",
        "rvs_minus_iv", "rv_slow_over_rv_fast_1_3",
        "move_to_open_1_std", "rv_slow", "iv",
    ] if c in std_df.columns]

    calc_out = calc_df[calc_keep].drop_duplicates(subset=["trade_day", "feature_ts"], keep="first").copy()
    std_out = std_df[std_keep].drop_duplicates(subset=["trade_day", "feature_ts"], keep="first").copy()

    out = calc_out.merge(
        std_out,
        on=["trade_day", "feature_ts", "feature_time"],
        how="outer"
    )
    for col in ["move_to_open_1_std", "rv_slow", "iv"]:
        out = _coalesce_prefer_left(out, f"{col}_x", f"{col}_y", col)

    numeric_cols = [
        "one_std", "dte_std",
        "gap", "orb_1_std", "ivp_12m", "skew",
        "rvs_minus_iv", "rv_slow_over_rv_fast_1_3",
        "move_to_open_1_std", "rv_slow", "iv",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.sort_values(["trade_day", "feature_ts"]).reset_index(drop=True)
    return out

def enrich_trade_details_with_features(trade_details_df, feature_df, feature_cols_to_print):
    if trade_details_df.empty:
        return trade_details_df.copy()

    feat = feature_df
    feature_cols_available = [c for c in feature_cols_to_print if c in feat.columns]
    if not feature_cols_available:
        return trade_details_df.copy()

    feature_join_df = feat[["trade_day", "feature_ts"] + feature_cols_available].copy()
    feature_join_df = feature_join_df.drop_duplicates(subset=["trade_day", "feature_ts"], keep="first")

    out = trade_details_df.copy()
    if "Trade_Day" not in out.columns:
        raise ValueError("trade_details_df missing required column: Trade_Day")
    if "Signal_Entry_TS" not in out.columns:
        raise ValueError("trade_details_df missing required column: Signal_Entry_TS")

    out["Trade_Day"] = pd.to_datetime(out["Trade_Day"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["Signal_Entry_TS"] = pd.to_datetime(out["Signal_Entry_TS"], errors="coerce")

    # Drop any feature columns already present (NaN stubs from build_trade_rows) so the
    # merge does not create _x/_y duplicates that prevent simplify_trade_details_output
    # from finding columns by their original names in FEATURE_COLUMNS_TO_PRINT.
    _pre_existing = [c for c in feature_cols_available if c in out.columns]
    if _pre_existing:
        out = out.drop(columns=_pre_existing)

    out = out.merge(
        feature_join_df,
        left_on=["Trade_Day", "Signal_Entry_TS"],
        right_on=["trade_day", "feature_ts"],
        how="left",
    )

    drop_cols = [c for c in ["trade_day", "feature_ts"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    return out


# ============================================================
# MAIN
# ============================================================
def main():
    _t_start = perf_counter()
    print("CONFIG")
    print(f"ENTRY_MODE         : {cfg.ENTRY_MODE}")
    print(f"OPTION_ACCESS_MODE : {cfg.OPTION_ACCESS_MODE}")
    print(f"DATE_RANGE         : {cfg.START_DATE} -> {cfg.END_DATE}")
    print(f"SAVE_TIMELINE      : {getattr(cfg, 'SAVE_INTRATRADE_TIMELINE', False)}")

    index_15m = build_index_signal_bars(
        index_path=cfg.INDEX_DATA_PATH,
        timeframe=cfg.SIGNAL_TIMEFRAME,
        manual_entry_times=cfg.MANUAL_ENTRY_TIMES,
        restrict_to_idx=True,
    )
    _t1 = perf_counter(); print(f"[time] index_signal_bars       : {_t1 - _t_start:7.2f}s")

    expiry_raw = load_table_auto(cfg.EXPIRY_MAP_PATH)
    expiry_df = prepare_expiry_map(expiry_raw)
    _t2 = perf_counter(); print(f"[time] expiry_map               : {_t2 - _t1:7.2f}s")

    signal_panel = build_signal_panel(index_15m, expiry_df)
    _t3 = perf_counter(); print(f"[time] build_signal_panel       : {_t3 - _t2:7.2f}s")
    signal_panel_selected = apply_entry_mode(signal_panel, cfg.ENTRY_MODE, cfg.FEATURE_SHEET_PATH)
    _t4 = perf_counter(); print(f"[time] apply_entry_mode         : {_t4 - _t3:7.2f}s")

    # Patch 3a: time-window deduplication (no-op when ENTRY_WINDOW_MINUTES=None)
    signal_panel_selected = apply_time_window_dedup(signal_panel_selected)
    _t4b = perf_counter(); print(f"[time] time_window_dedup        : {_t4b - _t4:7.2f}s  rows={len(signal_panel_selected)}")

    # Patch 3b: attach regime params (no-op when REGIME_DRIVEN_ENTRY=False)
    signal_panel_selected = attach_regime_params(signal_panel_selected, cfg.FEATURE_SHEET_PATH)
    _t4c = perf_counter(); print(f"[time] attach_regime_params     : {_t4c - _t4b:7.2f}s")
    if "rc_regime" in signal_panel_selected.columns:
        print("regime distribution:\n" + signal_panel_selected["rc_regime"].value_counts(dropna=False).to_string())

        if getattr(cfg, "REGIME_DRIVEN_ENTRY", False):
            print("regime-driven structure distribution:")
            if "rc_structure" in signal_panel_selected.columns:
                print(signal_panel_selected["rc_structure"].value_counts(dropna=False).to_string())

            print("regime-driven multiplier distribution:")
            if "rc_multiplier" in signal_panel_selected.columns:
                print(signal_panel_selected["rc_multiplier"].value_counts(dropna=False).to_string())

    base_trade_intents = build_trade_intents_base(signal_panel_selected)
    validate_trade_intents_schema(base_trade_intents)
    _t5 = perf_counter(); print(f"[time] build_trade_intents_base : {_t5 - _t4c:7.2f}s")
    option_source = build_option_source(base_trade_intents)
    _t6 = perf_counter(); print(f"[time] build_option_source      : {_t6 - _t5:7.2f}s")
    trade_intents = finalize_trade_intents_with_batman_selection(base_trade_intents, option_source)
    validate_trade_intents_schema(trade_intents)
    _t7 = perf_counter(); print(f"[time] finalize_trade_intents   : {_t7 - _t6:7.2f}s")

    print(f"signal_panel rows   : {len(signal_panel)}")
    print(f"selected signal rows: {len(signal_panel_selected)}")
    print(f"trade_intents rows  : {len(trade_intents)}")

    trade_details_df, trade_pnl_df, trade_timeline_df = run_full_backtest(trade_intents, option_source)

    # -- per-variant checkpoint saves (no logic change, just early flush) --
    output_file_tag = getattr(
        cfg,
        "OUTPUT_FILE_TAG",
        (
            f"{cfg.STRATEGY_NAME}_{getattr(cfg, 'ENTRY_MODE', getattr(cfg, 'ENTRY_MODE', 'entry'))}"
            f"_{getattr(cfg, 'START_DATE', getattr(cfg, 'START_DATE', 'start'))}"
            f"_to_{getattr(cfg, 'END_DATE', getattr(cfg, 'END_DATE', 'end'))}"
        ),
    )
    if getattr(cfg, "REGIME_DRIVEN_ENTRY", False) and not hasattr(cfg, "OUTPUT_FILE_TAG"):
        output_file_tag = (
            f"{cfg.STRATEGY_NAME}_REGIME_DRIVEN_"
            f"{getattr(cfg, 'START_DATE', getattr(cfg, 'START_DATE', 'start'))}"
            f"_to_{getattr(cfg, 'END_DATE', getattr(cfg, 'END_DATE', 'end'))}"
        )
    base_tag = f"{output_file_tag}"
    if (
        getattr(cfg, "SAVE_OUTPUTS_BY_VARIANT", False)
        and not getattr(cfg, "REGIME_DRIVEN_ENTRY", False)
        and trade_details_df is not None
        and not trade_details_df.empty
        and "VariantTag" in trade_details_df.columns
    ):
        for _variant_tag, _variant_df in trade_details_df.groupby("VariantTag", dropna=False):
            _tl = None
            if (
                trade_timeline_df is not None
                and not trade_timeline_df.empty
                and "VariantTag" in trade_timeline_df.columns
            ):
                _tl = trade_timeline_df[trade_timeline_df["VariantTag"] == _variant_tag]
            _save_variant_outputs(_variant_df, _tl, _variant_tag, base_tag)

    _t8 = perf_counter(); print(f"[time] run_full_backtest        : {_t8 - _t7:7.2f}s")

    if getattr(cfg, "PRINT_FEATURES_IN_TRADE_DETAILS", False):
        _feat_for_output = load_features_for_output(cfg.FEATURE_SHEET_PATH)
        trade_details_df = enrich_trade_details_with_features(
            trade_details_df=trade_details_df,
            feature_df=_feat_for_output,
            feature_cols_to_print=getattr(cfg, "FEATURE_COLUMNS_TO_PRINT", []),
        )
        _t9 = perf_counter(); print(f"[time] enrich_with_features           : {_t9 - _t8:7.2f}s")
    else:
        _t9 = _t8

    if cfg.SHOW_PREVIEWS:
        print("\nTRADE DETAILS PREVIEW")
        print(trade_details_df.head(10))
        print("\nSTATUS COUNTS")
        print(trade_details_df["Trade_Status"].value_counts(dropna=False))

        if getattr(cfg, "SAVE_INTRATRADE_TIMELINE", False):
            print("\nTRADE TIMELINE PREVIEW")
            print(trade_timeline_df.head(10))

    save_outputs(trade_details_df, None, trade_timeline_df)
    _t_end = perf_counter()
    print(f"[time] save_outputs             : {_t_end - _t9:7.2f}s")
    if getattr(cfg, "BENCHMARK_MODE", False):
        print("\n" + "=" * 55)
        print("  BENCHMARK SUMMARY")
        print("=" * 55)
        total_runtime = _t_end - _t_start
        stages = [
            ("index_signal_bars", _t1 - _t_start),
            ("expiry_map", _t2 - _t1),
            ("build_signal_panel", _t3 - _t2),
            ("apply_entry_mode", _t4 - _t3),
            ("build_trade_intents_base", _t5 - _t4),
            ("build_option_source", _t6 - _t5),
            ("finalize_trade_intents", _t7 - _t6),
            ("run_full_backtest", _t8 - _t7),
            ("enrich_with_features", _t9 - _t8),
            ("save_outputs", _t_end - _t9),
        ]
        for name, t in stages:
            bar = "█" * max(1, int((t / total_runtime) * 40)) if total_runtime > 0 else ""
            print(f"  {name:<28} {t:7.2f}s  {bar}")
        print("=" * 55)
    print("\n" + "="*55)
    print(f"  TOTAL WALL TIME : {_t_end - _t_start:.2f}s  ({(_t_end - _t_start)/60:.1f} min)")
    print("="*55)


if __name__ == "__main__":
    main()
