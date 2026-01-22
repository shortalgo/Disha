#!/usr/bin/env python3
"""
Walk-forward selector PER BUCKET + lot assignment across TOP-3 FINAL strategies per bucket.

FIX (ONLY): weeks_in_window must come from the expiry calendar (ExpiryDate list),
so it is identical for all strategies inside a fold window.

THIS BASELINE:
✅ Applies EXHAUSTIVE date parsing everywhere (expiry calendar + all trade-date parsing + any internal date masks).

PATCHES:
✅ Scoring system:
   - adjusted median pnl: 30%
   - win rate: 20%
   - cVaR: 30%
   - max dd: 20%
✅ Compute win rate per window.
✅ ADD FULL FILTERING LOGIC per bucket BEFORE walk-forward:
   On entire tradesheet dataset (full backtest window), compute:
     - Max Drawdown % (equity curve on CAPITAL)
     - Trade coverage = weeks_traded / total_expiries_period
     - Total returns % = total_pnl / CAPITAL
     - Win rate = wins / weeks_traded
   Remove strategies if:
     1) mdd_pct > 33%
     2) trade_coverage < 33%
     3) total_return_pct < 36%
     4) win_rate < 50%
✅ ONLY CHANGE: top_k from 100 -> 10
❌ Keep everything else unchanged (selection flow, folds, lots, exports, meta-selection, etc).
"""
#adjusted for nifty vix

import os
import glob
import math
import time
import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

# Excel export
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


# ============================================================
# CONFIG
# ============================================================
TOTAL_LOTS = 80
TOP_N_FINAL_PER_BUCKET = 3

BUCKET_DIRS: Dict[str, str] = {
    "B1":  r"D:\Downloads\Updated PNLZIP\PNL\DTE-1.0",
    "B2":  r"D:\Downloads\Updated PNLZIP\PNL\DTE-1.5",
    "B3":  r"D:\Downloads\Updated PNLZIP\PNL\DTE-2.0",
    "B4":  r"D:\Downloads\Updated PNLZIP\PNL\DTE-2.5",
    "B5":  r"D:\Downloads\Updated PNLZIP\PNL\DTE-3.0",
    "B6":  r"D:\Downloads\Updated PNLZIP\PNL\DTE-3.5",
    "B7":  r"D:\Downloads\Updated PNLZIP\PNL\DTE-4.0",
    "B8":  r"D:\Downloads\Updated PNLZIP\PNL\DTE-4.5",
}

# >>> use the UNIQUE expiry list you uploaded
EXPIRY_CALENDAR_CSV = r"D:\Downloads\unique_nifty_expiry_dates.csv"

OUTPUT_XLSX: Optional[str] = None

# Fixed backtest period (day-first)
BACKTEST_START = pd.to_datetime("01-01-2022", dayfirst=True).normalize()
BACKTEST_END   = pd.to_datetime("22-12-2025", dayfirst=True).normalize()

PRINT_EVERY_FILES = 200
PRINT_EVERY_FOLDS = 1
PRINT_TIMINGS = True
SHOW_EXCEPTIONS_SAMPLE = 3

# ============================================================
# PREFILTER (NEW) - Dataset-level removal thresholds
# ============================================================
CAPITAL = 87500.0

PREF_MAX_DD_PCT = 0.33          # remove if > 33% DD
PREF_MIN_TRADE_COVERAGE = 0.33  # remove if < 33% traded expiries
PREF_MIN_TOTAL_RETURN_PCT = 0.36# remove if < 36% total return
PREF_MIN_WIN_RATE = 0.50        # remove if < 50% win rate


@dataclass(frozen=True)
class WalkForwardConfig:
    # Walk-forward windowing (calendar months)
    is_months: int = 12
    oos_months: int = 3
    step_months: int = 3
    top_k: int = 100  

    # Score weights (percentiles)
    w_pnl: float = 0.30
    w_win: float = 0.20
    w_risk: float = 0.30
    w_mdd: float = 0.20

    # Final recency weights (calendar years)
    w_recent_year: float = 0.60
    w_prev_year: float = 0.30
    w_prior2years: float = 0.10

    # Constraints (meta-selection)
    min_oos_windows_selected: int = 3
    min_selected_in_recent_year: int = 1

    # Risk tail percentile
    risk_tail_q: float = 0.05

    # Last N years for lot sizing
    lot_lookback_years: int = 2

    # Numerical stability
    eps: float = 1e-9


# ============================================================
# EXHAUSTIVE DATETIME PARSING (robust: mixed formats + excel serial)
# ============================================================
_RE_ISO_DATE = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s*$")
_RE_ISO_MIN  = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*$")
_RE_ISO_SEC  = re.compile(r"^\s*\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*$")

_RE_DMY_DATE = re.compile(r"^\s*\d{2}-\d{2}-\d{4}\s*$")
_RE_DMY_MIN  = re.compile(r"^\s*\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}\s*$")
_RE_DMY_SEC  = re.compile(r"^\s*\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}\s*$")

_RE_NUMERIC = re.compile(r"^\s*-?\d+(\.\d+)?\s*$")


def _try_parse_formats(s: pd.Series, formats: List[str]) -> pd.Series:
    out = pd.to_datetime(pd.Series([pd.NaT] * len(s), index=s.index), errors="coerce")
    remaining = out.isna()

    for fmt in formats:
        if not remaining.any():
            break
        parsed = pd.to_datetime(s[remaining], errors="coerce", format=fmt)
        out.loc[remaining] = parsed
        remaining = out.isna()

    return out


def _parse_excel_serial(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    mask = num.notna() & (num >= 20000) & (num <= 90000)
    out = pd.to_datetime(pd.Series([pd.NaT] * len(series), index=series.index), errors="coerce")
    if mask.any():
        out.loc[mask] = pd.to_datetime(num.loc[mask], origin="1899-12-30", unit="D", errors="coerce")
    return out


def parse_datetime_series_exhaustive(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.to_datetime(series, errors="coerce")

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    s0 = series.copy()
    serial_parsed = _parse_excel_serial(s0)

    s = s0.astype(str).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "NaN": np.nan, "None": np.nan})
    s = s.str.replace("/", "-", regex=False).str.replace(".", "-", regex=False)

    out = serial_parsed.copy()
    remaining = out.isna() & s.notna()

    if remaining.any():
        s_rem = s[remaining]

        iso_date_mask = s_rem.str.match(_RE_ISO_DATE)
        iso_min_mask  = s_rem.str.match(_RE_ISO_MIN)
        iso_sec_mask  = s_rem.str.match(_RE_ISO_SEC)

        dmy_date_mask = s_rem.str.match(_RE_DMY_DATE)
        dmy_min_mask  = s_rem.str.match(_RE_DMY_MIN)
        dmy_sec_mask  = s_rem.str.match(_RE_DMY_SEC)

        if iso_sec_mask.any():
            out.loc[remaining[remaining].index[iso_sec_mask]] = pd.to_datetime(
                s_rem[iso_sec_mask], errors="coerce", format="%Y-%m-%d %H:%M:%S"
            )
        if iso_min_mask.any():
            out.loc[remaining[remaining].index[iso_min_mask]] = pd.to_datetime(
                s_rem[iso_min_mask], errors="coerce", format="%Y-%m-%d %H:%M"
            )
        if iso_date_mask.any():
            out.loc[remaining[remaining].index[iso_date_mask]] = pd.to_datetime(
                s_rem[iso_date_mask], errors="coerce", format="%Y-%m-%d"
            )

        if dmy_sec_mask.any():
            out.loc[remaining[remaining].index[dmy_sec_mask]] = pd.to_datetime(
                s_rem[dmy_sec_mask], errors="coerce", format="%d-%m-%Y %H:%M:%S"
            )
        if dmy_min_mask.any():
            out.loc[remaining[remaining].index[dmy_min_mask]] = pd.to_datetime(
                s_rem[dmy_min_mask], errors="coerce", format="%d-%m-%Y %H:%M"
            )
        if dmy_date_mask.any():
            out.loc[remaining[remaining].index[dmy_date_mask]] = pd.to_datetime(
                s_rem[dmy_date_mask], errors="coerce", format="%d-%m-%Y"
            )

    remaining = out.isna() & s.notna()
    if remaining.any():
        out2 = _try_parse_formats(
            s[remaining],
            formats=[
                "%d-%m-%Y %H:%M:%S",
                "%d-%m-%Y %H:%M",
                "%d-%m-%Y",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d",
            ],
        )
        out.loc[remaining] = out2

    remaining = out.isna() & s.notna()
    if remaining.any():
        out.loc[remaining] = pd.to_datetime(s[remaining], errors="coerce", dayfirst=True)

    return out


def parse_date_series_exhaustive(series: pd.Series) -> pd.Series:
    dt = parse_datetime_series_exhaustive(series)
    return pd.to_datetime(dt, errors="coerce").dt.normalize()


# ============================================================
# EXPIRY CALENDAR (UNIQUE ExpiryDate list)
# ============================================================
class ExpiryCalendar:
    def __init__(self, expiry_dates_d: np.ndarray):
        self.expiry_dates_d = expiry_dates_d
        self.min_expiry = pd.Timestamp(expiry_dates_d[0]).normalize()
        self.max_expiry = pd.Timestamp(expiry_dates_d[-1]).normalize()

    def map_trade_day_to_expiry(self, dates: pd.Series) -> pd.Series:
        if dates is None or dates.empty:
            return pd.to_datetime(dates, errors="coerce")

        d = parse_date_series_exhaustive(dates)
        d64 = d.values.astype("datetime64[D]")

        idx = np.searchsorted(self.expiry_dates_d, d64, side="left")
        out = np.full(len(d64), np.datetime64("NaT", "D"), dtype="datetime64[D]")
        ok = (idx >= 0) & (idx < len(self.expiry_dates_d))
        out[ok] = self.expiry_dates_d[idx[ok]]

        return pd.to_datetime(out).normalize()


def load_unique_expiry_calendar(path: str) -> ExpiryCalendar:
    t0 = time.time()
    df = pd.read_csv(path, sep=None, engine="python")

    if "ExpiryDate" in df.columns:
        exp_col = df["ExpiryDate"]
    else:
        exp_col = df.iloc[:, 0]

    exp = parse_date_series_exhaustive(exp_col)
    exp = exp.dropna().sort_values().unique()
    exp_u64d = pd.to_datetime(exp).to_numpy(dtype="datetime64[ns]").astype("datetime64[D]")

    if len(exp_u64d) == 0:
        raise ValueError("No valid expiry dates parsed from file.")

    if PRINT_TIMINGS:
        weeks = len(exp_u64d)
        print(f"[CAL] Loaded UNIQUE expiry list: {weeks} expiries | "
              f"{pd.Timestamp(exp_u64d[0]).date()} -> {pd.Timestamp(exp_u64d[-1]).date()} | "
              f"elapsed={time.time()-t0:.2f}s")

    return ExpiryCalendar(exp_u64d)


def expiry_dates_in_period(cal: ExpiryCalendar, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    exp = pd.to_datetime(cal.expiry_dates_d)
    exp = exp[(exp >= start) & (exp <= end)]
    return pd.DatetimeIndex(exp).sort_values()


def count_expiries_in_range(cal: ExpiryCalendar, start: pd.Timestamp, end: pd.Timestamp) -> int:
    exp = pd.to_datetime(cal.expiry_dates_d)
    return int(((exp >= start) & (exp <= end)).sum())


# ============================================================
# LOADER: return ROW-LEVEL records so trade lines can be counted
# ============================================================
def load_trade_rows_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if ("Entry Time" in df.columns) and ("PnL With Spread" in df.columns):
        out = pd.DataFrame({
            "date": parse_datetime_series_exhaustive(df["Entry Time"]),
            "pnl_row": pd.to_numeric(df["PnL With Spread"], errors="coerce").fillna(0.0),
        }).dropna(subset=["date"])
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        return out.sort_values("date").reset_index(drop=True)

    if ("Date" in df.columns) and ("PnL" in df.columns):
        out = pd.DataFrame({
            "date": parse_datetime_series_exhaustive(df["Date"]),
            "pnl_row": pd.to_numeric(df["PnL"], errors="coerce").fillna(0.0),
        }).dropna(subset=["date"])
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
        return out.sort_values("date").reset_index(drop=True)

    raise ValueError(
        f"File missing required columns. Need either "
        f"(Entry Time + PnL With Spread) OR (Date + PnL): {path}"
    )


def discover_csvs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def strategy_id_from_filename(path: str) -> Tuple[str, str]:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]
    return (name, name)


# ============================================================
# GROUP: rows -> expiry-cycle series
# ============================================================
def group_rows_to_expiry_cycles(rows_df: pd.DataFrame, cal: ExpiryCalendar) -> pd.DataFrame:
    if rows_df is None or rows_df.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "n_trades"])

    tmp = rows_df.copy()
    tmp["date"] = parse_date_series_exhaustive(tmp["date"])
    tmp = tmp.dropna(subset=["date"])

    tmp = tmp[(tmp["date"] >= BACKTEST_START) & (tmp["date"] <= BACKTEST_END)]
    if tmp.empty:
        return pd.DataFrame(columns=["date", "daily_pnl", "n_trades"])

    tmp["expiry_date"] = cal.map_trade_day_to_expiry(tmp["date"])
    tmp = tmp.dropna(subset=["expiry_date"])

    g = tmp.groupby("expiry_date", as_index=False).agg(
        daily_pnl=("pnl_row", "sum"),
        n_trades=("pnl_row", "size"),
    ).rename(columns={"expiry_date": "date"})

    g["date"] = parse_date_series_exhaustive(g["date"])
    g = g[(g["date"] >= BACKTEST_START) & (g["date"] <= BACKTEST_END)]

    return g.sort_values("date").reset_index(drop=True)


# ============================================================
# METRICS
# ============================================================
def _max_drawdown_raw(x: pd.Series) -> float:
    if x is None or len(x) == 0:
        return 0.0
    eq = x.cumsum()
    rm = eq.cummax()
    dd = eq - rm
    return float(dd.min())

def _cvar5_abs(x: pd.Series, q: float) -> float:
    if x is None or len(x) == 0:
        return 0.0
    qv = float(x.quantile(q))
    tail = x[x <= qv]
    if len(tail) == 0:
        return 0.0
    m = float(tail.mean())
    return float(max(0.0, -m))

def compute_window_metrics(df_window: pd.DataFrame, cfg: WalkForwardConfig, weeks_in_window: int) -> Dict[str, Any]:
    if df_window is None or len(df_window) == 0:
        return {
            "median_pnl_raw": 0.0,
            "median_pnl_adj": 0.0,
            "weeks_in_window": int(weeks_in_window),
            "weeks_traded": 0,
            "coverage_ratio": 0.0,
            "trades_in_window": 0,
            "win_rate": 0.0,
            "cvar5_abs": 0.0,
            "max_drawdown_abs": 0.0,
        }

    s = df_window["daily_pnl"]
    median_raw = float(s.median())

    weeks_traded = int((df_window["n_trades"] > 0).sum()) if "n_trades" in df_window.columns else int(len(df_window))
    denom = int(max(0, weeks_in_window))
    coverage_ratio = float(weeks_traded / denom) if denom > 0 else 0.0
    median_adj = float(median_raw * coverage_ratio)

    trades_in_window = int(df_window["n_trades"].sum()) if "n_trades" in df_window.columns else int(len(df_window))

    if "n_trades" in df_window.columns:
        traded_mask = (df_window["n_trades"] > 0)
        wins = int(((df_window["daily_pnl"] > 0) & traded_mask).sum())
        weeks_traded_eff = int(traded_mask.sum())
    else:
        wins = int((df_window["daily_pnl"] > 0).sum())
        weeks_traded_eff = int(len(df_window))

    win_rate = float(wins / weeks_traded_eff) if weeks_traded_eff > 0 else 0.0

    risk_abs = _cvar5_abs(s, cfg.risk_tail_q)

    mdd_raw = _max_drawdown_raw(s)
    mdd_abs = float(max(0.0, -mdd_raw))

    return {
        "median_pnl_raw": median_raw,
        "median_pnl_adj": median_adj,
        "weeks_in_window": denom,
        "weeks_traded": weeks_traded,
        "coverage_ratio": coverage_ratio,
        "trades_in_window": trades_in_window,
        "win_rate": win_rate,
        "cvar5_abs": risk_abs,
        "max_drawdown_abs": mdd_abs,
    }


# ============================================================
# PREFILTER HELPERS (NEW)
# ============================================================
def _max_drawdown_pct_from_pnl(pnl: pd.Series, capital: float) -> float:
    """
    Equity = capital + cumsum(pnl)
    DD = equity - running_max(equity)  (<= 0)
    mdd_pct = abs(min(DD)) / capital
    """
    if pnl is None or len(pnl) == 0:
        return 0.0
    eq = capital + pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    mdd_abs = float(max(0.0, -dd.min()))
    return float(mdd_abs / max(capital, 1e-9))


def compute_prefilter_stats(df_series: pd.DataFrame, total_expiries_period: int, capital: float) -> Dict[str, Any]:
    """
    Dataset-level stats computed on the FULL expiry-cycle series (entire backtest window).
    """
    if df_series is None or df_series.empty:
        return {
            "total_pnl": 0.0,
            "total_return_pct": 0.0,
            "mdd_pct": 0.0,
            "weeks_traded": 0,
            "trade_coverage": 0.0,
            "win_rate": 0.0,
        }

    pnl = df_series["daily_pnl"].astype(float)
    total_pnl = float(pnl.sum())
    total_return_pct = float(total_pnl / max(capital, 1e-9))

    if "n_trades" in df_series.columns:
        traded_mask = (df_series["n_trades"] > 0)
        weeks_traded = int(traded_mask.sum())
    else:
        traded_mask = pd.Series([True] * len(df_series), index=df_series.index)
        weeks_traded = int(len(df_series))

    denom = int(max(1, total_expiries_period))
    trade_coverage = float(weeks_traded / denom)

    if weeks_traded > 0:
        wins = int((df_series.loc[traded_mask, "daily_pnl"] > 0).sum())
        win_rate = float(wins / weeks_traded)
    else:
        win_rate = 0.0

    mdd_pct = _max_drawdown_pct_from_pnl(pnl, capital=capital)

    return {
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "mdd_pct": mdd_pct,
        "weeks_traded": weeks_traded,
        "trade_coverage": trade_coverage,
        "win_rate": win_rate,
    }


def prefilter_should_drop(stats: Dict[str, Any]) -> Tuple[bool, str]:
    reasons = []
    if float(stats["mdd_pct"]) > PREF_MAX_DD_PCT:
        reasons.append(f"DD>{PREF_MAX_DD_PCT:.0%}")
    if float(stats["trade_coverage"]) < PREF_MIN_TRADE_COVERAGE:
        reasons.append(f"Trades<{PREF_MIN_TRADE_COVERAGE:.0%}")
    if float(stats["total_return_pct"]) < PREF_MIN_TOTAL_RETURN_PCT:
        reasons.append(f"Ret<{PREF_MIN_TOTAL_RETURN_PCT:.0%}")
    if float(stats["win_rate"]) < PREF_MIN_WIN_RATE:
        reasons.append(f"Win<{PREF_MIN_WIN_RATE:.0%}")
    return (len(reasons) > 0), ";".join(reasons)


# ============================================================
# PERCENTILES + SCORE
# ============================================================
def _percentile_rank(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    if series.empty:
        return series
    pct = series.rank(pct=True, method="average")
    return pct if higher_is_better else (1.0 - pct)

def add_percentile_scores(df: pd.DataFrame, cfg: WalkForwardConfig) -> pd.DataFrame:
    out = df.copy()
    out["pctl_pnl"]  = _percentile_rank(out["median_pnl_adj"], higher_is_better=True)
    out["pctl_win"]  = _percentile_rank(out["win_rate"], higher_is_better=True)
    out["pctl_risk"] = _percentile_rank(out["cvar5_abs"], higher_is_better=False)
    out["pctl_mdd"]  = _percentile_rank(out["max_drawdown_abs"], higher_is_better=False)

    out["score"] = (
        cfg.w_pnl  * out["pctl_pnl"] +
        cfg.w_win  * out["pctl_win"] +
        cfg.w_risk * out["pctl_risk"] +
        cfg.w_mdd  * out["pctl_mdd"]
    )
    return out


# ============================================================
# FOLDS (calendar months) - built ONCE globally
# ============================================================
def build_calendar_month_folds(all_dates: pd.Series, cfg: WalkForwardConfig) -> List[Dict[str, Any]]:
    dts = parse_date_series_exhaustive(all_dates).dropna().sort_values().unique()
    if len(dts) == 0:
        return []

    start_m = pd.Timestamp(dts[0]).to_period("M")
    end_m = pd.Timestamp(dts[-1]).to_period("M")
    months = pd.period_range(start=start_m, end=end_m, freq="M")

    folds: List[Dict[str, Any]] = []
    i = 0
    while True:
        train_months = months[i: i + cfg.is_months]
        test_months = months[i + cfg.is_months: i + cfg.is_months + cfg.oos_months]

        if len(train_months) < cfg.is_months or len(test_months) < cfg.oos_months:
            break

        train_start = train_months[0].to_timestamp(how="start").normalize()
        train_end = train_months[-1].to_timestamp(how="end").normalize()

        test_start = (train_end + pd.Timedelta(days=1)).normalize()
        test_end = test_months[-1].to_timestamp(how="end").normalize()

        folds.append({
            "fold_id": len(folds) + 1,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
            "oos_year": int(test_start.year),
        })
        i += cfg.step_months

    return folds

def _mask_date_range(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    d = parse_date_series_exhaustive(df["date"])
    return (d >= start) & (d <= end)


# ============================================================
# CORE WALK-FORWARD RUN (one bucket) using GLOBAL folds
# ============================================================
def run_walkforward_from_folder(
    folder: str,
    cfg: WalkForwardConfig,
    cal: ExpiryCalendar,
    folds: List[Dict[str, Any]],
    total_expiries_period: int,
    bucket_id: str = "",
) -> Dict[str, pd.DataFrame]:
    t0 = time.time()

    paths = discover_csvs(folder)
    if not paths:
        raise FileNotFoundError(f"No CSV files found in: {folder}")

    print(f"\n[{bucket_id}] Found {len(paths)} CSV files in: {folder}")

    daily_map: Dict[Tuple[str, str], pd.DataFrame] = {}
    parse_errors = 0
    shown_errors = 0

    # PREFILTER counters (per bucket)
    prefilter_dropped = 0
    prefilter_passed = 0
    prefilter_reasons_count: Dict[str, int] = {}

    for i, p in enumerate(paths, start=1):
        sid = strategy_id_from_filename(p)
        try:
            rows_df = load_trade_rows_csv(p)
            cycle_df = group_rows_to_expiry_cycles(rows_df, cal)
        except Exception as e:
            parse_errors += 1
            if shown_errors < SHOW_EXCEPTIONS_SAMPLE:
                shown_errors += 1
                print(f"[{bucket_id}] !! ERROR reading {os.path.basename(p)}: {repr(e)}")
            continue

        # ====================================================
        # PREFILTER (NEW): evaluate full dataset metrics
        # ====================================================
        stats = compute_prefilter_stats(
            df_series=cycle_df,
            total_expiries_period=total_expiries_period,
            capital=CAPITAL
        )
        drop, reason = prefilter_should_drop(stats)

        if drop:
            prefilter_dropped += 1
            if reason:
                # Count each reason token (split by ;)
                for tok in reason.split(";"):
                    tok = tok.strip()
                    if tok:
                        prefilter_reasons_count[tok] = prefilter_reasons_count.get(tok, 0) + 1
            continue

        prefilter_passed += 1
        daily_map[sid] = cycle_df

        if (i % PRINT_EVERY_FILES) == 0 or i == 1 or i == len(paths):
            elapsed = time.time() - t0
            msg = (f"[{bucket_id}] Loaded {i}/{len(paths)} files | "
                   f"strategies_ok={len(daily_map)} | errors={parse_errors} | "
                   f"prefilter_passed={prefilter_passed} dropped={prefilter_dropped}")
            if PRINT_TIMINGS:
                msg += f" | elapsed={elapsed:,.1f}s"
            print(msg)

    print(f"[{bucket_id}] Prefilter summary: passed={prefilter_passed}, dropped={prefilter_dropped}, parse_errors={parse_errors}")
    if prefilter_reasons_count:
        top_reasons = sorted(prefilter_reasons_count.items(), key=lambda x: -x[1])
        top_str = ", ".join([f"{k}:{v}" for k, v in top_reasons[:10]])
        print(f"[{bucket_id}] Prefilter reasons (top): {top_str}")

    if len(daily_map) == 0:
        raise ValueError(f"[{bucket_id}] All strategies were filtered out or failed to parse. errors={parse_errors}, dropped={prefilter_dropped}")

    folds_df = pd.DataFrame(folds)
    print(f"[{bucket_id}] Using GLOBAL folds: {len(folds)} folds | Backtest {BACKTEST_START.date()}..{BACKTEST_END.date()}")

    strategies = list(daily_map.keys())

    is_rows, sel_rows, oos_rows = [], [], []

    for fi, f in enumerate(folds, start=1):
        fold_id = f["fold_id"]

        is_weeks_total = count_expiries_in_range(cal, f["train_start"], f["train_end"])
        oos_weeks_total = count_expiries_in_range(cal, f["test_start"], f["test_end"])

        if (fi % PRINT_EVERY_FOLDS) == 0:
            print(f"[{bucket_id}] Fold {fi}/{len(folds)} (fold_id={fold_id}) "
                  f"IS {f['train_start'].date()}..{f['train_end'].date()} (weeks={is_weeks_total}) | "
                  f"OOS {f['test_start'].date()}..{f['test_end'].date()} (weeks={oos_weeks_total})")

        # 1) IN-SAMPLE
        fold_scores = []
        for sid in strategies:
            df_s = daily_map[sid]
            mask = _mask_date_range(df_s, f["train_start"], f["train_end"])
            df_w = df_s.loc[mask]
            m = compute_window_metrics(df_w, cfg, weeks_in_window=is_weeks_total)

            fold_scores.append({
                "fold_id": fold_id,
                "strategy_id": sid,
                "strategy_name": sid[0],
                "params_key": sid[1],
                "train_start": f["train_start"],
                "train_end": f["train_end"],
                "test_start": f["test_start"],
                "test_end": f["test_end"],
                **m,
            })

        is_fold_df = pd.DataFrame(fold_scores)
        is_fold_df = add_percentile_scores(is_fold_df, cfg)
        is_fold_df = is_fold_df.sort_values("score", ascending=False).reset_index(drop=True)
        is_fold_df["is_rank"] = np.arange(1, len(is_fold_df) + 1)
        is_rows.extend(is_fold_df.to_dict("records"))

        # 2) Select TOP K (IS)
        selected = is_fold_df.head(cfg.top_k).copy()
        selected["selected_rank"] = np.arange(1, len(selected) + 1)

        for _, r in selected.iterrows():
            sel_rows.append({
                "fold_id": fold_id,
                "strategy_id": r["strategy_id"],
                "strategy_name": r["strategy_name"],
                "params_key": r["params_key"],
                "is_rank": int(r["is_rank"]),
                "selected_rank": int(r["selected_rank"]),
                "train_start": f["train_start"],
                "train_end": f["train_end"],
                "test_start": f["test_start"],
                "test_end": f["test_end"],

                "is_median_pnl_raw": float(r["median_pnl_raw"]),
                "is_median_pnl_adj": float(r["median_pnl_adj"]),
                "is_weeks_in_window": int(r["weeks_in_window"]),
                "is_weeks_traded": int(r["weeks_traded"]),
                "is_coverage_ratio": float(r["coverage_ratio"]),
                "is_trades_in_window": int(r["trades_in_window"]),

                "is_win_rate": float(r.get("win_rate", 0.0)),
                "is_cvar5_abs": float(r["cvar5_abs"]),
                "is_max_drawdown_abs": float(r["max_drawdown_abs"]),

                "is_pctl_pnl": float(r["pctl_pnl"]),
                "is_pctl_win": float(r.get("pctl_win", 0.0)),
                "is_pctl_risk": float(r["pctl_risk"]),
                "is_pctl_mdd": float(r["pctl_mdd"]),
                "is_score": float(r["score"]),
            })

        # 3) OUT-OF-SAMPLE: evaluate selected only
        oos_fold_rows = []
        for _, r in selected.iterrows():
            sid = r["strategy_id"]
            df_s = daily_map[sid]
            mask = _mask_date_range(df_s, f["test_start"], f["test_end"])
            df_w = df_s.loc[mask]
            m = compute_window_metrics(df_w, cfg, weeks_in_window=oos_weeks_total)

            oos_fold_rows.append({
                "fold_id": fold_id,
                "oos_year": int(f["oos_year"]),
                "strategy_id": sid,
                "strategy_name": sid[0],
                "params_key": sid[1],
                "selected_rank": int(r["selected_rank"]),
                "test_start": f["test_start"],
                "test_end": f["test_end"],
                **{f"oos_{k}": v for k, v in m.items()},
            })

        oos_fold_df = pd.DataFrame(oos_fold_rows)
        if not oos_fold_df.empty:
            tmp = oos_fold_df.rename(columns={
                "oos_median_pnl_adj": "median_pnl_adj",
                "oos_win_rate": "win_rate",
                "oos_cvar5_abs": "cvar5_abs",
                "oos_max_drawdown_abs": "max_drawdown_abs",
            })
            tmp = add_percentile_scores(tmp, cfg)
            oos_fold_df["oos_pctl_pnl"] = tmp["pctl_pnl"].values
            oos_fold_df["oos_pctl_win"] = tmp["pctl_win"].values
            oos_fold_df["oos_pctl_risk"] = tmp["pctl_risk"].values
            oos_fold_df["oos_pctl_mdd"] = tmp["pctl_mdd"].values
            oos_fold_df["oos_score"] = tmp["score"].values

        oos_rows.extend(oos_fold_df.to_dict("records"))

    is_rankings_df = pd.DataFrame(is_rows)
    selections_df = pd.DataFrame(sel_rows)
    oos_results_df = pd.DataFrame(oos_rows)

    # 4) FINAL META-SELECTION (unchanged)
    if oos_results_df.empty:
        final_scores_df = pd.DataFrame()
    else:
        most_recent_year = int(oos_results_df["oos_year"].max())
        prev_year = most_recent_year - 1
        prior_years = [most_recent_year - 2, most_recent_year - 3]

        selected_count_total = (
            oos_results_df.groupby("strategy_id")["fold_id"].nunique()
            .rename("selected_count_total").reset_index()
        )
        selected_count_recent = (
            oos_results_df[oos_results_df["oos_year"] == most_recent_year]
            .groupby("strategy_id")["fold_id"].nunique()
            .rename("selected_count_recent_year").reset_index()
        )

        def median_for_years(years: List[int], col: str) -> pd.DataFrame:
            return (
                oos_results_df[oos_results_df["oos_year"].isin(years)]
                .groupby("strategy_id")["oos_score"].median()
                .rename(col).reset_index()
            )

        med_recent = median_for_years([most_recent_year], "median_recent_year")
        med_prev = median_for_years([prev_year], "median_prev_year")
        med_prior2 = median_for_years(prior_years, "median_prior2years")

        final = (
            med_recent
            .merge(med_prev, on="strategy_id", how="left")
            .merge(med_prior2, on="strategy_id", how="left")
            .merge(selected_count_total, on="strategy_id", how="left")
            .merge(selected_count_recent, on="strategy_id", how="left")
        )

        final["median_prev_year"] = final["median_prev_year"].fillna(0.0)
        final["median_prior2years"] = final["median_prior2years"].fillna(0.0)
        final["selected_count_total"] = final["selected_count_total"].fillna(0).astype(int)
        final["selected_count_recent_year"] = final["selected_count_recent_year"].fillna(0).astype(int)

        before_n = len(final)
        final = final[
            (final["selected_count_total"] >= cfg.min_oos_windows_selected) &
            (final["selected_count_recent_year"] >= cfg.min_selected_in_recent_year)
        ].copy()
        after_n = len(final)

        final["final_score"] = (
            cfg.w_recent_year * final["median_recent_year"] +
            cfg.w_prev_year * final["median_prev_year"] +
            cfg.w_prior2years * final["median_prior2years"]
        )

        final["strategy_name"] = final["strategy_id"].apply(lambda x: x[0])
        final["params_key"] = final["strategy_id"].apply(lambda x: x[1])

        final_scores_df = final.sort_values("final_score", ascending=False).reset_index(drop=True)

        print(f"[{bucket_id}] Final meta-selection: candidates={before_n} -> passed_constraints={after_n} "
              f"(need total>= {cfg.min_oos_windows_selected}, recent>= {cfg.min_selected_in_recent_year})")

    if PRINT_TIMINGS:
        print(f"[{bucket_id}] Completed bucket in {time.time()-t0:,.1f}s | strategies_ok={len(daily_map)} | errors={parse_errors}")

    return {
        "folds_df": folds_df,
        "is_rankings_df": is_rankings_df,
        "selections_df": selections_df,
        "oos_results_df": oos_results_df,
        "final_scores_df": final_scores_df,
        "_daily_map": daily_map,
    }


# ============================================================
# LOTS / WEIGHTS (last 2 years) - FIX weeks_in_2y via calendar
# ============================================================
def compute_last2y_stats(df_series: pd.DataFrame, cfg: WalkForwardConfig, cal: ExpiryCalendar) -> Dict[str, Any]:
    if df_series.empty:
        return {
            "median_2y_raw": 0.0,
            "median_2y_adj": 0.0,
            "weeks_in_2y": 0,
            "weeks_traded_2y": 0,
            "coverage_2y": 0.0,
            "cvar5_2y_abs": 0.0,
        }

    end = min(BACKTEST_END, cal.max_expiry).normalize()
    start = (end - pd.DateOffset(years=cfg.lot_lookback_years)).normalize()

    weeks_in_2y = count_expiries_in_range(cal, start, end)

    m = _mask_date_range(df_series, start, end)
    df2 = df_series.loc[m].copy()

    if df2.empty:
        return {
            "median_2y_raw": 0.0,
            "median_2y_adj": 0.0,
            "weeks_in_2y": int(weeks_in_2y),
            "weeks_traded_2y": 0,
            "coverage_2y": 0.0,
            "cvar5_2y_abs": 0.0,
        }

    s = df2["daily_pnl"]
    median_raw = float(s.median())

    weeks_traded_2y = int((df2["n_trades"] > 0).sum()) if "n_trades" in df2.columns else int(len(df2))
    denom = int(max(0, weeks_in_2y))
    coverage_2y = float(weeks_traded_2y / denom) if denom > 0 else 0.0

    median_adj = float(median_raw * coverage_2y)
    cvar_abs = _cvar5_abs(s, cfg.risk_tail_q)

    return {
        "median_2y_raw": median_raw,
        "median_2y_adj": median_adj,
        "weeks_in_2y": denom,
        "weeks_traded_2y": weeks_traded_2y,
        "coverage_2y": coverage_2y,
        "cvar5_2y_abs": float(cvar_abs),
    }


def allocate_lots_float_and_integer(df: pd.DataFrame, total_lots: int) -> pd.DataFrame:
    out = df.copy()
    wsum = float(out["weight"].sum()) if "weight" in out.columns and len(out) else 0.0

    if wsum <= 0:
        out["lots_float"] = 0.0
        out["lots_rounded"] = 0
        return out

    out["lots_float"] = total_lots * out["weight"] / wsum

    base = np.floor(out["lots_float"]).astype(int)
    remainder = out["lots_float"] - base

    lots_left = int(total_lots - base.sum())
    add = np.zeros(len(out), dtype=int)
    if lots_left > 0:
        idx = np.argsort(-remainder.values)
        add[idx[:lots_left]] = 1

    out["lots_rounded"] = base + add

    diff = int(total_lots - out["lots_rounded"].sum())
    if diff != 0 and len(out) > 0:
        label = out["weight"].idxmax()
        out.loc[label, "lots_rounded"] = int(out.loc[label, "lots_rounded"] + diff)

    return out


# ============================================================
# EXCEL EXPORT HELPERS (unchanged)
# ============================================================
def _safe_sheet_name(name: str) -> str:
    bad = ['\\', '/', '*', '[', ']', ':', '?']
    for b in bad:
        name = name.replace(b, "_")
    name = name.strip()
    return (name[:31] if len(name) > 31 else name) or "Sheet"

def _sanitize_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].apply(lambda x: str(x) if isinstance(x, (tuple, list, dict, set)) else x)
    return out

def _style_sheet_as_table(ws, table_name: str, nrows: int, ncols: int, freeze_panes_cell="A2"):
    ws.freeze_panes = freeze_panes_cell
    end_col = ws.cell(row=1, column=ncols).column_letter
    end_row = nrows
    table = Table(displayName=table_name, ref=f"A1:{end_col}{end_row}")

    style = TableStyleInfo(
        name="TableStyleMedium9",
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    table.tableStyleInfo = style
    ws.add_table(table)

    header_fill = PatternFill("solid", fgColor="1F4E79")
    header_font = Font(color="FFFFFF", bold=True)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    for col in range(1, ncols + 1):
        ws.column_dimensions[ws.cell(row=1, column=col).column_letter].width = 18

def _write_df(ws, df: pd.DataFrame, table_name: str):
    df = _sanitize_for_excel(df)
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)
    if ws.max_row >= 2 and ws.max_column >= 1:
        _style_sheet_as_table(ws, table_name, ws.max_row, ws.max_column)

def export_all_to_excel(cfg, top3_df, lots_detail_df, bucket_outputs, out_path, total_expiries_period):
    t0 = time.time()
    print(f"\n[EXCEL] Writing workbook: {out_path}")

    wb = Workbook()
    wb.remove(wb.active)

    ws = wb.create_sheet("Config")
    ws.append(["Key", "Value"])
    rows = [
        ("Grouping", "EXPIRY DATE (weekly cycle). date=ExpiryDate; daily_pnl=cycle pnl."),
        ("Backtest window", f"{BACKTEST_START.date()} .. {BACKTEST_END.date()}"),
        ("Total expiries in backtest", int(total_expiries_period)),
        ("IS Months", cfg.is_months),
        ("OOS Months", cfg.oos_months),
        ("Step Months", cfg.step_months),
        ("Top K (IS)", cfg.top_k),
        ("Final top per bucket", TOP_N_FINAL_PER_BUCKET),
        ("Score weights pnl/win/risk/mdd", f"{cfg.w_pnl}/{cfg.w_win}/{cfg.w_risk}/{cfg.w_mdd}"),
        ("PnL metric in score", "median_pnl_adj = median_pnl_raw * (weeks_traded / weeks_in_window)"),
        ("Win metric in score", "win_rate = wins / weeks_traded  (wins: daily_pnl>0 among traded weeks)"),
        ("Risk metric in score", "cvar5_abs = max(0, -mean(worst 5% tail))  [POSITIVE magnitude; lower is better]"),
        ("Risk tail q", cfg.risk_tail_q),
        ("Lot lookback years", cfg.lot_lookback_years),
        ("Lot median (2y)", "median_2y_adj = median_2y_raw * (weeks_traded_2y / weeks_in_2y)"),
        ("Lot risk (2y)", "cvar5_2y_abs = max(0, -mean(worst 5% tail))"),
        ("Weight formula", "weight = sqrt( median_2y_adj / cvar5_2y_abs )"),
        ("Total lots", TOTAL_LOTS),

        # Prefilter description (added to config only; does not alter rest)
        ("Prefilter capital", CAPITAL),
        ("Prefilter rules", f"Drop if DD>{PREF_MAX_DD_PCT:.0%} OR Trades<{PREF_MIN_TRADE_COVERAGE:.0%} OR "
                            f"Return<{PREF_MIN_TOTAL_RETURN_PCT:.0%} OR Win<{PREF_MIN_WIN_RATE:.0%}"),
        ("Prefilter metrics", "DD% on (capital + cumsum(pnl)); Trades=weeks_traded/total_expiries; Return=sum(pnl)/capital; Win=wins/weeks_traded"),
    ]
    for k, v in rows:
        ws.append([k, v])

    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 110
    ws["A1"].font = Font(bold=True)
    ws["B1"].font = Font(bold=True)

    ws_t3 = wb.create_sheet("Top3_Final_Per_Bucket")
    _write_df(ws_t3, top3_df, "tblTop3Final")

    ws_ld = wb.create_sheet("Selected_Weights_Lots")
    _write_df(ws_ld, lots_detail_df, "tblWeightsLots")

    for bucket_id, out in bucket_outputs.items():
        prefix = _safe_sheet_name(bucket_id)
        print(f"[EXCEL] Writing bucket sheets for {bucket_id}...")

        folds = out.get("folds_df", pd.DataFrame())
        isdf = out.get("is_rankings_df", pd.DataFrame())
        seldf = out.get("selections_df", pd.DataFrame())
        oosdf = out.get("oos_results_df", pd.DataFrame())
        final = out.get("final_scores_df", pd.DataFrame())

        if not folds.empty:
            _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_Folds")), folds, f"t{prefix}Folds")
        if not isdf.empty:
            _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_IS")), isdf, f"t{prefix}IS")
        if not seldf.empty:
            _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_Selected")), seldf, f"t{prefix}Sel")
        if not oosdf.empty:
            _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_OOS")), oosdf, f"t{prefix}OOS")
        if not final.empty:
            _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_Final")), final, f"t{prefix}Final")

    wb.save(out_path)
    if PRINT_TIMINGS:
        print(f"[EXCEL] Saved workbook in {time.time() - t0:,.1f}s")


# ============================================================
# MAIN
# ============================================================
def main():
    cfg = WalkForwardConfig()

    print(f"[SANITY] TOTAL_LOTS={TOTAL_LOTS}, top_k(IS)={cfg.top_k}, buckets={len(BUCKET_DIRS)}, top_final_per_bucket={TOP_N_FINAL_PER_BUCKET}")
    print(f"[SANITY] Buckets: {', '.join(BUCKET_DIRS.keys())}")
    print(f"[SANITY] Fixed backtest window: {BACKTEST_START.date()}..{BACKTEST_END.date()}")

    cal = load_unique_expiry_calendar(EXPIRY_CALENDAR_CSV)

    exp_in_period = expiry_dates_in_period(cal, BACKTEST_START, BACKTEST_END)
    total_expiries_period = int(len(exp_in_period))
    print(f"[SANITY] Total expiries in backtest window: {total_expiries_period}")

    folds = build_calendar_month_folds(pd.Series(exp_in_period), cfg)
    if not folds:
        raise ValueError("No folds could be built from expiry dates in the backtest window.")
    print(f"[SANITY] Built GLOBAL folds: {len(folds)} folds (will be reused for all buckets)")

    first_bucket = next(iter(BUCKET_DIRS.values()))
    out_xlsx = OUTPUT_XLSX or os.path.join(first_bucket, "walkforward_bucketed_audit_FIXED_FOLDS_WEEK_COVERAGE_FIXED_EXHAUSTIVE_100IS_PREFILTER.xlsx")

    bucket_outputs: Dict[str, Dict[str, pd.DataFrame]] = {}
    t_all = time.time()

    # 1) Run walkforward per bucket
    for bi, (bucket_id, folder) in enumerate(BUCKET_DIRS.items(), start=1):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"[{bucket_id}] Bucket folder not found: {folder}")

        print(f"\n=== BUCKET {bi}/{len(BUCKET_DIRS)}: {bucket_id} ===")
        print(f"[{bucket_id}] Path: {folder}")

        out = run_walkforward_from_folder(
            folder=folder,
            cfg=cfg,
            cal=cal,
            folds=folds,
            total_expiries_period=total_expiries_period,
            bucket_id=bucket_id,
        )
        bucket_outputs[bucket_id] = out

        final_df = out["final_scores_df"]
        if final_df.empty:
            print(f"[{bucket_id}] Final: EMPTY (No strategy passed constraints)")
        else:
            print(f"[{bucket_id}] Final: rows={len(final_df)} | top1={final_df.iloc[0]['strategy_name']} score={float(final_df.iloc[0]['final_score']):.6f}")

    # 2) Top-3 per bucket (unchanged)
    print("\n=== TOP-3 FINAL PER BUCKET ===")
    top3_rows = []
    for bucket_id, out in bucket_outputs.items():
        final_df = out.get("final_scores_df", pd.DataFrame())
        if final_df is None or final_df.empty:
            top3_rows.append({
                "bucket_id": bucket_id,
                "final_rank": "",
                "strategy_name": "",
                "params_key": "",
                "final_score": np.nan,
                "median_recent_year": np.nan,
                "median_prev_year": np.nan,
                "median_prior2years": np.nan,
                "selected_count_total": np.nan,
                "selected_count_recent_year": np.nan,
                "note": "No strategy passed constraints",
            })
            print(f"[{bucket_id}] Top3: NONE (no final strategies)")
            continue

        n_take = min(TOP_N_FINAL_PER_BUCKET, len(final_df))
        topn = final_df.head(n_take).copy()
        topn["final_rank"] = np.arange(1, len(topn) + 1)

        print(f"[{bucket_id}] Taking top {n_take} from final_scores_df")

        for _, r in topn.iterrows():
            top3_rows.append({
                "bucket_id": bucket_id,
                "final_rank": int(r["final_rank"]),
                "strategy_name": r["strategy_name"],
                "params_key": r["params_key"],
                "final_score": float(r["final_score"]),
                "median_recent_year": float(r.get("median_recent_year", 0.0)),
                "median_prev_year": float(r.get("median_prev_year", 0.0)),
                "median_prior2years": float(r.get("median_prior2years", 0.0)),
                "selected_count_total": int(r.get("selected_count_total", 0)),
                "selected_count_recent_year": int(r.get("selected_count_recent_year", 0)),
                "note": "",
            })

    top3_df = pd.DataFrame(top3_rows)

    # 3) Lots weights across all selections (unchanged)
    print("\n=== WEIGHTS + LOTS (ALL SELECTED STRATEGIES) ===")
    lot_rows = []

    selections = top3_df[(top3_df["strategy_name"].astype(str).str.len() > 0) & (top3_df["note"].fillna("") == "")]
    print(f"[LOTS] Total selected strategies (across all buckets): {len(selections)}")

    for _, r in selections.iterrows():
        bucket_id = r["bucket_id"]
        strategy_name = r["strategy_name"]
        final_rank = int(r["final_rank"])
        final_score = float(r["final_score"])

        daily_map = bucket_outputs[bucket_id]["_daily_map"]
        sid = (strategy_name, strategy_name)

        if sid not in daily_map:
            found = None
            for k in daily_map.keys():
                if k[0] == strategy_name:
                    found = k
                    break
            sid = found

        if sid is None:
            lot_rows.append({
                "bucket_id": bucket_id,
                "final_rank": final_rank,
                "strategy_name": strategy_name,
                "params_key": r.get("params_key", strategy_name),
                "final_score": final_score,
                "median_2y_raw": 0.0,
                "median_2y_adj": 0.0,
                "weeks_in_2y": 0,
                "weeks_traded_2y": 0,
                "coverage_2y": 0.0,
                "cvar5_2y_abs": 0.0,
                "total_trades_period": 0,
                "expiries_with_trades": 0,
                "trade_vs_expiry_ratio": 0.0,
                "expiry_coverage_ratio": 0.0,
                "weight": 0.0,
                "note": "series not found",
            })
            continue

        df_series = daily_map[sid]

        total_trades_period = int(df_series["n_trades"].sum()) if "n_trades" in df_series.columns else int(len(df_series))
        expiries_with_trades = int((df_series["n_trades"] > 0).sum()) if "n_trades" in df_series.columns else int(len(df_series))
        denom = max(1, total_expiries_period)
        trade_vs_expiry_ratio = float(total_trades_period / denom)
        expiry_coverage_ratio = float(expiries_with_trades / denom)

        stats = compute_last2y_stats(df_series, cfg, cal)
        median_2y_raw = float(stats["median_2y_raw"])
        median_2y_adj = float(stats["median_2y_adj"])
        weeks_in_2y = int(stats["weeks_in_2y"])
        weeks_traded_2y = int(stats["weeks_traded_2y"])
        coverage_2y = float(stats["coverage_2y"])
        cvar5_2y_abs = float(stats["cvar5_2y_abs"])

        if (median_2y_adj > 0.0) and (cvar5_2y_abs > 0.0):
            weight = math.sqrt(median_2y_adj / cvar5_2y_abs)
        else:
            weight = 0.0

        lot_rows.append({
            "bucket_id": bucket_id,
            "final_rank": final_rank,
            "strategy_name": strategy_name,
            "params_key": r.get("params_key", strategy_name),
            "final_score": final_score,
            "median_2y_raw": median_2y_raw,
            "median_2y_adj": median_2y_adj,
            "weeks_in_2y": weeks_in_2y,
            "weeks_traded_2y": weeks_traded_2y,
            "coverage_2y": coverage_2y,
            "cvar5_2y_abs": cvar5_2y_abs,
            "total_trades_period": total_trades_period,
            "expiries_with_trades": expiries_with_trades,
            "trade_vs_expiry_ratio": trade_vs_expiry_ratio,
            "expiry_coverage_ratio": expiry_coverage_ratio,
            "weight": weight,
            "note": "",
        })

    lots_detail_df = pd.DataFrame(lot_rows)
    if not lots_detail_df.empty:
        lots_detail_df = allocate_lots_float_and_integer(lots_detail_df, TOTAL_LOTS)

    lots_detail_df = lots_detail_df.sort_values(
        ["bucket_id", "final_rank", "final_score"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    export_all_to_excel(cfg, top3_df, lots_detail_df, bucket_outputs, out_xlsx, total_expiries_period)

    print("\nSaved Excel:", out_xlsx)
    if PRINT_TIMINGS:
        print(f"\n[DONE] Total runtime: {time.time() - t_all:,.1f}s")


if __name__ == "__main__":
    main()
