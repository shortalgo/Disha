#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WALK-FORWARD SELECTOR (PER BUCKET) with:

Step 0) Parse metadata from filename (bucket_id, entry_time, SL, prem_threshold, identity=filename)

Step 1) Deduplicate strategies by identical DAILY PnL behavior in BOTH 2024 & 2025
        - sig = hash(pnl_vector_2024 over bucket trading dates, pnl_vector_2025 over bucket trading dates)
        - keep: lowest SL, then higher prem_threshold, then lexicographically smallest filename

Step 2) Folds: 9M IS / 3M OOS, step 3M (calendar months), within 2024-2025

Step 3) IS scoring per fold using z-scores across ALL strategies in bucket+fold
        - metrics: total_pnl, win_rate, var55_abs, max_drawdown_abs
        - weights: 0.30 / 0.20 / 0.30 / 0.20  (PnL / Win / Risk / MDD)
        - select Top 100 IS

Step 4) OOS evaluation on Top100; compute OOS score via same weighted z-score (computed over candidates)
        - pick Top-K OOS performers per fold (K=20 default)

Step 5) Consistency filter: must appear in Top-K OOS at least 3 folds

Step 6) Final scoring on LAST 2 YEARS slice (ending at dataset end), with recency weighting 0.6/0.4:
        - last 12 months weight=0.60, previous 12 months weight=0.40
        - apply weights to pnl series, compute weighted metrics, z-score across remaining strategies

Step 7) Enforce "no same entry time" within bucket:
        - greedy by final_score, skip same entry_time, stop at TOP_N_FINAL_PER_BUCKET

Exports an Excel workbook per run with per-bucket sheets and summaries.

IMPORTANT CHANGE: "EXHAUSTIVE PARSING EVERYWHERE"
- Robust CSV reading (tries multiple encodings)
- Robust column matching (case/space tolerant)
- Robust datetime parsing (multi-format pass + safe fallback, no dayfirst warning spam)
- ALL boolean filtering uses a strict numpy bool mask to avoid pandas NA-bool crashes.
"""

import os
import glob
import re
import time
import hashlib
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd

# Excel export
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


# ============================================================
# CONFIG
# ============================================================
TOP_N_FINAL_PER_BUCKET = 3
TOP_N_IS_PER_FOLD = 100
TOP_K_OOS_PER_FOLD = 100  # DEFAULT

BUCKET_DIRS: Dict[str, str] = {
    "B1":  r"D:\Downloads\SENSEX_PNL\PNL\DTE-1.0",
    "B2":  r"D:\Downloads\SENSEX_PNL\PNL\DTE-1.5",
    "B3":  r"D:\Downloads\SENSEX_PNL\PNL\DTE-2.0",
    "B4":  r"D:\Downloads\SENSEX_PNL\PNL\DTE-2.5",
    "B5":  r"D:\Downloads\SENSEX_PNL\PNL\DTE-3.0",
    "B6":  r"D:\Downloads\SENSEX_PNL\PNL\DTE-3.5",
    "B7":  r"D:\Downloads\SENSEX_PNL\PNL\DTE-4.0",
    "B8":  r"D:\Downloads\SENSEX_PNL\PNL\DTE-4.5",
}

OUTPUT_XLSX: Optional[str] = None

DATA_START = pd.Timestamp("2024-01-01").normalize()
DATA_END   = pd.Timestamp("2025-12-31").normalize()

PRINT_EVERY_FILES = 200
PRINT_EVERY_FOLDS = 1
PRINT_TIMINGS = True
SHOW_EXCEPTIONS_SAMPLE = 3

EPS = 1e-12


# ============================================================
# WEIGHTS (30/20/30/20)
# ============================================================
W_PNL  = 0.30
W_WIN  = 0.20
W_RISK = 0.30
W_MDD  = 0.20


# ============================================================
# Step 0 — Metadata parsing from filename
# ============================================================
_RE_TIME = re.compile(r"(?:^|[_-])Time[-_]?(\d{1,2})[-_](\d{2})(?:[_-]|$)", re.IGNORECASE)
_RE_SL   = re.compile(r"(?:^|[_-])SL[-_]?(\d+(?:\.\d+)?)(?:[_-]|$)", re.IGNORECASE)
_RE_PREM = re.compile(r"(?:^|[_-])prem[-_]?(\d+(?:\.\d+)?)(?:[_-]|$)", re.IGNORECASE)

@dataclass(frozen=True)
class StrategyMeta:
    bucket_id: str
    filename: str          # identity = exact filename (without extension)
    entry_time: str        # "HH:MM" or "NA"
    sl: float              # +inf if missing
    prem_threshold: float  # -inf if missing


def parse_strategy_meta_from_path(path: str, bucket_id: str) -> StrategyMeta:
    base = os.path.basename(path)
    name = os.path.splitext(base)[0]  # exact identity after filtering

    m_time = _RE_TIME.search(name)
    if m_time:
        hh = int(m_time.group(1))
        mm = int(m_time.group(2))
        entry_time = f"{hh:02d}:{mm:02d}"
    else:
        entry_time = "NA"

    m_sl = _RE_SL.search(name)
    sl = float(m_sl.group(1)) if m_sl else float("inf")

    m_prem = _RE_PREM.search(name)
    prem = float(m_prem.group(1)) if m_prem else float("-inf")

    return StrategyMeta(
        bucket_id=bucket_id,
        filename=name,
        entry_time=entry_time,
        sl=sl,
        prem_threshold=prem,
    )

# ============================================================
# PnL Adjustment (trade-frequency normalization)
# ============================================================
USE_ADJUSTED_PNL = True  # <-- turn ON/OFF easily

def adjust_pnl_by_trade_ratio(arr: np.ndarray) -> tuple[np.ndarray, float, int, int]:
    """
    Adjust pnl by trade frequency:
      factor = (#days with pnl != 0) / (#days in window)
      adjusted_arr = arr * factor

    Returns: (adjusted_arr, factor, n_traded, n_days)
    """
    arr = np.asarray(arr, dtype=np.float64)
    n_days = int(arr.size)
    if n_days <= 0:
        return arr, 0.0, 0, 0

    traded = arr != 0.0
    n_traded = int(traded.sum())
    factor = (n_traded / n_days) if n_days > 0 else 0.0
    return (arr * factor), float(factor), n_traded, n_days


def trade_ratio_from_weights(traded_mask: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted trade frequency:
      factor = sum(weights on traded days) / sum(weights on all days)
    Useful for Step-6 where you apply 0.6/0.4 weights.
    """
    traded_mask = np.asarray(traded_mask, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    denom = float(w.sum())
    if denom <= 0:
        return 0.0
    return float(w[traded_mask].sum() / denom)

# ============================================================
# Robust CSV reading + robust column matching
# ============================================================
_NULL_TOKENS = {"", "nan", "NaN", "NONE", "None", "null", "NULL"}

def discover_csvs(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.csv")))

def _canonical_col(s: str) -> str:
    # lowercase, strip, collapse spaces, remove non-alnum except space
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return s

def _find_col(df: pd.DataFrame, wanted: List[str]) -> Optional[str]:
    """
    wanted: list of acceptable names (canonical comparison)
    returns actual df column name if match
    """
    if df is None or df.empty:
        return None
    canon_to_real = {_canonical_col(c): c for c in df.columns}
    for w in wanted:
        w_can = _canonical_col(w)
        if w_can in canon_to_real:
            return canon_to_real[w_can]
    return None

def read_csv_robust(path: str) -> pd.DataFrame:
    """
    Tries common encodings. Keeps it simple (no separator sniffing unless needed).
    """
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1", "utf-16"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, engine="python")
        except Exception as e:
            last_err = e
    # final attempt default
    try:
        return pd.read_csv(path, engine="python")
    except Exception as e:
        raise RuntimeError(f"Could not read CSV: {os.path.basename(path)} | last_err={repr(last_err)} | final_err={repr(e)}")


# ============================================================
# Exhaustive datetime parsing (vectorized, warning-safe)
# ============================================================
# Formats we see commonly in tradesheets
_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",

    "%d-%m-%Y %H:%M:%S",
    "%d-%m-%Y %H:%M",
    "%d-%m-%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",

    "%d-%m-%y %H:%M:%S",
    "%d-%m-%y %H:%M",
    "%d-%m-%y",
    "%d/%m/%y %H:%M:%S",
    "%d/%m/%y %H:%M",
    "%d/%m/%y",

    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%Y",
]

def parse_datetime_exhaustive(series: pd.Series, *, dayfirst_default: bool = True) -> pd.Series:
    """
    Exhaustive parse without dayfirst warnings:
    1) Handle numeric epoch seconds/ms if present
    2) Try a sequence of explicit formats (fast, no warning)
    3) Fallback to pandas to_datetime with dayfirst_default, warning-suppressed
    """
    if series is None:
        return pd.Series([], dtype="datetime64[ns]")

    s = series.astype(str).str.strip()
    s = s.replace(list(_NULL_TOKENS), np.nan)
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

    # 1) epoch handling (pure digits)
    is_num = s.notna() & s.str.fullmatch(r"\d+")
    if is_num.any():
        nums = pd.to_numeric(s[is_num], errors="coerce")
        # decide ms vs sec by magnitude
        # sec ~ 1e9..2e9, ms ~ 1e12..2e12
        ms_mask = nums >= 10**11
        sec_mask = ~ms_mask
        if ms_mask.any():
            out.loc[nums.index[ms_mask]] = pd.to_datetime(nums[ms_mask], errors="coerce", unit="ms")
        if sec_mask.any():
            out.loc[nums.index[sec_mask]] = pd.to_datetime(nums[sec_mask], errors="coerce", unit="s")

    # 2) explicit format passes
    rem = out.isna() & s.notna()
    if rem.any():
        sr = s[rem]
        for fmt in _DT_FORMATS:
            if not rem.any():
                break
            parsed = pd.to_datetime(sr, errors="coerce", format=fmt)
            ok = parsed.notna()
            if ok.any():
                out.loc[parsed.index[ok]] = parsed.loc[ok].values
                rem = out.isna() & s.notna()
                sr = s[rem]

    # 3) fallback inference (warning suppressed)
    rem = out.isna() & s.notna()
    if rem.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            parsed = pd.to_datetime(s[rem], errors="coerce", dayfirst=dayfirst_default)
        out.loc[parsed.index] = parsed.values

    return out


# ============================================================
# CSV loading (daily PnL series)
# ============================================================
def load_daily_pnl_series_csv(path: str) -> pd.DataFrame:
    """
    Returns df with columns: date (normalized), daily_pnl (sum per day)

    Supports:
      - (Entry Time + PnL With Spread)
      - (Date + PnL)

    Robust:
      - column names are matched case/space/punct tolerant
      - datetime parsing is exhaustive
      - boolean filtering uses strict numpy bool array (no NA-bool crash)
    """
    df = read_csv_robust(path)

    col_entry_time = _find_col(df, ["Entry Time", "EntryTime", "entry time", "entrytime"])
    col_pnl_spread = _find_col(df, ["PnL With Spread", "PNL With Spread", "pnl with spread", "pnl_with_spread"])

    col_date = _find_col(df, ["Date", "date", "Trade Date", "trade date", "Trading Date", "trading date"])
    col_pnl = _find_col(df, ["PnL", "PNL", "pnl"])

    if col_entry_time and col_pnl_spread:
        dt = parse_datetime_exhaustive(df[col_entry_time], dayfirst_default=True)
        pnl = pd.to_numeric(df[col_pnl_spread], errors="coerce").fillna(0.0)
    elif col_date and col_pnl:
        dt = parse_datetime_exhaustive(df[col_date], dayfirst_default=True)
        pnl = pd.to_numeric(df[col_pnl], errors="coerce").fillna(0.0)
    else:
        raise ValueError(
            f"Missing required columns in {os.path.basename(path)} | "
            f"found_cols={list(df.columns)[:30]}"
        )

    out = pd.DataFrame({"date": dt, "pnl_row": pnl}).dropna(subset=["date"])
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    # Restrict to 2024-2025 dataset window (SAFE mask)
    mask = (out["date"].notna()) & (out["date"] >= DATA_START) & (out["date"] <= DATA_END)
    mask = mask.to_numpy(dtype=bool, na_value=False)
    out = out.loc[mask].copy()

    if out.empty:
        return pd.DataFrame(columns=["date", "daily_pnl"])

    g = out.groupby("date", as_index=False)["pnl_row"].sum()
    g = g.rename(columns={"pnl_row": "daily_pnl"}).sort_values("date").reset_index(drop=True)
    return g


# ============================================================
# Step 1 — Deduplicate by identical 2024 + 2025 daily vectors
# ============================================================
def _hash_two_year_vectors(v2024: np.ndarray, v2025: np.ndarray) -> str:
    a = np.round(v2024.astype(np.float64), 6)
    b = np.round(v2025.astype(np.float64), 6)
    h = hashlib.blake2b(digest_size=16)
    h.update(a.tobytes())
    h.update(b.tobytes())
    return h.hexdigest()


def dedupe_strategies_by_behavior(
    metas: List[StrategyMeta],
    series_map: Dict[str, pd.DataFrame],
    bucket_dates_2024: pd.DatetimeIndex,
    bucket_dates_2025: pd.DatetimeIndex,
) -> Tuple[List[StrategyMeta], pd.DataFrame]:
    rows = []
    sig_to_group: Dict[str, List[StrategyMeta]] = {}

    for m in metas:
        df = series_map.get(m.filename)
        if df is None or df.empty:
            s = pd.Series([], dtype=float)
        else:
            s = pd.Series(df["daily_pnl"].values, index=pd.to_datetime(df["date"], errors="coerce"))

        v24 = s.reindex(bucket_dates_2024, fill_value=0.0).values.astype(np.float64)
        v25 = s.reindex(bucket_dates_2025, fill_value=0.0).values.astype(np.float64)

        # keep dedupe signature on RAW daily behavior
        sig = _hash_two_year_vectors(v24, v25)



        rows.append({
            "filename": m.filename,
            "entry_time": m.entry_time,
            "sl": m.sl,
            "prem_threshold": m.prem_threshold,
            "sig": sig,
        })
        sig_to_group.setdefault(sig, []).append(m)

    kept: List[StrategyMeta] = []
    drop_set = set()

    for sig, group in sig_to_group.items():
        if len(group) == 1:
            kept.append(group[0])
            continue

        group_sorted = sorted(
            group,
            key=lambda x: (
                x.sl,                 # lowest SL
                -x.prem_threshold,    # higher prem
                x.filename.lower(),   # lexicographically smallest
            )
        )
        winner = group_sorted[0]
        kept.append(winner)
        for loser in group_sorted[1:]:
            drop_set.add(loser.filename)

    rep = pd.DataFrame(rows)
    rep["is_kept"] = ~rep["filename"].isin(drop_set)
    rep["group_size"] = rep.groupby("sig")["sig"].transform("size")
    rep = rep.sort_values(["group_size", "sig", "is_kept"], ascending=[False, True, False]).reset_index(drop=True)
    return kept, rep


# ============================================================
# Step 2 — Build folds (9/3 rolling by 3 months)
# ============================================================
@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_month_folds_9_3_step3(start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[Fold]:
    start_m = start_date.to_period("M")
    end_m = end_date.to_period("M")
    months = pd.period_range(start=start_m, end=end_m, freq="M")

    folds: List[Fold] = []
    i = 0
    while True:
        train_months = months[i:i+9]
        test_months = months[i+9:i+12]
        if len(train_months) < 9 or len(test_months) < 3:
            break

        train_start = train_months[0].to_timestamp(how="start").normalize()
        train_end = train_months[-1].to_timestamp(how="end").normalize()

        test_start = (train_end + pd.Timedelta(days=1)).normalize()
        test_end = test_months[-1].to_timestamp(how="end").normalize()

        folds.append(Fold(
            fold_id=len(folds) + 1,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
        i += 3

    return folds


# ============================================================
# Metrics, z-scores, scoring
# ============================================================
def max_drawdown_abs(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(max(0.0, -dd.min(initial=0.0)))


def var55_abs(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    q = np.quantile(pnl, 0.45)
    return float(max(0.0, -q))


def win_rate(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    traded = pnl != 0.0
    n = int(traded.sum())
    if n == 0:
        return 0.0
    wins = int((pnl[traded] > 0.0).sum())
    return float(wins / n)


def compute_metrics_for_range(
    s: pd.Series,
    date_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> Dict[str, float]:
    # SAFE mask
    mask = (date_index >= start) & (date_index <= end)
    mask = mask.to_numpy(dtype=bool, na_value=False) if hasattr(mask, "to_numpy") else np.asarray(mask, dtype=bool)
    idx = date_index[mask]

    arr = s.reindex(idx, fill_value=0.0).values.astype(np.float64)

    # --- Adjust PnL by trade ratio (global scaling) ---
    adj_factor = 1.0
    if USE_ADJUSTED_PNL:
        arr, adj_factor, n_traded, n_days = adjust_pnl_by_trade_ratio(arr)
    else:
        n_days = int(len(arr))
        n_traded = int((arr != 0.0).sum())

    return {
        "total_pnl": float(arr.sum()),
        "win_rate": float(win_rate(arr)),
        "var55_abs": float(var55_abs(arr)),
        "max_drawdown_abs": float(max_drawdown_abs(arr)),
        "n_days": int(n_days),
        "n_traded_days": int(n_traded),
        "pnl_adj_factor": float(adj_factor),   # <-- added (useful to audit)
    }


    # return {
    #     "total_pnl": float(arr.sum()),
    #     "win_rate": float(win_rate(arr)),
    #     "var55_abs": float(var55_abs(arr)),
    #     "max_drawdown_abs": float(max_drawdown_abs(arr)),
    #     "n_days": int(len(idx)),
    #     "n_traded_days": int((arr != 0.0).sum()),
    # }


def zscore_series(x: pd.Series) -> pd.Series:
    mu = float(x.mean())
    sd = float(x.std(ddof=0))
    if not np.isfinite(sd) or sd < EPS:
        return pd.Series([0.0] * len(x), index=x.index)
    return (x - mu) / sd


def score_with_z(
    df: pd.DataFrame,
    weights=(W_PNL, W_WIN, W_RISK, W_MDD),
) -> pd.DataFrame:
    out = df.copy()

    out["z_pnl"] = zscore_series(out["total_pnl"])
    out["z_win"] = zscore_series(out["win_rate"])

    out["z_risk"] = zscore_series(-out["var55_abs"])
    out["z_mdd"]  = zscore_series(-out["max_drawdown_abs"])

    w_pnl, w_win, w_risk, w_mdd = weights
    out["score"] = (
        w_pnl * out["z_pnl"] +
        w_win * out["z_win"] +
        w_risk * out["z_risk"] +
        w_mdd * out["z_mdd"]
    )
    return out


# ============================================================
# Weighted metrics for Step 6 (0.6/0.4 on last 12m vs prev 12m)
# ============================================================
def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    w = np.clip(weights.astype(np.float64), 0.0, None)
    if w.sum() <= 0:
        return float(np.quantile(values, q))
    order = np.argsort(values)
    v = values[order]
    w = w[order]
    cdf = np.cumsum(w) / w.sum()
    return float(v[np.searchsorted(cdf, q, side="left")])


def compute_weighted_last2y_metrics(
    s: pd.Series,
    date_index: pd.DatetimeIndex,
    end_date: pd.Timestamp,
    half_weights=(0.60, 0.40),
) -> Dict[str, float]:
    a, b = half_weights
    end_date = end_date.normalize()
    start_2y = (end_date - pd.DateOffset(years=2) + pd.Timedelta(days=1)).normalize()
    split_1y = (end_date - pd.DateOffset(years=1) + pd.Timedelta(days=1)).normalize()

    mask = (date_index >= start_2y) & (date_index <= end_date)
    mask = mask.to_numpy(dtype=bool, na_value=False) if hasattr(mask, "to_numpy") else np.asarray(mask, dtype=bool)
    idx = date_index[mask]

    raw = s.reindex(idx, fill_value=0.0).values.astype(np.float64)

    if len(idx) == 0:
        return {
            "w_total_pnl": 0.0,
            "w_win_rate": 0.0,
            "w_var55_abs": 0.0,
            "w_max_drawdown_abs": 0.0,
            "n_days_2y": 0,
            "n_traded_days_2y": 0,
            "pnl_adj_factor_2y": 0.0,
        }

    # weights 0.6/0.4 within last 2y
    w = np.full(len(idx), b, dtype=np.float64)
    mask_1y = (idx >= split_1y)
    mask_1y = mask_1y.to_numpy(dtype=bool, na_value=False) if hasattr(mask_1y, "to_numpy") else np.asarray(mask_1y, dtype=bool)
    w[mask_1y] = a

    traded_mask = (raw != 0.0)
    n_traded = int(traded_mask.sum())

    # --- Adjust PnL by weighted trade ratio (global scaling) ---
    adj_factor = 1.0
    if USE_ADJUSTED_PNL:
        adj_factor = trade_ratio_from_weights(traded_mask, w)
    arr = raw * adj_factor  # adjusted pnl series

    # weighted total pnl
    w_total_pnl = float((arr * w).sum())

    # weighted win rate (use traded_mask from RAW, not from adjusted)
    w_traded = w[traded_mask]
    if w_traded.sum() > 0:
        w_wins = float((w[traded_mask] * (arr[traded_mask] > 0.0)).sum())
        w_win_rate = float(w_wins / w_traded.sum())
    else:
        w_win_rate = 0.0

    # weighted VaR55 on adjusted series
    q = weighted_quantile(arr, w, 0.45)
    w_var55_abs = float(max(0.0, -q))

    # weighted MDD on adjusted & weighted path
    w_mdd = float(max_drawdown_abs(arr * w))

    return {
        "w_total_pnl": w_total_pnl,
        "w_win_rate": w_win_rate,
        "w_var55_abs": w_var55_abs,
        "w_max_drawdown_abs": w_mdd,
        "n_days_2y": int(len(idx)),
        "n_traded_days_2y": int(n_traded),
        "pnl_adj_factor_2y": float(adj_factor),
    }


def score_with_z_final(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["z_pnl"] = zscore_series(out["w_total_pnl"])
    out["z_win"] = zscore_series(out["w_win_rate"])
    out["z_risk"] = zscore_series(-out["w_var55_abs"])
    out["z_mdd"]  = zscore_series(-out["w_max_drawdown_abs"])
    out["final_score"] = (
        W_PNL * out["z_pnl"] +
        W_WIN * out["z_win"] +
        W_RISK * out["z_risk"] +
        W_MDD * out["z_mdd"]
    )
    return out


# ============================================================
# Excel helpers
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
        ws.column_dimensions[ws.cell(row=1, column=col).column_letter].width = 20


def _write_df(ws, df: pd.DataFrame, table_name: str):
    df = _sanitize_for_excel(df)
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)
    if ws.max_row >= 2 and ws.max_column >= 1:
        _style_sheet_as_table(ws, table_name, ws.max_row, ws.max_column)


# ============================================================
# Bucket runner (implements Steps 0-7)
# ============================================================
def run_bucket(bucket_id: str, folder: str) -> Dict[str, Any]:
    t0 = time.time()
    paths = discover_csvs(folder)
    if not paths:
        raise FileNotFoundError(f"[{bucket_id}] No CSV files found in: {folder}")

    print(f"\n[{bucket_id}] Found {len(paths)} CSV files")

    metas: List[StrategyMeta] = []
    series_map: Dict[str, pd.DataFrame] = {}
    parse_errors = 0
    shown_errors = 0

    dates_2024 = set()
    dates_2025 = set()

    for i, p in enumerate(paths, start=1):
        meta = parse_strategy_meta_from_path(p, bucket_id=bucket_id)
        try:
            df_daily = load_daily_pnl_series_csv(p)
        except Exception as e:
            parse_errors += 1
            if shown_errors < SHOW_EXCEPTIONS_SAMPLE:
                shown_errors += 1
                print(f"[{bucket_id}] !! ERROR reading {os.path.basename(p)}: {repr(e)}")
            continue

        metas.append(meta)
        series_map[meta.filename] = df_daily

        if not df_daily.empty:
            dts = pd.to_datetime(df_daily["date"], errors="coerce").dt.normalize()
            d24 = dts[dts.dt.year == 2024]
            d25 = dts[dts.dt.year == 2025]
            dates_2024.update(d24.tolist())
            dates_2025.update(d25.tolist())

        if (i % PRINT_EVERY_FILES) == 0 or i == 1 or i == len(paths):
            elapsed = time.time() - t0
            msg = (f"[{bucket_id}] Loaded {i}/{len(paths)} | ok={len(metas)} | errors={parse_errors}")
            if PRINT_TIMINGS:
                msg += f" | elapsed={elapsed:,.1f}s"
            print(msg)

    if len(metas) == 0:
        raise ValueError(f"[{bucket_id}] No strategies parsed successfully. errors={parse_errors}")

    bucket_dates_2024 = pd.DatetimeIndex(sorted(dates_2024)) if len(dates_2024) else pd.DatetimeIndex([])
    bucket_dates_2025 = pd.DatetimeIndex(sorted(dates_2025)) if len(dates_2025) else pd.DatetimeIndex([])

    # Step 1: Deduplicate
    kept_metas, dedupe_report_df = dedupe_strategies_by_behavior(
        metas=metas,
        series_map=series_map,
        bucket_dates_2024=bucket_dates_2024,
        bucket_dates_2025=bucket_dates_2025,
    )

    print(f"[{bucket_id}] Dedup: before={len(metas)} after={len(kept_metas)} (dropped={len(metas)-len(kept_metas)})")

    # Build bucket master date index (union)
    bucket_dates_all = pd.DatetimeIndex(sorted(set(bucket_dates_2024.tolist()) | set(bucket_dates_2025.tolist())))
    if len(bucket_dates_all) == 0:
        u = set()
        for m in kept_metas:
            df = series_map.get(m.filename)
            if df is not None and not df.empty:
                u.update(pd.to_datetime(df["date"], errors="coerce").dt.normalize().tolist())
        bucket_dates_all = pd.DatetimeIndex(sorted(u))

    # Strategy series
    strat_series: Dict[str, pd.Series] = {}
    for m in kept_metas:
        df = series_map.get(m.filename)
        if df is None or df.empty:
            strat_series[m.filename] = pd.Series([], dtype=float)
        else:
            strat_series[m.filename] = pd.Series(
                df["daily_pnl"].values.astype(np.float64),
                index=pd.to_datetime(df["date"], errors="coerce").dt.normalize()
            )

    # Step 2: Folds
    folds = build_month_folds_9_3_step3(DATA_START, DATA_END)
    folds_df = pd.DataFrame([{
        "fold_id": f.fold_id,
        "train_start": f.train_start,
        "train_end": f.train_end,
        "test_start": f.test_start,
        "test_end": f.test_end,
    } for f in folds])

    print(f"[{bucket_id}] Folds built: {len(folds)} (9M/3M step 3M)")

    # Step 3-4: Walk-forward
    is_top100_rows = []
    oos_topk_rows = []
    oos_topk_count: Dict[str, int] = {m.filename: 0 for m in kept_metas}

    for fi, f in enumerate(folds, start=1):
        if (fi % PRINT_EVERY_FOLDS) == 0:
            print(f"[{bucket_id}] Fold {fi}/{len(folds)} "
                  f"IS {f.train_start.date()}..{f.train_end.date()} | "
                  f"OOS {f.test_start.date()}..{f.test_end.date()}")

        # IS metrics for all
        fold_rows = []
        for m in kept_metas:
            s = strat_series[m.filename]
            met = compute_metrics_for_range(s, bucket_dates_all, f.train_start, f.train_end)
            fold_rows.append({
                "bucket_id": bucket_id,
                "fold_id": f.fold_id,
                "filename": m.filename,
                "entry_time": m.entry_time,
                "sl": m.sl,
                "prem_threshold": m.prem_threshold,
                **met
            })

        is_df = pd.DataFrame(fold_rows)
        is_scored = score_with_z(is_df)
        is_scored = is_scored.sort_values("score", ascending=False).reset_index(drop=True)
        is_scored["is_rank"] = np.arange(1, len(is_scored) + 1)

        top100 = is_scored.head(min(TOP_N_IS_PER_FOLD, len(is_scored))).copy()
        top100["selected_is_rank"] = np.arange(1, len(top100) + 1)
        is_top100_rows.extend(top100.to_dict("records"))

        # OOS only Top100
        oos_rows = []
        for _, r in top100.iterrows():
            fn = r["filename"]
            s = strat_series[fn]
            met_oos = compute_metrics_for_range(s, bucket_dates_all, f.test_start, f.test_end)
            oos_rows.append({
                "bucket_id": bucket_id,
                "fold_id": f.fold_id,
                "filename": fn,
                "entry_time": r["entry_time"],
                "sl": r["sl"],
                "prem_threshold": r["prem_threshold"],
                "selected_is_rank": int(r["selected_is_rank"]),
                "oos_total_pnl": met_oos["total_pnl"],
                "oos_win_rate": met_oos["win_rate"],
                "oos_var55_abs": met_oos["var55_abs"],
                "oos_max_drawdown_abs": met_oos["max_drawdown_abs"],
                "oos_n_days": met_oos["n_days"],
                "oos_n_traded_days": met_oos["n_traded_days"],
            })

        oos_df = pd.DataFrame(oos_rows)
        if not oos_df.empty:
            tmp = oos_df.rename(columns={
                "oos_total_pnl": "total_pnl",
                "oos_win_rate": "win_rate",
                "oos_var55_abs": "var55_abs",
                "oos_max_drawdown_abs": "max_drawdown_abs",
            })
            tmp_scored = score_with_z(tmp)

            oos_df["oos_score"] = tmp_scored["score"].values
            oos_df = oos_df.sort_values("oos_score", ascending=False).reset_index(drop=True)
            oos_df["oos_rank"] = np.arange(1, len(oos_df) + 1)

            topk = oos_df.head(min(TOP_K_OOS_PER_FOLD, len(oos_df))).copy()
            topk["is_top100_flag"] = True
            oos_topk_rows.extend(topk.to_dict("records"))

            for fn in topk["filename"].tolist():
                oos_topk_count[fn] = oos_topk_count.get(fn, 0) + 1

    is_top100_df = pd.DataFrame(is_top100_rows)
    oos_topk_df = pd.DataFrame(oos_topk_rows)

    # Step 5: Consistency filter
    cons_rows = []
    for m in kept_metas:
        cons_rows.append({
            "bucket_id": bucket_id,
            "filename": m.filename,
            "entry_time": m.entry_time,
            "sl": m.sl,
            "prem_threshold": m.prem_threshold,
            "oos_topk_count": int(oos_topk_count.get(m.filename, 0)),
        })
    consistency_df = pd.DataFrame(cons_rows)
    consistent = consistency_df[consistency_df["oos_topk_count"] >= 3].copy()
    print(f"[{bucket_id}] Consistency: kept {len(consistent)}/{len(kept_metas)} (need oos_topk_count>=3)")

    consistent_set = set(consistent["filename"].tolist())

    # Step 6: Final scoring on last 2 years with 0.6/0.4 recency
    end_date = pd.Timestamp(bucket_dates_all.max()).normalize() if len(bucket_dates_all) else DATA_END

    final_rows = []
    for m in kept_metas:
        if m.filename not in consistent_set:
            continue
        s = strat_series[m.filename]
        met = compute_weighted_last2y_metrics(
            s=s,
            date_index=bucket_dates_all,
            end_date=end_date,
            half_weights=(0.60, 0.40),
        )
        final_rows.append({
            "bucket_id": bucket_id,
            "filename": m.filename,
            "entry_time": m.entry_time,
            "sl": m.sl,
            "prem_threshold": m.prem_threshold,
            "oos_topk_count": int(oos_topk_count.get(m.filename, 0)),
            **met
        })

    final_metrics_df = pd.DataFrame(final_rows)
    if final_metrics_df.empty:
        final_scored_df = pd.DataFrame()
        chosen_df = pd.DataFrame()
        print(f"[{bucket_id}] Final scoring: EMPTY after consistency filter.")
    else:
        final_scored_df = score_with_z_final(final_metrics_df)
        final_scored_df = final_scored_df.sort_values("final_score", ascending=False).reset_index(drop=True)
        final_scored_df["final_rank"] = np.arange(1, len(final_scored_df) + 1)

        # Step 7: No same entry_time (greedy)
        picked = []
        used_times = set()
        for _, r in final_scored_df.iterrows():
            et = str(r.get("entry_time", "NA"))
            if et in used_times:
                continue
            picked.append(r.to_dict())
            used_times.add(et)
            if len(picked) >= TOP_N_FINAL_PER_BUCKET:
                break

        chosen_df = pd.DataFrame(picked)
        print(f"[{bucket_id}] Chosen (no same entry_time): {len(chosen_df)}/{TOP_N_FINAL_PER_BUCKET}")

    if PRINT_TIMINGS:
        print(f"[{bucket_id}] Completed bucket in {time.time()-t0:,.1f}s | parse_errors={parse_errors}")

    return {
        "dedupe_report_df": dedupe_report_df,
        "folds_df": folds_df,
        "is_top100_df": is_top100_df,
        "oos_topk_df": oos_topk_df,
        "consistency_df": consistency_df.sort_values("oos_topk_count", ascending=False).reset_index(drop=True),
        "final_scored_df": final_scored_df,
        "chosen_df": chosen_df,
        "bucket_dates_2024_n": int(len(bucket_dates_2024)),
        "bucket_dates_2025_n": int(len(bucket_dates_2025)),
        "kept_after_dedup": int(len(kept_metas)),
        "total_parsed": int(len(metas)),
        "parse_errors": int(parse_errors),
    }


# ============================================================
# Excel export (one workbook for all buckets)
# ============================================================
def export_to_excel(all_bucket_outputs: Dict[str, Dict[str, Any]], out_path: str):
    t0 = time.time()
    print(f"\n[EXCEL] Writing workbook: {out_path}")

    wb = Workbook()
    wb.remove(wb.active)

    # Config sheet
    ws = wb.create_sheet("Config")
    ws.append(["Key", "Value"])
    rows = [
        ("DATA_START", str(DATA_START.date())),
        ("DATA_END", str(DATA_END.date())),
        ("Folds", "IS=9 months, OOS=3 months, step=3 months"),
        ("IS Selection", f"Top {TOP_N_IS_PER_FOLD} by IS z-score score"),
        ("OOS TopK per fold", TOP_K_OOS_PER_FOLD),
        ("Consistency", "oos_topk_count >= 3"),
        ("Final recency", "0.60 last 12 months, 0.40 previous 12 months (within last 2 years)"),
        ("Final selection", f"Greedy no-same-entry_time until {TOP_N_FINAL_PER_BUCKET}"),
        ("Weights pnl/win/risk/mdd", f"{W_PNL}/{W_WIN}/{W_RISK}/{W_MDD}"),
        ("Risk metric", "var55_abs (55% confidence => 45th percentile), lower is better"),
        ("MDD metric", "max_drawdown_abs on (weighted) cumulative pnl, lower is better"),
        ("Parsing", "Exhaustive datetime parsing + robust CSV encoding + safe bool masks"),
    ]
    for k, v in rows:
        ws.append([k, v])
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 120
    ws["A1"].font = Font(bold=True)
    ws["B1"].font = Font(bold=True)

    # Bucket summary
    summ_rows = []
    for bid, out in all_bucket_outputs.items():
        summ_rows.append({
            "bucket_id": bid,
            "total_parsed": out.get("total_parsed", 0),
            "parse_errors": out.get("parse_errors", 0),
            "kept_after_dedup": out.get("kept_after_dedup", 0),
            "bucket_dates_2024_n": out.get("bucket_dates_2024_n", 0),
            "bucket_dates_2025_n": out.get("bucket_dates_2025_n", 0),
            "chosen_n": 0 if out.get("chosen_df") is None else int(len(out.get("chosen_df"))),
        })
    ws_sum = wb.create_sheet("Bucket_Summary")
    _write_df(ws_sum, pd.DataFrame(summ_rows), "tblBucketSummary")

    # Per-bucket sheets
    for bid, out in all_bucket_outputs.items():
        prefix = _safe_sheet_name(bid)

        _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_Dedupe")),
                  out.get("dedupe_report_df", pd.DataFrame()),
                  f"t{prefix}Dedupe")

        _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_Folds")),
                  out.get("folds_df", pd.DataFrame()),
                  f"t{prefix}Folds")

        _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_IS_Top100")),
                  out.get("is_top100_df", pd.DataFrame()),
                  f"t{prefix}ISTop100")

        _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_OOS_TopK")),
                  out.get("oos_topk_df", pd.DataFrame()),
                  f"t{prefix}OOSTopK")

        _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_Consistency")),
                  out.get("consistency_df", pd.DataFrame()),
                  f"t{prefix}Cons")

        _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_FinalScores")),
                  out.get("final_scored_df", pd.DataFrame()),
                  f"t{prefix}Final")

        _write_df(wb.create_sheet(_safe_sheet_name(f"{prefix}_Chosen")),
                  out.get("chosen_df", pd.DataFrame()),
                  f"t{prefix}Chosen")

    wb.save(out_path)
    if PRINT_TIMINGS:
        print(f"[EXCEL] Saved workbook in {time.time() - t0:,.1f}s")


# ============================================================
# MAIN
# ============================================================
def main():
    print(f"[SANITY] Buckets={len(BUCKET_DIRS)} | IS TopN={TOP_N_IS_PER_FOLD} | OOS TopK={TOP_K_OOS_PER_FOLD} | Final pick={TOP_N_FINAL_PER_BUCKET}")
    print(f"[SANITY] Data window: {DATA_START.date()} .. {DATA_END.date()}")

    first_bucket = next(iter(BUCKET_DIRS.values()))
    out_xlsx = OUTPUT_XLSX or os.path.join(first_bucket, "walkforward_dedup_zscore_9_3_recency_0p6_0p4_NEW.xlsx")

    all_bucket_outputs: Dict[str, Dict[str, Any]] = {}
    t_all = time.time()

    for bi, (bucket_id, folder) in enumerate(BUCKET_DIRS.items(), start=1):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"[{bucket_id}] Bucket folder not found: {folder}")

        print(f"\n=== BUCKET {bi}/{len(BUCKET_DIRS)}: {bucket_id} ===")
        print(f"[{bucket_id}] Path: {folder}")

        out = run_bucket(bucket_id=bucket_id, folder=folder)
        all_bucket_outputs[bucket_id] = out

        chosen = out.get("chosen_df", pd.DataFrame())
        if chosen is None or chosen.empty:
            print(f"[{bucket_id}] Chosen: NONE")
        else:
            top = chosen.iloc[0]
            print(f"[{bucket_id}] Chosen top1: {top.get('filename','')} | entry_time={top.get('entry_time','')} | final_score={float(top.get('final_score',0.0)):.4f}")

    export_to_excel(all_bucket_outputs, out_xlsx)

    print("\nSaved Excel:", out_xlsx)
    if PRINT_TIMINGS:
        print(f"\n[DONE] Total runtime: {time.time() - t_all:,.1f}s")


if __name__ == "__main__":
    main()
