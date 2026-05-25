#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tradesheet Sanity Checker — multiprocessing (CSV/XLSX) + Rule 9 (CE & PE same exit time)

Premium-scoped rules:
- 0/Negative/NaN checks ONLY on CE_OTM_Premium & PE_OTM_Premium.
- Negative premiums are EXPECTED for Action == 'Long' (not flagged).
- Premium tolerance is applied ONLY to Action == 'Short' and ONLY on CE_OTM_Premium & PE_OTM_Premium.

Duplicates rule:
- Full-row duplicates are flagged.
- Key-based duplicate detection EXCLUDES any date-like columns (Date, ExpiryDate, trade_date, dt),
  so repeated dates due to having two actions per day are NOT counted as duplicates.

Outputs one Excel workbook:
  00_METRICS, 01_SUMMARY,
  02_MISSING_VALUES, 03_EXPIRY_MISMATCH, 04_CEPE_COUNT,
  05_NEGATIVES, 06_ZEROS, 07_DUPLICATES,
  08_SAME_ENTRY_EXIT, 09_OUTLIERS, 10_PREMIUM_TOLERANCE,
  11_CEPE_SAME_EXIT_TIME
"""

import os
import re
import glob
import argparse
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------- I/O helpers -----------------

def safe_read_csv(path: str) -> pd.DataFrame:
    """Robust CSV reader: strips null bytes and skips bad lines."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read().replace(b"\x00", b"")
        tmp = path + ".tmpclean"
        with open(tmp, "wb") as fh:
            fh.write(raw)
        try:
            df = pd.read_csv(tmp, engine="python", on_bad_lines="skip")
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass
        return df
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding_errors="ignore")

def safe_read_file(path: str) -> pd.DataFrame:
    """Universal reader: chooses read_excel or read_csv by extension (with fallbacks)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path)
        except Exception:
            return safe_read_csv(path)
    if ext == ".csv":
        return safe_read_csv(path)
    # unknown: try excel then csv
    try:
        return pd.read_excel(path)
    except Exception:
        return safe_read_csv(path)

# ----------------- Column helpers -----------------

def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def find_premium_cols(df: pd.DataFrame) -> List[str]:
    """Return the two premium columns (best-effort, case-insensitive)."""
    exact = []
    for name in ["CE_OTM_Premium", "PE_OTM_Premium"]:
        col = find_column(df, [name])
        if col:
            exact.append(col)
    if len(exact) == 2:
        return exact
    patt_ce = re.compile(r"^ce[_\s-]*otm[_\s-]*prem", re.I)
    patt_pe = re.compile(r"^pe[_\s-]*otm[_\s-]*prem", re.I)
    ce_col = next((c for c in df.columns if patt_ce.search(str(c))), None)
    pe_col = next((c for c in df.columns if patt_pe.search(str(c))), None)
    out = []
    if ce_col: out.append(ce_col)
    if pe_col: out.append(pe_col)
    return out

def list_numeric_columns(df: pd.DataFrame) -> List[str]:
    num_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            try:
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.notna().any():
                    num_cols.append(c)
            except Exception:
                pass
    return num_cols

def to_float(s: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")

def robust_outlier_mask(series: pd.Series, z_thresh: float = 15.0) -> pd.Series:
    """Robust outliers via median absolute deviation (MAD)."""
    s = to_float(series)
    if s.notna().sum() < 20:
        return pd.Series([False] * len(s), index=s.index)
    med = s.median()
    mad = (s - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return pd.Series([False] * len(s), index=s.index)
    z = (s - med).abs() / (mad + 1e-9)
    return z > z_thresh

def parse_premium_from_filename(filename: str):
    """Extract premium number from tokens like 'premium_15' or 'premium15'."""
    base = os.path.basename(filename)
    m = re.search(r"premium[_-]?([0-9]+(?:\.[0-9]+)?)", base, flags=re.I)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None

def coerce_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Create a unified 'dt' column when possible."""
    df = df.copy()
    dt_col = None
    dt_like = [c for c in df.columns if "datetime" in str(c).lower()]
    if dt_like:
        best = None
        best_count = -1
        for c in dt_like:
            c_parsed = pd.to_datetime(df[c], errors="coerce")
            count = c_parsed.notna().sum()
            if count > best_count:
                best, best_count = c_parsed, count
        dt_col = best
    else:
        date_col = find_column(df, ["Date", "trade_date"])
        time_col = find_column(df, ["Time", "trade_time", "theoretical_time"])
        if date_col and time_col:
            d = pd.to_datetime(df[date_col], errors="coerce").dt.date.astype(str)
            t = df[time_col].astype(str)
            dt_col = pd.to_datetime(d + " " + t, errors="coerce")
        elif date_col:
            dt_col = pd.to_datetime(df[date_col], errors="coerce")
    if dt_col is not None:
        df["dt"] = pd.to_datetime(dt_col, errors="coerce")
    return df

# ----------------- Core checks -----------------

def run_checks_on_df(df: pd.DataFrame, file: str) -> Dict[str, Any]:
    df = df.copy()
    df = coerce_datetime_cols(df)

    issues = {
        "missing_values": [],           # NaNs only in premium columns
        "expiry_mismatch": [],
        "cepe_count_mismatch": [],
        "negatives": [],                # negatives only in premium columns, excluding Action=='Long'
        "zeros": [],                    # zeros only in premium columns
        "duplicates": [],               # full-row dupes + key-based dupes excluding date-like cols
        "same_entry_exit_time": [],
        "outliers": [],
        "premium_tolerance": [],        # only for Short rows in premium columns
        "cepe_same_exit_time": [],
    }

    # Columns
    premium_cols = find_premium_cols(df)
    action_col = find_column(df, ["Action"])
    date_col   = find_column(df, ["Date", "trade_date"])
    expiry_col = find_column(df, ["ExpiryDate", "expiry_date", "Expiry"])
    entry_time_col = find_column(df, ["entry_time", "EntryTime", "entry time", "entrytime"])
    exit_time_col  = find_column(df, ["exit_time", "ExitTime", "exit time", "exittime"])
    type_col = find_column(df, ["type", "LEG", "option_type", "Type"])
    marker_col = find_column(df, ["entry_exit_error", "marker", "status", "Reason", "reason"])

    # Action masks
    if action_col:
        actions = df[action_col].astype(str).str.strip().str.lower()
        is_long = actions.eq("long")
        is_short = actions.eq("short")
    else:
        is_long = pd.Series(False, index=df.index)
        is_short = pd.Series(False, index=df.index)

    # 1) Missing values (ONLY in premium columns)
    if premium_cols:
        nan_mask_any = pd.Series(False, index=df.index)
        for c in premium_cols:
            nan_mask_any = nan_mask_any | df[c].isna()
        na_rows = df[nan_mask_any]
        if not na_rows.empty:
            tmp = na_rows[premium_cols].copy()
            tmp["__file__"] = os.path.basename(file)
            issues["missing_values"].append(tmp)

    # 2) Expiry-day check: Date == ExpiryDate
    if date_col and expiry_col:
        dvals = pd.to_datetime(df[date_col], errors="coerce").dt.date
        evals = pd.to_datetime(df[expiry_col], errors="coerce").dt.date
        mismatch = df[(dvals.notna()) & (evals.notna()) & (dvals != evals)]
        if not mismatch.empty:
            tmp = mismatch[[date_col, expiry_col]].copy()
            tmp["__file__"] = os.path.basename(file)
            tmp.rename(columns={date_col: "Date", expiry_col: "ExpiryDate"}, inplace=True)
            issues["expiry_mismatch"].append(tmp)

    # 3) Equal # of CE and PE entries (best-effort)
    if type_col is not None:
        types = df[type_col].astype(str).str.upper().str.strip()
        if marker_col is not None:
            markers = df[marker_col].astype(str).str.lower()
            is_entry = markers.str.contains("entry", na=False)
        else:
            is_entry = pd.Series([True] * len(df), index=df.index)
        ce_count = ((types.isin(["CE", "CALL"])) & is_entry).sum()
        pe_count = ((types.isin(["PE", "PUT"])) & is_entry).sum()
        if ce_count != pe_count:
            issues["cepe_count_mismatch"].append(pd.DataFrame({
                "__file__": [os.path.basename(file)],
                "CE_entries": [int(ce_count)],
                "PE_entries": [int(pe_count)]
            }))

    # 4) Negatives / Zeros ONLY on premium columns
    if premium_cols:
        for c in premium_cols:
            s = to_float(df[c])
            # negatives: count only when NOT Long
            neg_idx = s[s < 0].index
            if len(neg_idx) > 0:
                to_flag = df.index.isin(neg_idx) & (~is_long.values)
                if to_flag.any():
                    tmp = df.loc[to_flag, [c]].copy()
                    tmp["__file__"] = os.path.basename(file)
                    tmp["column"] = c
                    tmp["Action"] = df.loc[to_flag, action_col] if action_col else "NA"
                    issues["negatives"].append(tmp)
            # zeros: always flagged
            zero_idx = s[s == 0].index
            if len(zero_idx) > 0:
                tmp = df.loc[zero_idx, [c]].copy()
                tmp["__file__"] = os.path.basename(file)
                tmp["column"] = c
                issues["zeros"].append(tmp)

    # 5) Duplicates
    # 5a) Full-row duplicates (always meaningful)
    full_dups = df.duplicated(keep=False)
    if full_dups.any():
        tmp = df[full_dups].copy()
        tmp["__file__"] = os.path.basename(file)
        tmp["duplicate_scope"] = "full_row"
        issues["duplicates"].append(tmp)

    # 5b) Key-based duplicates (EXCLUDE date-like columns so repeated dates are fine)
    date_like = set([c for c in df.columns if re.search(r"(?:^|_)date(?:$|_)", str(c), re.I)] + ([ "dt" ] if "dt" in df.columns else []))
    key_candidates = [
        "trade_number", "Trade_ID", "trade_id",
        "instrument_name", "Instrument", "symbol",
        "strike", "StrikePrice",
        "type", "LEG",
        # intentionally excluding any date-like columns (Date/ExpiryDate/trade_date/dt)
    ]
    keys = [c for c in key_candidates if c in df.columns and c not in date_like]
    if keys:
        soft_dups = df.duplicated(subset=keys, keep=False)
        if soft_dups.any():
            tmp = df[soft_dups].copy()
            tmp["__file__"] = os.path.basename(file)
            tmp["duplicate_scope"] = "keys:" + ",".join(keys)
            issues["duplicates"].append(tmp)

    # 6) Same entry & exit time (row-level or fallback)
    if entry_time_col and exit_time_col:
        et = pd.to_datetime(df[entry_time_col], errors="coerce")
        xt = pd.to_datetime(df[exit_time_col], errors="coerce")
        same = df[(et.notna()) & (xt.notna()) & (et == xt)]
        if not same.empty:
            tmp = same[[entry_time_col, exit_time_col]].copy()
            tmp["__file__"] = os.path.basename(file)
            tmp.rename(columns={entry_time_col: "EntryTime", exit_time_col: "ExitTime"}, inplace=True)
            issues["same_entry_exit_time"].append(tmp)
    else:
        trade_id_col = find_column(df, ["trade_number", "Trade_ID", "trade_id"])
        time_col = find_column(df, ["trade_time", "theoretical_time", "Time"])
        marker_col2 = find_column(df, ["entry_exit_error", "marker", "status"])
        if trade_id_col and time_col and marker_col2:
            g = df[[trade_id_col, time_col, marker_col2]].copy()
            g["_is_entry"] = g[marker_col2].astype(str).str.lower().str.contains("entry", na=False)
            g["_is_exit"]  = g[marker_col2].astype(str).str.lower().str.contains("exit", na=False)
            t_same_list = []
            for tid, sub in g.groupby(trade_id_col):
                t_entry = pd.to_datetime(sub.loc[sub["_is_entry"], time_col], errors="coerce")
                t_exit  = pd.to_datetime(sub.loc[sub["_is_exit"], time_col], errors="coerce")
                if t_entry.notna().any() and t_exit.notna().any():
                    if any(t_entry.values == t_exit.values[:, None]):
                        t_same_list.append({"__file__": os.path.basename(file), "trade_id": tid})
            if t_same_list:
                issues["same_entry_exit_time"].append(pd.DataFrame(t_same_list))

    # 7) Outliers (robust MAD) across numeric columns
    for c in list_numeric_columns(df):
        mask = robust_outlier_mask(df[c])
        if mask.any():
            tmp = df.loc[mask, [c]].copy()
            tmp["__file__"] = os.path.basename(file)
            tmp["column"] = c
            issues["outliers"].append(tmp)

    # 8) Premium tolerance ONLY for Short and ONLY on premium columns
    prem_fname = parse_premium_from_filename(file)
    if prem_fname is not None and premium_cols:
        tol = 0.50 if prem_fname < 50 else 0.30
        low = prem_fname * (1 - tol)
        high = prem_fname * (1 + tol)
        for c in premium_cols:
            s = to_float(df[c])
            bad = is_short & s.notna() & ((s < low) | (s > high))
            if bad.any():
                tmp = df.loc[bad, [c]].copy()
                tmp["__file__"] = os.path.basename(file)
                tmp["column"] = c
                tmp["expected_low"] = low
                tmp["expected_high"] = high
                tmp["premium_from_filename"] = prem_fname
                tmp["Action"] = df.loc[bad, action_col] if action_col else "NA"
                issues["premium_tolerance"].append(tmp)

    # 9) CE & PE share the same exit time (pair-level)
    # prefer explicit exit_time; fallback to marker+time
    type_col2 = type_col
    trade_id_col2 = find_column(df, ["trade_number", "Trade_ID", "trade_id"])
    exit_time_col2 = exit_time_col or find_column(df, ["exit_time", "ExitTime", "exit time", "exittime"])
    marker_col3 = marker_col or find_column(df, ["entry_exit_error", "marker", "status"])
    time_col2 = find_column(df, ["trade_time", "theoretical_time", "Time"])

    if type_col2 is not None and trade_id_col2 is not None:
        types2 = df[type_col2].astype(str).str.upper().str.strip()
        if exit_time_col2:
            x_time = pd.to_datetime(df[exit_time_col2], errors="coerce")
            pairs = []
            for tid, sub in df[[trade_id_col2, type_col2]].join(x_time.rename("_x")).groupby(trade_id_col2):
                ce_xt = pd.to_datetime(sub.loc[types2.loc[sub.index].isin(["CE","CALL"]), "_x"], errors="coerce").dropna().unique()
                pe_xt = pd.to_datetime(sub.loc[types2.loc[sub.index].isin(["PE","PUT"]), "_x"], errors="coerce").dropna().unique()
                if ce_xt.size > 0 and pe_xt.size > 0 and any(ce_xt_i == pe_xt_j for ce_xt_i in ce_xt for pe_xt_j in pe_xt):
                    pairs.append({"__file__": os.path.basename(file), "trade_id": tid, "ExitTime": ce_xt[0]})
            if pairs:
                issues["cepe_same_exit_time"].append(pd.DataFrame(pairs))
        elif marker_col3 and time_col2:
            g = df[[trade_id_col2, type_col2, marker_col3, time_col2]].copy()
            g["_is_exit"] = g[marker_col3].astype(str).str.lower().str.contains("exit", na=False)
            pairs = []
            for tid, sub in g.groupby(trade_id_col2):
                ce_xt = pd.to_datetime(
                    sub.loc[sub["_is_exit"] & sub[type_col2].astype(str).str.upper().isin(["CE","CALL"]), time_col2],
                    errors="coerce"
                ).dropna().unique()
                pe_xt = pd.to_datetime(
                    sub.loc[sub["_is_exit"] & sub[type_col2].astype(str).str.upper().isin(["PE","PUT"]), time_col2],
                    errors="coerce"
                ).dropna().unique()
                if ce_xt.size > 0 and pe_xt.size > 0 and any(ce_xt_i == pe_xt_j for ce_xt_i in ce_xt for pe_xt_j in pe_xt):
                    pairs.append({"__file__": os.path.basename(file), "trade_id": tid, "ExitTime": ce_xt[0]})
            if pairs:
                issues["cepe_same_exit_time"].append(pd.DataFrame(pairs))

    # Per-file counts
    counts = {
        "file": os.path.basename(file),
        "rows": int(len(df)),
        "missing_rows": int(sum(len(x) for x in issues["missing_values"])),
        "expiry_mismatches": int(sum(len(x) for x in issues["expiry_mismatch"])),
        "cepe_mismatch_flag": int(len(issues["cepe_count_mismatch"]) > 0),
        "negative_cells": int(sum(len(x) for x in issues["negatives"])),
        "zero_cells": int(sum(len(x) for x in issues["zeros"])),
        "duplicate_rows": int(sum(len(x) for x in issues["duplicates"])),
        "same_entry_exit_rows": int(sum(len(x) for x in issues["same_entry_exit_time"])),
        "outlier_rows": int(sum(len(x) for x in issues["outliers"])),
        "premium_tolerance_rows": int(sum(len(x) for x in issues["premium_tolerance"])),
        "cepe_same_exit_pairs": int(sum(len(x) for x in issues["cepe_same_exit_time"])),
    }

    return {"counts": counts, "issues": issues}

# ----------------- Driver & Report -----------------

def scan_folder(root: str, pattern: str, recursive: bool = False):
    files = glob.glob(os.path.join(root, "**", pattern), recursive=True) if recursive \
            else glob.glob(os.path.join(root, pattern))
    return sorted([f for f in files if os.path.isfile(f)])

def write_excel_report(out_path: str, summary_df: pd.DataFrame, issues: Dict[str, pd.DataFrame]):
    # Prefer xlsxwriter; fallback to openpyxl
    engine = "xlsxwriter"
    try:
        with pd.ExcelWriter(out_path, engine=engine) as writer:
            _write_sheets(writer, summary_df, issues)
        return
    except Exception:
        pass
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        _write_sheets(writer, summary_df, issues)

def _write_sheets(writer: pd.ExcelWriter, summary_df: pd.DataFrame, issues: Dict[str, pd.DataFrame]):
    total_files = len(summary_df) if summary_df is not None else 0
    total_rows = int(summary_df["rows"].sum()) if total_files else 0
    metrics = pd.DataFrame([{
        "files_processed": total_files,
        "total_rows": total_rows,
        "files_with_cepe_mismatch": int((summary_df["cepe_mismatch_flag"] > 0).sum()) if total_files else 0,
        "total_missing_rows": int(summary_df["missing_rows"].sum()) if total_files else 0,
        "total_expiry_mismatches": int(summary_df["expiry_mismatches"].sum()) if total_files else 0,
        "total_negative_cells": int(summary_df["negative_cells"].sum()) if total_files else 0,
        "total_zero_cells": int(summary_df["zero_cells"].sum()) if total_files else 0,
        "total_duplicate_rows": int(summary_df["duplicate_rows"].sum()) if total_files else 0,
        "total_same_entry_exit_rows": int(summary_df["same_entry_exit_rows"].sum()) if total_files else 0,
        "total_outlier_rows": int(summary_df["outlier_rows"].sum()) if total_files else 0,
        "total_premium_tolerance_rows": int(summary_df["premium_tolerance_rows"].sum()) if total_files else 0,
        "total_cepe_same_exit_pairs": int(summary_df["cepe_same_exit_pairs"].sum()) if total_files else 0,
    }])

    metrics.to_excel(writer, sheet_name="00_METRICS", index=False)
    (summary_df if summary_df is not None else pd.DataFrame()).to_excel(
        writer, sheet_name="01_SUMMARY", index=False
    )
    issues.get("missing_values", pd.DataFrame()).to_excel(writer, sheet_name="02_MISSING_VALUES", index=False)
    issues.get("expiry_mismatch", pd.DataFrame()).to_excel(writer, sheet_name="03_EXPIRY_MISMATCH", index=False)
    issues.get("cepe_count_mismatch", pd.DataFrame()).to_excel(writer, sheet_name="04_CEPE_COUNT", index=False)
    issues.get("negatives", pd.DataFrame()).to_excel(writer, sheet_name="05_NEGATIVES", index=False)
    issues.get("zeros", pd.DataFrame()).to_excel(writer, sheet_name="06_ZEROS", index=False)
    issues.get("duplicates", pd.DataFrame()).to_excel(writer, sheet_name="07_DUPLICATES", index=False)
    issues.get("same_entry_exit_time", pd.DataFrame()).to_excel(writer, sheet_name="08_SAME_ENTRY_EXIT", index=False)
    issues.get("outliers", pd.DataFrame()).to_excel(writer, sheet_name="09_OUTLIERS", index=False)
    issues.get("premium_tolerance", pd.DataFrame()).to_excel(writer, sheet_name="10_PREMIUM_TOLERANCE", index=False)
    issues.get("cepe_same_exit_time", pd.DataFrame()).to_excel(writer, sheet_name="11_CEPE_SAME_EXIT_TIME", index=False)

# ----------------- Multiprocessing -----------------

def process_one_file(path: str) -> Optional[Dict[str, Any]]:
    """Worker: read file, run checks, return results. Returns minimal structure on read error."""
    try:
        df = safe_read_file(path)
    except Exception:
        return {
            "counts": {
                "file": os.path.basename(path), "rows": 0,
                "missing_rows": 0, "expiry_mismatches": 0, "cepe_mismatch_flag": 0,
                "negative_cells": 0, "zero_cells": 0, "duplicate_rows": 0,
                "same_entry_exit_rows": 0, "outlier_rows": 0,
                "premium_tolerance_rows": 0, "cepe_same_exit_pairs": 0,
            },
            "issues": {k: [] for k in [
                "missing_values","expiry_mismatch","cepe_count_mismatch","negatives","zeros",
                "duplicates","same_entry_exit_time","outliers","premium_tolerance","cepe_same_exit_time"
            ]},
        }
    return run_checks_on_df(df, path)

# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser(description="Tradesheet Sanity Checker (multiprocessing, CSV/XLSX, Rule 9 + premium-scoped checks)")
    parser.add_argument("--root", required=True, help="Root folder containing tradesheets")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern (e.g., *.csv, *.xlsx, *)")
    parser.add_argument("--recursive", action="store_true", help="Scan subfolders recursively")
    parser.add_argument("--out", default="tradesheet_sanity_report.xlsx", help="Output Excel path")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Max worker processes (default: CPU cores - 1)")
    parser.add_argument("--mp-chunksize", type=int, default=50,
                        help="Files per task chunk for executor submission")
    args = parser.parse_args()

    files = scan_folder(args.root, args.pattern, args.recursive)
    if not files:
        print("No files found. Check --root / --pattern / --recursive.")
        return

    summary_rows: List[Dict[str, Any]] = []
    agg_issues: Dict[str, List[pd.DataFrame]] = {
        "missing_values": [],
        "expiry_mismatch": [],
        "cepe_count_mismatch": [],
        "negatives": [],
        "zeros": [],
        "duplicates": [],
        "same_entry_exit_time": [],
        "outliers": [],
        "premium_tolerance": [],
        "cepe_same_exit_time": [],
    }

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for i in range(0, len(files), args.mp_chunksize):
            for f in files[i:i + args.mp_chunksize]:
                futures.append(ex.submit(process_one_file, f))

        for fut in as_completed(futures):
            res = fut.result()
            if not res:
                continue
            summary_rows.append(res["counts"])
            for k, chunks in res["issues"].items():
                if chunks:
                    agg_issues[k].append(pd.concat(chunks, ignore_index=True))

    summary_df = pd.DataFrame(summary_rows).sort_values("file") if summary_rows else pd.DataFrame()

    final_issues: Dict[str, pd.DataFrame] = {}
    for k, dfs in agg_issues.items():
        final_issues[k] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    write_excel_report(args.out, summary_df, final_issues)

    print(f"Processed {len(files)} file(s) using {args.workers} worker(s).")
    print(f"Report written to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()


# /home/newberry3/main/Data/straddle_premium_data/NIFTY_candle_5T_premium_tp_NA_entry_13,35_exit_15,15_premium_15_stoploss_0,4_target_0,7_combined_target_0,4_vix_range_(12, 14,5).csv
#/home/newberry3/disha/tradesheet_error_SS.py
# python /home/newberry3/disha/tradesheet_error_SS.py \
#   --root /home/newberry3/main/Data/straddle_premium_data/data_check_error_code/Trade_Sheets \
#   --pattern "*.csv" \
#   --recursive \
#   --workers 8 \
#   --out /home/newberry3/main/Data/straddle_premium_data/data_check_error_code/tradesheet_sanity_report.xlsx