from pathlib import Path


# ============================================================
# BACKTEST CONFIG
# Edit only this file for normal usage
# ============================================================


# ---------- Strategy identity ----------
STRATEGY_NAME = "VIX_NIFTY_15M_HOLD_TO_EXPIRY"


# ---------- Strategy structure variants ----------
# Used ONLY when ENTRY_MODE = "base_vix".
# Regime-based entry overrides both of these per-trade from REGIME_TO_STRATEGY.
STRATEGY_STRUCTURES           = ["IRON_CONDOR"]           # "IRON_CONDOR" or "BATMAN"
SHORT_DTE_STD_MULTIPLIERS     = [1.8]                # used by base_vix path only

BATMAN_TARGET_PREMIUM_FRACTION    = 0.70
BATMAN_NEAREST_PREMIUM_FALLBACK_MIN = 4
BATMAN_INNER_LONG_QTY             = 1
BATMAN_SHORT_QTY                  = 3
BATMAN_OUTER_LONG_QTY             = 2
IRON_CONDOR_LEG_QTY               = 3
SAVE_OUTPUTS_BY_VARIANT           = True


# ── Patch 3: Regime-driven entry ─────────────────────────────────
# ENTRY_WINDOW_MINUTES : int or None
# ---------- Time windows ----------
ENTRY_WINDOW_MINUTES = None   # ← keep None, windows are handled by TIME_WINDOWS
TIME_WINDOWS = []
#TIME_WINDOWS = [
#     ("W1", "09:30", "11:30"),
#     ("W2", "11:31", "13:30"),
#     ("W3", "13:31", "15:15"),
# ]
# REGIME_DRIVEN_ENTRY : bool
#   True  → call regime_classifier.resolve_trade_params per signal → 1 (structure, multiplier) pair
#   False → use STRATEGY_STRUCTURES × SHORT_DTE_STD_MULTIPLIERS grid (original behaviour)
REGIME_DRIVEN_ENTRY = False


# ---------- Run window ----------
START_DATE         = "2022-06-07"
END_DATE           = "2025-10-28"
EXCLUDED_EXPIRIES  = ["2024-06-06"]


# ---------- Input paths ----------
INDEX_DATA_PATH       = "/home/newberry3/disha/vix_backtesting/Nifty_index_data.csv"
OPTIONS_DATA_FOLDER   = "/home/newberry3/disha/vix_backtesting/ohlc_with_all_strikes_new/ohlc_with_all_strikes_new"
EXPIRY_MAP_PATH       = "/home/newberry3/disha/vix_backtesting/dte_mapping.csv"
FEATURE_SHEET_PATH    = "/home/newberry3/disha/vix_backtesting/features_15min_new.xlsx"


# ---------- Output ----------
OUTPUT_DIR = Path("/home/newberry3/disha/vix_backtesting/vix_backtest_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ENTRY MODE
# ============================================================
# Layer 1 — feature gate (unchanged):
#   base_vix          -> every signal row becomes a trade
#   feature_filter_v1 -> only feature-filtered signal rows become trades gao+orb
#   feature_filter_v2 -> gap+rvs/rvf
#
# Layer 2 — entry style (applied AFTER Layer 1 passes):
#   all_signals       -> take every bar that passes Layer 1 (original behaviour)
#   time_windows      -> take only the FIRST signal per time window per day
#                        that passes Layer 1.  Windows defined in TIME_WINDOWS.
#
# Layer 3 — structure selection:
#   fixed             -> use STRATEGY_STRUCTURES / SHORT_DTE_STD_MULTIPLIERS above
#   regime_based      -> pick structure + multiplier from REGIME_TO_STRATEGY per bar
#
ENTRY_MODE           = "feature_filter_v2"    # Layer 1: feature gate


# ---------- Time windows (used when ENTRY_FREQUENCY = "time_windows") ----------
# Take at most 1 trade per window per day (earliest signal that clears all gates).
# Format: (window_label, start_HH:MM_inclusive, end_HH:MM_inclusive)


# ---------- Option access mode ----------
# lazy                       -> fetch one contract on demand and cache it
# preload_required_contracts -> prefetch only contracts actually needed by trade intents
OPTION_ACCESS_MODE = "preload_required_contracts"


# ---------- Signal settings ----------
SIGNAL_TIMEFRAME   = "15min"
SESSION_START      = "09:15"
SESSION_END        = "15:29"
MANUAL_ENTRY_TIMES = None
# Example:
# MANUAL_ENTRY_TIMES = ["09:30", "09:45", "10:00", "10:15"]


# ---------- Contract / strike settings ----------
UNDERLYING_NAME = "NIFTY"
STRIKE_STEP     = 50
QTY_PER_LEG     = 1
LOT_SIZE        = 1


# ---------- Entry / exit pricing ----------
ENTRY_PRICE_FIELD              = "Open"
EXIT_PRICE_FIELD               = "Open"
ALLOW_NEAREST_FILL             = True
MAX_ENTRY_FALLBACK_MIN         = 10
MAX_EXIT_FALLBACK_MIN          = 10
SKIP_TRADE_IF_ANY_LEG_MISSING  = True


# ---------- Option price gap filling ----------
OPTION_PRICE_FILL_METHOD          = "ffill+bfill"   # "ffill" | "bfill" | "ffill+bfill" | None
OPTION_PRICE_FILL_LIMIT           = None             # max consecutive bars, or None = unlimited
NEAREST_STRIKE_FALLBACK_ENABLED   = True
MAX_STRIKE_FALLBACK_STEPS         = 3


# ---------- Exit policy ----------
EXIT_POLICY           = "HOLD_TILL_EXPIRY"
EXPIRY_EXIT_TIME      = "15:15"
EXIT_REASON_DEFAULT   = "EXPIRY_EXIT"


# ---------- Forced option fallback price ----------
FORCE_OPTION_PRICE_IF_MISSING = True
FORCED_EXPIRY_PRICE           = 0.00
FORCED_NON_EXPIRY_PRICE       = 0.05
FORCED_EXPIRY_CUTOFF_TIME     = "15:15"


# ---------- Cache / preload ----------
CONTRACT_CACHE_MAX_ITEMS        = 5000
PRELOAD_BATCH_SIZE              = 1_000_000
PRELOAD_PROGRESS_EVERY_BATCHES  = 10


# ---------- Logging ----------
PRINT_PROGRESS_EVERY = 100
SHOW_PREVIEWS        = True


# ---------- Save ----------
SAVE_CSV = True


# ---------- Optional intratrade MTM timeline ----------
SAVE_INTRATRADE_TIMELINE          = True
INTRATRADE_MARK_TIMEFRAME         = "15min"
INTRATRADE_MAX_FALLBACK_MIN       = 15
INTRATRADE_STRICT_AFTER_ONLY      = False
INTRATRADE_INCLUDE_FINAL_EXIT_ROW = True
SAVE_INTRATRADE_TIMELINE_CSV      = True


# ---------- Feature mode v1 rules ----------
FEATURE_RULE_NAME                = "feature_filter_v2"
FEATURE_FILTER_V1_SCAN_START     = "09:30"
FEATURE_FILTER_V1_ORB_EARLIEST   = "10:15"
FEATURE_FILTER_V1_DTE0_LAST_ENTRY = "10:00"


# ---------- Feature columns to print in trade_details output ----------
PRINT_FEATURES_IN_TRADE_DETAILS = True

FEATURE_COLUMNS_TO_PRINT = [
    "one_std",
    "dte_std",
    "gap",
    "orb_1_std",
    "rv_slow_over_rv_fast_1_3",
    "move_to_open_1_std",
    "rv_slow",
    "iv",
]


# ---------- Parallelism ----------
NUM_WORKERS = 0


# ---------- Benchmark ----------
BENCHMARK_MODE = True


# ============================================================
# REGIME CLASSIFICATION ENGINE
# ============================================================
# Used when REGIME_ENTRY_MODE = "regime_based".
# Feature columns required (already in FEATURE_SHEET_PATH):
#   rv_slow            : annualised realised vol (decimal, e.g. 0.09 = 9%)
#   iv                 : implied volatility (decimal, e.g. 0.115 = 11.5%)
#   move_to_open_1_std : today's gap/move normalised by 1-std (signed float)
# ============================================================

# ---- Zone thresholds ----
# Each feature is bucketed: Low / Medium / High
# Boundaries are INCLUSIVE at the upper end of each lower bucket.
#   Low    : value <= low_max
#   Medium : low_max < value <= med_max
#   High   : value > med_max
ZONE_THRESHOLDS = {
    "move_to_open_1_std": {"low_max": -0.125, "med_max":  0.25},
    "rv_slow":            {"low_max":  0.09,  "med_max":  0.115},
    "iv":                 {"low_max":  0.115, "med_max":  0.17},
}


# ---- 27-variation master table: (old_no, rv_zone, iv_zone, move_zone) ----
# Order: RV × IV × Move (the 3-axis zone triplet space)
VARIATION_DEFS_27 = [
    (1,  "Low",    "Low",    "Low"),
    (2,  "Low",    "Low",    "Medium"),
    (3,  "Low",    "Low",    "High"),
    (4,  "Low",    "Medium", "Low"),
    (5,  "Low",    "Medium", "Medium"),
    (6,  "Low",    "Medium", "High"),
    (7,  "Low",    "High",   "Low"),
    (8,  "Low",    "High",   "Medium"),
    (9,  "Low",    "High",   "High"),
    (10, "Medium", "Low",    "Low"),
    (11, "Medium", "Low",    "Medium"),
    (12, "Medium", "Low",    "High"),
    (13, "Medium", "Medium", "Low"),
    (14, "Medium", "Medium", "Medium"),
    (15, "Medium", "Medium", "High"),
    (16, "Medium", "High",   "Low"),
    (17, "Medium", "High",   "Medium"),
    (18, "Medium", "High",   "High"),
    (19, "High",   "Low",    "Low"),
    (20, "High",   "Low",    "Medium"),
    (21, "High",   "Low",    "High"),
    (22, "High",   "Medium", "Low"),
    (23, "High",   "Medium", "Medium"),
    (24, "High",   "Medium", "High"),
    (25, "High",   "High",   "Low"),
    (26, "High",   "High",   "Medium"),
    (27, "High",   "High",   "High"),
]


# ---- Collapse 27 old variations → 20 new regimes ----
OLD_TO_NEW_REGIME_MAP = {
    1: 1,   2: 2,   3: 3,   4: 4,   5: 5,   6: 6,
    7: 7,   8: 7,   9: 7,
    10: 8,  11: 8,  12: 8,
    13: 9,  14: 10, 15: 11,
    16: 12, 17: 13, 18: 14,
    19: 15, 20: 15, 21: 15, 22: 15,
    23: 16, 24: 17,
    25: 18, 26: 19, 27: 20,
}


# ---- Regime → Strategy ----
# Maps new regime number (1-20) to the structure and DTE-std multiplier
# to use at entry for that regime.
# Keys:
#   structure   : "BATMAN" | "IRON_CONDOR"
#   multiplier  : SHORT_DTE_STD_MULTIPLIER for this trade
# REGIME_TO_STRATEGY = {
#     1:  {"structure": "BATMAN",       "multiplier": 0.8},
#     2:  {"structure": "BATMAN",       "multiplier": 0.8},
#     3:  {"structure": "IRON_CONDOR",  "multiplier": 1.2},
#     4:  {"structure": "IRON_CONDOR",  "multiplier": 1.2},
#     5:  {"structure": "BATMAN",       "multiplier": 1.2},
#     6:  {"structure": "BATMAN",       "multiplier": 1.2},
#     7:  {"structure": "IRON_CONDOR",  "multiplier": 1.2},
#     8:  {"structure": "BATMAN",       "multiplier": 0.8},
#     9:  {"structure": "BATMAN",       "multiplier": 1.2},
#     10: {"structure": "BATMAN",       "multiplier": 1.2},
#     11: {"structure": "BATMAN",       "multiplier": 1.2},
#     12: {"structure": "BATMAN",       "multiplier": 1.2},
#     13: {"structure": "BATMAN",       "multiplier": 1.2},
#     14: {"structure": "BATMAN",       "multiplier": 1.2},
#     15: {"structure": "IRON_CONDOR",  "multiplier": 1.2},
#     16: {"structure": "IRON_CONDOR",  "multiplier": 1.2},
#     17: {"structure": "IRON_CONDOR",  "multiplier": 1.2},
#     18: {"structure": "IRON_CONDOR",  "multiplier": 1.2},
#     19: {"structure": "IRON_CONDOR",  "multiplier": 1.2},
#     20: {"structure": "IRON_CONDOR",  "multiplier": 1.2},
# }
REGIME_TO_STRATEGY = {
    1:  {"structure": "IRON_CONDOR", "multiplier": 0.8},
    2:  {"structure": "IRON_CONDOR", "multiplier": 0.8},
    3:  {"structure": "IRON_CONDOR", "multiplier": 0.8},
    4:  {"structure": "IRON_CONDOR", "multiplier": 0.8},
    5:  {"structure": "IRON_CONDOR", "multiplier": 1.2},
    6:  {"structure": "IRON_CONDOR", "multiplier": 1.2},
    7:  {"structure": "IRON_CONDOR", "multiplier": 1.2},
    8:  {"structure": "IRON_CONDOR", "multiplier": 0.8},
    9:  {"structure": "IRON_CONDOR", "multiplier": 1.2},
    10: {"structure": "IRON_CONDOR", "multiplier": 1.0},
    11: {"structure": "IRON_CONDOR", "multiplier": 1.2},
    12: {"structure": "IRON_CONDOR", "multiplier": 1.2},
    13: {"structure": "IRON_CONDOR", "multiplier": 1.2},
    14: {"structure": "IRON_CONDOR", "multiplier": 1.2},
    15: {"structure": "IRON_CONDOR", "multiplier": 1.0},
    16: {"structure": "IRON_CONDOR", "multiplier": 1.0},
    17: {"structure": "IRON_CONDOR", "multiplier": 1.2},
    18: {"structure": "IRON_CONDOR", "multiplier": 1.2},
    19: {"structure": "IRON_CONDOR", "multiplier": 1.2},
    20: {"structure": "IRON_CONDOR", "multiplier": 1.2},
}

# ============================================================
# EXIT RULES ENGINE — SL BY REGIME
# ============================================================
# ---- Regime → Stop-Loss definition ----
# sl_type:
#   "SD"          -> BATMAN regimes.  Exit one side when spot crosses
#                    (short_strike ± sl_value × one_std).
#                    CE side : trigger when spot >= short_CE_strike + sl_value * one_std
#                    PE side : trigger when spot <= short_PE_strike - sl_value * one_std
#                    Exact tie in loss increase on both sides → CUT PE.
#
#   "PCT_CREDIT"  -> IRON_CONDOR regimes.  Exit one side when the current
#                    mark price of that side's short leg reaches
#                    (side_entry_credit × sl_value).
#                    CE side : trigger when short_CE_mark >= short_CE_entry * sl_value
#                    PE side : trigger when short_PE_mark >= short_PE_entry * sl_value
#                    Exact tie in short-loss increase on both sides → CUT PE.
#
# sl_value  : the multiplier (SD count for "SD", premium ratio for "PCT_CREDIT")
# exit_side : "one_side"  -> cut only the triggered side; other side holds to expiry
#             "full"      -> exit entire position (not currently used in any regime)
#
# SL on final expiry bar → mark as EXPIRY_EXIT not SL_EXIT (handled in runner).

# for hold to expiry
REGIME_TO_SL = {
    1:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    2:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    3:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    4:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    5:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    6:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    7:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    8:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    9:  {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    10: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    11: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    12: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    13: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    14: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    15: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    16: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    17: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    18: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    19: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
    20: {"sl_type": "PCT_CREDIT", "sl_value": 9999, "exit_side": "one_side"},
}
#SL FOR REGIME BASED
# REGIME_TO_SL = {
#     1:  {"sl_type": "SD",         "sl_value": 1.10, "exit_side": "one_side"},
#     2:  {"sl_type": "SD",         "sl_value": 1.10, "exit_side": "one_side"},
#     3:  {"sl_type": "PCT_CREDIT", "sl_value": 3.00, "exit_side": "one_side"},
#     4:  {"sl_type": "PCT_CREDIT", "sl_value": 2.90, "exit_side": "one_side"},
#     5:  {"sl_type": "SD",         "sl_value": 1.50, "exit_side": "one_side"},
#     6:  {"sl_type": "SD",         "sl_value": 1.20, "exit_side": "one_side"},
#     7:  {"sl_type": "PCT_CREDIT", "sl_value": 3.60, "exit_side": "one_side"},
#     8:  {"sl_type": "SD",         "sl_value": 1.30, "exit_side": "one_side"},
#     9:  {"sl_type": "SD",         "sl_value": 1.40, "exit_side": "one_side"},
#     10: {"sl_type": "SD",         "sl_value": 1.30, "exit_side": "one_side"},
#     11: {"sl_type": "SD",         "sl_value": 1.40, "exit_side": "one_side"},
#     12: {"sl_type": "SD",         "sl_value": 1.40, "exit_side": "one_side"},
#     13: {"sl_type": "SD",         "sl_value": 1.10, "exit_side": "one_side"},
#     14: {"sl_type": "SD",         "sl_value": 1.40, "exit_side": "one_side"},
#     15: {"sl_type": "PCT_CREDIT", "sl_value": 1.60, "exit_side": "one_side"},
#     16: {"sl_type": "PCT_CREDIT", "sl_value": 1.90, "exit_side": "one_side"},
#     17: {"sl_type": "PCT_CREDIT", "sl_value": 1.50, "exit_side": "one_side"},
#     18: {"sl_type": "PCT_CREDIT", "sl_value": 1.50, "exit_side": "one_side"},
#     19: {"sl_type": "PCT_CREDIT", "sl_value": 2.10, "exit_side": "one_side"},
#     20: {"sl_type": "PCT_CREDIT", "sl_value": 1.90, "exit_side": "one_side"},
# }

# ---- SL evaluation timeframe ----
# Bar frequency used to check SL conditions intraday.
# Must match or be finer than INTRATRADE_MARK_TIMEFRAME.
SL_EVAL_TIMEFRAME       = "1min"
SL_EVAL_MAX_FALLBACK_MIN = 15
SLTOP_MAX_FALLBACK_MIN  = 15


# ============================================================
# COMBINED PROFIT TARGET — by regime
# ============================================================
# Checked every 1-min bar BEFORE the SL check.
# Fires ONLY when BOTH CE and PE sides are still fully open.
# (Once SL has cut one side, this check is disabled for that trade.)
#
# target_pct : e.g. 90  -> exit when net P&L >= 90% of entry net credit
#              e.g. 110 -> exit when net P&L >= 110% of entry net credit
#
# ENABLE_COMBINED_PROFIT_TARGET = False disables entirely for all regimes.
ENABLE_COMBINED_PROFIT_TARGET = False

REGIME_TO_COMBINED_TARGET = {
    1:  {"target_pct": 110},
    2:  {"target_pct": 130},
    3:  {"target_pct":  40},
    4:  {"target_pct": 110},
    5:  {"target_pct": 130},
    6:  {"target_pct": 130},
    7:  {"target_pct":  90},
    8:  {"target_pct": 120},
    9:  {"target_pct": 120},
    10: {"target_pct": 110},
    11: {"target_pct":  30},
    12: {"target_pct": 120},
    13: {"target_pct": 120},
    14: {"target_pct": 110},
    15: {"target_pct":  80},
    16: {"target_pct":  90},
    17: {"target_pct":  90},
    18: {"target_pct":  90},
    19: {"target_pct":  80},
    20: {"target_pct":  90},
}


# ============================================================
# SECOND LEG MANAGEMENT ENGINE
# ============================================================
# When ENABLE_SECOND_LEG_MANAGEMENT = True:
#   After one side is cut by SL, the surviving side is re-entered
#   at that exact bar with a fresh net credit (surviving short mark price).
#   That second leg is then independently managed with its own SL and
#   individual profit target until expiry.
#
# sl_type is INHERITED from REGIME_TO_SL (same "SD" or "PCT_CREDIT").
# sl_value_2 is the second-leg SL multiplier (SD count or pct multiplier).
# target_pct_2 is the individual profit target as % of second-leg entry credit.
#   e.g. 100 -> exit when second-leg running PnL >= 1x second-leg entry credit.
#
# For SD regimes: one_std is re-read from spot at the second-leg entry bar.
ENABLE_SECOND_LEG_MANAGEMENT = False

REGIME_TO_SECOND_LEG = {
    1:  {"sl_value_2": 1.00, "target_pct_2": 100},
    2:  {"sl_value_2": 1.00, "target_pct_2":  70},
    3:  {"sl_value_2": 1.50, "target_pct_2":  95},
    4:  {"sl_value_2": 1.50, "target_pct_2":  95},
    5:  {"sl_value_2": 0.50, "target_pct_2":  70},
    6:  {"sl_value_2": 1.00, "target_pct_2": 120},
    7:  {"sl_value_2": 1.00, "target_pct_2":  95},
    8:  {"sl_value_2": 1.50, "target_pct_2": 100},
    9:  {"sl_value_2": 0.50, "target_pct_2":  40},
    10: {"sl_value_2": 0.50, "target_pct_2": 100},
    11: {"sl_value_2": 0.50, "target_pct_2":  60},
    12: {"sl_value_2": 0.50, "target_pct_2":  80},
    13: {"sl_value_2": 1.00, "target_pct_2":  90},
    14: {"sl_value_2": 1.00, "target_pct_2":  60},
    15: {"sl_value_2": 3.00, "target_pct_2":  95},
    16: {"sl_value_2": 2.50, "target_pct_2":  95},
    17: {"sl_value_2": 2.00, "target_pct_2":  95},
    18: {"sl_value_2": 2.00, "target_pct_2":  95},
    19: {"sl_value_2": 3.50, "target_pct_2":  95},
    20: {"sl_value_2": 3.50, "target_pct_2":  95},
}
