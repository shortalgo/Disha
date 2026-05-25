"""
regime_classifier.py
====================
Stateless utility module for regime classification and per-regime
strategy / SL lookups.

All thresholds and mappings are sourced from vix_backtest_config.py.
No data is loaded here — pass raw feature values in, get regime/strategy/SL out.

Public API
----------
classify_zone(value, feature_name)          -> "Low" | "Medium" | "High" | None
classify_all_zones(rv_slow, iv, move)       -> (rv_zone, iv_zone, move_zone)
get_regime(rv_slow, iv, move)               -> int (1-20) | None
get_strategy_for_regime(regime)             -> {"structure", "multiplier"} | None
get_sl_for_regime(regime)                   -> {"sl_type", "sl_value", "exit_side"} | None
resolve_trade_params(rv_slow, iv, move)     -> full dict or None  (convenience wrapper)
"""

import pandas as pd
import vix_backtest_config as cfg


_OLD_VARIATION_LOOKUP = {
    (rv, iv, mv): old_no
    for old_no, rv, iv, mv in cfg.VARIATION_DEFS_27
}


def classify_zone(value, feature_name: str):
    """
    Classify a single feature value into "Low", "Medium", or "High"
    using the thresholds in cfg.ZONE_THRESHOLDS.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(v):
        return None

    thresholds = cfg.ZONE_THRESHOLDS.get(feature_name)
    if thresholds is None:
        raise ValueError(
            f"[regime_classifier] Unknown feature_name='{feature_name}'. "
            f"Valid keys: {list(cfg.ZONE_THRESHOLDS.keys())}"
        )

    if feature_name in ("rv_slow", "iv") and v < 0:
        return None

    low_max = thresholds["low_max"]
    med_max = thresholds["med_max"]

    if v <= low_max:
        return "Low"
    if v <= med_max:
        return "Medium"
    return "High"


def classify_all_zones(rv_slow, iv, move_to_open_1_std):
    rv_zone = classify_zone(rv_slow, "rv_slow")
    iv_zone = classify_zone(iv, "iv")
    move_zone = classify_zone(move_to_open_1_std, "move_to_open_1_std")
    return rv_zone, iv_zone, move_zone


def get_regime(rv_slow, iv, move_to_open_1_std):
    """
    Resolve the new regime number (1-20) from raw feature values.
    """
    rv_zone, iv_zone, move_zone = classify_all_zones(rv_slow, iv, move_to_open_1_std)

    if rv_zone is None or iv_zone is None or move_zone is None:
        return None

    key = (rv_zone, iv_zone, move_zone)
    old_no = _OLD_VARIATION_LOOKUP.get(key)
    if old_no is None:
        raise ValueError(
            f"[regime_classifier] Zone triplet (rv={rv_zone}, iv={iv_zone}, "
            f"move={move_zone}) is not present in VARIATION_DEFS_27. "
            f"This should never occur — check zone classification logic."
        )

    new_regime = cfg.OLD_TO_NEW_REGIME_MAP.get(old_no)
    if new_regime is None:
        raise ValueError(
            f"[regime_classifier] old_variation={old_no} has no entry in "
            f"OLD_TO_NEW_REGIME_MAP. Check the mapping table in config."
        )

    return new_regime


def get_strategy_for_regime(regime):
    """
    Return the strategy parameters for a given regime number.
    """
    if regime is None:
        return None
    params = cfg.REGIME_TO_STRATEGY.get(int(regime))
    if params is None:
        raise ValueError(
            f"[regime_classifier] regime={regime} not found in REGIME_TO_STRATEGY. "
            f"Valid regimes: 1-20."
        )
    return dict(params)


def get_sl_for_regime(regime):
    """
    Return the stop-loss definition for a given regime number.
    """
    if regime is None:
        return None
    params = cfg.REGIME_TO_SL.get(int(regime))
    if params is None:
        raise ValueError(
            f"[regime_classifier] regime={regime} not found in REGIME_TO_SL. "
            f"Valid regimes: 1-20."
        )
    return dict(params)


def resolve_trade_params(rv_slow, iv, move_to_open_1_std):
    """
    Single-call convenience wrapper.
    """
    rv_zone, iv_zone, move_zone = classify_all_zones(rv_slow, iv, move_to_open_1_std)

    if rv_zone is None or iv_zone is None or move_zone is None:
        return None

    regime = get_regime(rv_slow, iv, move_to_open_1_std)
    strategy = get_strategy_for_regime(regime)
    sl = get_sl_for_regime(regime)

    return {
        "regime": regime,
        "rv_zone": rv_zone,
        "iv_zone": iv_zone,
        "move_zone": move_zone,
        "structure": strategy["structure"],
        "multiplier": strategy["multiplier"],
        "sl_type": sl["sl_type"],
        "sl_value": sl["sl_value"],
        "exit_side": sl["exit_side"],
    }