#!/usr/bin/env python3
import sys
import argparse
import getpass
from urllib.parse import quote_plus
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text

import upstox_client
from upstox_client.rest import ApiException


# =========================================
# DB CONFIG
# =========================================
# Prod DB
DB_USER = "postgres"
DB_PASSWORD = "New@1234"   # TODO: move to env var
DB_HOST = "192.168.18.23"
DB_PORT = 5432
DB_NAME = "postgres"

# Test database (commented)
# DB_USER = "postgres"
# DB_PASSWORD = "New@121"   # TODO: move to env var
# DB_HOST = "localhost"
# DB_PORT = 5432
# DB_NAME = "postgres"


# =========================================
# DB CONNECTION
# =========================================
def get_engine():
    pw = quote_plus(DB_PASSWORD)
    url = f"postgresql+psycopg2://{DB_USER}:{pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    eng = create_engine(url, future=True)
    with eng.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("[OK] DB connection successful.")
    return eng


# =========================================
# HELPERS
# =========================================
def normalize_underlying(raw: str | None) -> str | None:
    """
    Normalize underlying names so they match instrument_name in our EOD table.
    Only NIFTY, SENSEX, BANKNIFTY, FINNIFTY, MIDCPNIFTY are of interest.
    """
    if raw is None:
        return None

    s = str(raw).upper().strip().replace(" ", "")

    if s in ("NIFTY", "NIFTY50"):
        return "NIFTY"
    if s in ("BANKNIFTY", "NIFTYBANK"):
        return "BANKNIFTY"
    if "SENSEX" in s:
        return "SENSEX"
    if "FINNIFTY" in s:
        return "FINNIFTY"
    # MIDCAP NIFTY index – handle correct + common typo
    if s in ("MIDCPNIFTY", "MIDCPNICTY"):
        return "MIDCPNIFTY"

    # anything else is irrelevant for this script
    return None


def parse_opt_type_from_ts(ts: str) -> str | None:
    ts = str(ts).upper().strip()
    if ts.endswith("CE"):
        return "CE"
    if ts.endswith("PE"):
        return "PE"
    return None


def parse_expiry(expiry_val):
    """
    Upstox instruments JSON uses long timestamps (ms since epoch) for derivatives.
    Handle ms, s, or ISO-style strings robustly.
    """
    if pd.isna(expiry_val):
        return None

    if isinstance(expiry_val, (int, float)):
        # Heuristic: > 1e10 => ms
        if expiry_val > 10_000_000_000:
            dt = pd.to_datetime(expiry_val, unit="ms", errors="coerce")
        else:
            dt = pd.to_datetime(expiry_val, unit="s", errors="coerce")
    else:
        dt = pd.to_datetime(expiry_val, errors="coerce")

    if pd.isna(dt):
        return None
    return dt.strftime("%Y-%m-%d")


# =========================================
# LOAD & NORMALIZE UPSTOX INSTRUMENT MASTER
# =========================================
def load_instruments():
    """
    Load all F&O options from Upstox instrument masters and normalize.

    Returns a DataFrame with columns:
        instrument_key, underlying, opt_type, strike, expiry_str
    but **filtered only to NIFTY, SENSEX, BANKNIFTY, FINNIFTY, MIDCPNIFTY** options.
    """
    print("[INFO] Loading Upstox instrument masters...")
    urls = [
        "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
        "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz",
    ]

    frames = []
    for u in urls:
        try:
            df = pd.read_json(u)
            print(f"[OK] Loaded instruments from {u} ({len(df)} rows)")
            frames.append(df)
        except Exception as e:
            print(f"[WARN] Failed to load {u}: {e}")

    if not frames:
        print("[FATAL] Could not load any instrument master.")
        sys.exit(1)

    inst = pd.concat(frames, ignore_index=True)
    print("[DEBUG] Instrument master columns:", list(inst.columns))

    # Sanity check
    required = ["segment", "instrument_type", "instrument_key",
                "trading_symbol", "strike_price", "expiry"]
    for col in required:
        if col not in inst.columns:
            print(f"[FATAL] Required column '{col}' missing in instrument master.")
            sys.exit(1)

    inst["segment"] = inst["segment"].astype(str).str.upper()
    inst["instrument_type"] = inst["instrument_type"].astype(str).str.upper()
    inst["trading_symbol"] = inst["trading_symbol"].astype(str).str.upper()

    # Options: F&O segments + instrument_type CE/PE
    is_opt = inst["segment"].isin(["NSE_FO", "BSE_FO", "MCX_FO"]) & \
             inst["instrument_type"].isin(["CE", "PE"])

    inst_opt = inst[is_opt].copy()
    print(f"[INFO] Detected {len(inst_opt)} option rows before normalization.")

    if inst_opt.empty:
        print("[FATAL] No option instruments detected. Check JSON format.")
        sys.exit(1)

    # Strike
    inst_opt["strike"] = pd.to_numeric(inst_opt["strike_price"], errors="coerce")

    # Fallback: parse strike from trading_symbol if needed
    missing_strike = inst_opt["strike"].isna()
    if missing_strike.any():
        ts = inst_opt.loc[missing_strike, "trading_symbol"]
        parsed = ts.str.extract(r"(\d+(?:\.\d+)?)")[0]
        inst_opt.loc[missing_strike, "strike"] = pd.to_numeric(parsed, errors="coerce")

    # Expiry
    inst_opt["expiry_str"] = inst_opt["expiry"].apply(parse_expiry)

    # Option type (prefer instrument_type, fallback to TS)
    inst_opt["opt_type"] = inst_opt["instrument_type"]
    bad_opt = ~inst_opt["opt_type"].isin(["CE", "PE"])
    if bad_opt.any():
        inst_opt.loc[bad_opt, "opt_type"] = inst_opt.loc[bad_opt, "trading_symbol"].apply(
            parse_opt_type_from_ts
        )

    # Underlying: prefer underlying_symbol; else derive from TS prefix
    if "underlying_symbol" in inst_opt.columns:
        base = inst_opt["underlying_symbol"]
    else:
        base = pd.Series([None] * len(inst_opt), index=inst_opt.index)

    need = base.isna() | (base.astype(str).str.strip() == "")
    if need.any():
        prefix = inst_opt.loc[need, "trading_symbol"].str.extract(r"^([A-Z]+)")[0]
        base = base.copy()
        base.loc[need] = prefix

    inst_opt["underlying"] = base.apply(normalize_underlying)

    # Final validity (only our indices will survive normalize_underlying)
    mask_valid = (
        inst_opt["opt_type"].isin(["CE", "PE"])
        & inst_opt["strike"].notna()
        & inst_opt["expiry_str"].notna()
        & inst_opt["underlying"].notna()
    )
    inst_opt_valid = inst_opt[mask_valid].copy()

    print(f"[INFO] Valid option rows after normalization (before index filter): {len(inst_opt_valid)}")

    if inst_opt_valid.empty:
        print("[ERROR] No valid option rows after normalization.")
        sys.exit(1)

    # Now explicitly filter to the 5 indices
    wanted = {"NIFTY", "SENSEX", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"}
    inst_opt_valid = inst_opt_valid[inst_opt_valid["underlying"].isin(wanted)].copy()

    print(f"[INFO] Rows after filtering to {wanted}: {len(inst_opt_valid)}")

    if inst_opt_valid.empty:
        print("[ERROR] No NIFTY/SENSEX/BANKNIFTY/FINNIFTY/MIDCPNIFTY options found after filtering.")
        sys.exit(1)

    # Return a clean DF, one row per instrument_key/contract
    inst_opt_valid = (
        inst_opt_valid[["instrument_key", "underlying", "opt_type", "strike", "expiry_str"]]
        .drop_duplicates(subset=["instrument_key", "underlying", "opt_type", "strike", "expiry_str"])
        .reset_index(drop=True)
    )

    print("[OK] Instrument option table ready (5 indices).")
    return inst_opt_valid


# =========================================
# UPSTOX CLIENT + LTP FETCH
# =========================================
def get_upstox_client():
    access_token = getpass.getpass("Enter Upstox access token: ").strip()
    if not access_token:
        print("[FATAL] No access token provided.")
        sys.exit(1)

    configuration = upstox_client.Configuration()
    configuration.access_token = access_token

    api_client = upstox_client.ApiClient(configuration)
    api_instance = upstox_client.MarketQuoteApi(api_client)
    print("[OK] Upstox client initialized.")
    return api_instance


def fetch_ltp_batch(api_instance, instrument_keys):
    """
    Fetch LTP for a batch of instrument_keys.

    Upstox ltp() typically returns:
        data = {
          "SOME_SYMBOL": {
            "last_price": ...,
            "instrument_token": "NSE_FO|XXXXX"
          },
          ...
        }
    """
    if not instrument_keys:
        return {}

    symbol = ",".join(instrument_keys)
    api_version = "2.0"

    try:
        resp = api_instance.ltp(symbol, api_version)
        data = resp.to_dict().get("data", {})
    except ApiException as e:
        print(f"[ERROR] Upstox LTP API failed: {e}")
        return {}

    out = {}
    for _, payload in data.items():
        ikey = (
            payload.get("instrument_token")
            or payload.get("instrument_key")
            or None
        )
        ltp = payload.get("last_price")
        if ikey and ltp is not None:
            out[ikey] = float(ltp)

    return out


# =========================================
# MAIN: FETCH LTPs FOR 5 INDICES AND SAVE TO EOD TABLE
# =========================================
def main():
    parser = argparse.ArgumentParser(
        description="Fetch EOD LTP for NIFTY/SENSEX/BANKNIFTY/FINNIFTY/MIDCPNIFTY options and store in core.option_eod_prices_all"
    )
    parser.add_argument(
        "--date",
        help="Trade date (YYYY-MM-DD) for which these marks are considered EOD. Default = today.",
    )
    args = parser.parse_args()

    if args.date:
        try:
            trade_date = date.fromisoformat(args.date)
        except ValueError:
            print("[FATAL] Invalid --date; expected YYYY-MM-DD")
            sys.exit(1)
    else:
        trade_date = date.today()

    print(f"[INFO] Using trade_date = {trade_date}")

    engine = get_engine()
    inst_df = load_instruments()  # already filtered to 5 indices
    api = get_upstox_client()

    # Build LTP map for ALL unique instrument_keys (only for the 5 indices universe)
    unique_instr_keys = sorted(inst_df["instrument_key"].dropna().unique())
    print(f"[INFO] Fetching LTP for {len(unique_instr_keys)} unique option instrument keys...")

    batch_size = 100
    ltp_map = {}
    for i in range(0, len(unique_instr_keys), batch_size):
        batch = unique_instr_keys[i: i + batch_size]
        print(f"[INFO] LTP batch {i}..{i + len(batch) - 1}")
        res = fetch_ltp_batch(api, batch)
        ltp_map.update(res)

    print(f"[INFO] Retrieved LTP for {len(ltp_map)} instruments.")

    # Attach prices to DF
    inst_df["price"] = inst_df["instrument_key"].map(ltp_map)
    df_has_price = inst_df.dropna(subset=["price"]).copy()

    print(f"[INFO] {len(df_has_price)} option rows have a valid LTP and will be inserted/updated.")

    if df_has_price.empty:
        print("[WARN] No prices resolved. Nothing to store.")
        return

    # Insert/Upsert into core.option_eod_prices_all
    insert_sql = text("""
        INSERT INTO core.option_eod_prices_all (
            trade_date,
            instrument_name,
            option_type,
            strike,
            expiry_date,
            price,
            source,
            instrument_key,
            retrieved_at
        )
        VALUES (
            :trade_date,
            :instrument_name,
            :option_type,
            :strike,
            :expiry_date,
            :price,
            'upstox_ltp',
            :instrument_key,
            NOW()
        )
        ON CONFLICT (trade_date, instrument_name, option_type, strike, expiry_date)
        DO UPDATE SET
            price         = EXCLUDED.price,
            source        = EXCLUDED.source,
            instrument_key = EXCLUDED.instrument_key,
            retrieved_at  = EXCLUDED.retrieved_at;
    """)

    with engine.begin() as conn:
        rows = 0
        for _, r in df_has_price.iterrows():
            conn.execute(
                insert_sql,
                {
                    "trade_date": trade_date,
                    "instrument_name": r["underlying"],  # NIFTY/SENSEX/BANKNIFTY/FINNIFTY/MIDCPNIFTY
                    "option_type": r["opt_type"],
                    "strike": float(r["strike"]),
                    "expiry_date": r["expiry_str"],
                    "price": float(r["price"]),
                    "instrument_key": r["instrument_key"],
                },
            )
            rows += 1

    print(f"[OK] Inserted/updated {rows} rows into core.option_eod_prices_all for {trade_date}.")


if __name__ == "__main__":
    main()
