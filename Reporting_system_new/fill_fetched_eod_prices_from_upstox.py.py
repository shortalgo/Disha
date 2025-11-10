#!/usr/bin/env python3
import sys
import getpass
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text

import upstox_client
from upstox_client.rest import ApiException


# =========================================
# DB CONFIG
# =========================================
DB_USER = "postgres"
DB_PASSWORD = "New@1234"   # TODO: move to env var
DB_HOST = "192.168.18.23"
DB_PORT = 5432
DB_NAME = "postgres"


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
    Normalize underlying names so they match reports.fetched_eod_prices.instrument_name.
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

    return s or None


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
    Build lookup:
        (underlying, opt_type, strike, expiry_str) -> instrument_key
    for F&O options.
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

    # Sanity
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

    # Final validity
    mask_valid = (
        inst_opt["opt_type"].isin(["CE", "PE"])
        & inst_opt["strike"].notna()
        & inst_opt["expiry_str"].notna()
        & inst_opt["underlying"].notna()
    )
    inst_opt_valid = inst_opt[mask_valid].copy()

    print(f"[INFO] Valid option rows after normalization: {len(inst_opt_valid)}")

    if inst_opt_valid.empty:
        print("[ERROR] No valid option rows after normalization.")
        sys.exit(1)

    # Build MultiIndex lookup
    inst_opt_valid = (
        inst_opt_valid[["instrument_key", "underlying", "opt_type", "strike", "expiry_str"]]
        .drop_duplicates(
            subset=["underlying", "opt_type", "strike", "expiry_str", "instrument_key"]
        )
        .set_index(["underlying", "opt_type", "strike", "expiry_str"])
        .sort_index()
    )

    print("[OK] Instrument lookup table ready.")
    return inst_opt_valid


# =========================================
# LOAD ROWS NEEDING PRICES
# =========================================
def load_missing_eod(engine):
    q = """
        SELECT id,
               date,
               instrument_name,
               option_type,
               strike,
               expiry_date
        FROM reports.fetched_eod_prices
        WHERE price IS NULL
        ORDER BY date, instrument_name, option_type, strike, expiry_date, id
    """
    df = pd.read_sql_query(q, engine)
    print(f"[INFO] Found {len(df)} rows needing prices.")
    return df


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

    We key by instrument_token / instrument_key from payload so it matches our mapping.
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
# MAIN: MAP fetched_eod_prices -> Upstox LTP
# =========================================
def main():
    engine = get_engine()
    inst_lookup = load_instruments()

    missing = load_missing_eod(engine)
    if missing.empty:
        print("[OK] Nothing to fill. Exiting.")
        return

    api = get_upstox_client()

    mapped_rows = []
    misses = 0

    # Map DB rows -> Upstox instrument_key via (underlying, opt_type, strike, expiry)
    for _, r in missing.iterrows():
        underlying = normalize_underlying(str(r["instrument_name"]).strip())
        opt_type = str(r["option_type"]).upper().strip()
        strike = float(r["strike"])
        expiry_str = pd.to_datetime(r["expiry_date"]).strftime("%Y-%m-%d")

        if not underlying:
            print(f"[MISS] Invalid underlying for id={r['id']}: {r['instrument_name']}")
            misses += 1
            continue

        key = (underlying, opt_type, strike, expiry_str)

        try:
            matches = inst_lookup.loc[[key]]
            instr_key = matches["instrument_key"].iloc[0]
            mapped_rows.append((r["id"], instr_key))
        except KeyError:
            print(f"[MISS] No instrument match for {key}")
            misses += 1

    print(f"[INFO] Successfully mapped {len(mapped_rows)} of {len(missing)} rows to instrument_key.")
    if misses:
        print(f"[INFO] {misses} rows could not be mapped and will be skipped.")

    if not mapped_rows:
        print("[WARN] No instruments could be mapped. Check naming/strikes/expiries.")
        return

    # Unique instrument_keys for LTP
    id_to_instr = {row_id: ikey for (row_id, ikey) in mapped_rows}
    unique_instr_keys = sorted(set(id_to_instr.values()))
    print(f"[INFO] Fetching LTP for {len(unique_instr_keys)} unique instrument keys...")

    batch_size = 100
    ltp_map = {}
    for i in range(0, len(unique_instr_keys), batch_size):
        batch = unique_instr_keys[i : i + batch_size]
        res = fetch_ltp_batch(api, batch)
        ltp_map.update(res)

    print(f"[INFO] Retrieved LTP for {len(ltp_map)} instruments.")

    # Build update list using instrument_key-based map
    updates = []
    missing_ltp = 0
    for row_id, ikey in id_to_instr.items():
        px = ltp_map.get(ikey)
        if px is not None:
            updates.append((row_id, px))
        else:
            missing_ltp += 1

    print(f"[INFO] Updating {len(updates)} fetched_eod_prices rows with LTP.")
    if missing_ltp:
        print(f"[INFO] {missing_ltp} mapped instruments had no LTP in response (skipped).")

    if not updates:
        print("[WARN] No prices resolved. Nothing to update.")
        return

    # Apply updates in one transaction
    with engine.begin() as conn:
        for row_id, px in updates:
            conn.execute(
                text(
                    """
                    UPDATE reports.fetched_eod_prices
                    SET price = :px,
                        retrieved_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"px": px, "id": row_id},
            )

    print("[OK] Updates committed.")


if __name__ == "__main__":
    main()





#eyJ0eXAiOiJKV1QiLCJrZXlfaWQiOiJza192MS4wIiwiYWxnIjoiSFMyNTYifQ.eyJzdWIiOiI0TEM5VksiLCJqdGkiOiI2OTBjYWUyZDk4MTRmYjM5NTdkNGY1ZTciLCJpc011bHRpQ2xpZW50IjpmYWxzZSwiaXNQbHVzUGxhbiI6ZmFsc2UsImlhdCI6MTc2MjQzODcwMSwiaXNzIjoidWRhcGktZ2F0ZXdheS1zZXJ2aWNlIiwiZXhwIjoxNzYyNDY2NDAwfQ.AOQA-TrpmaOk7dlvHoEpGxsPwJpk5tAhfybjArk4iUg