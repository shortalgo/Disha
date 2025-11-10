#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end: DIM sync + Slippage/PnL ETL + Error flags

Highlights:
- Standard columns everywhere (instrument_name, instrument_type, option_type, strike)
  with option_type CHECK ('CE','PE','NA'), NOT NULL, default 'NA'.
- instrument_type derived consistently: 'OPTION' for CE/PE, else 'UNKNOWN'.
- If day_flow vs EOD mismatch → flag once per (day, acct/strat/user/variant);
  aggregate all strikes into strikes_csv; exclude those buckets from EOD.
  Ignore buckets already flagged in ops.erroneous_trades.
- CIN count QC; staging dedup to avoid ON CONFLICT multi-hit; per-day DATE binds.
"""

import argparse
from datetime import date, timedelta, datetime, timezone, time as dt_time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ----------------- CONFIG -----------------
SOURCE_DB_URL = "postgresql+psycopg2://disha:zxcvbnm@192.168.18.18:5432/algo_department"
REPORT_DB_URL = "postgresql+psycopg2://disha:zxcvbnm@192.168.18.18:5432/report"
MANAGE_SCHEMA_EXTERNALLY = False
# -----------------------------------------

IST_TZ = "Asia/Kolkata"
VALID_OPT = {"CE", "PE", "NA"}

# ===================== Parsing (IST-safe) =====================
def _parse_date_india_first(x) -> pd.Timestamp:
    if pd.isna(x): return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)): return pd.Timestamp(x)
    if isinstance(x, date): return pd.Timestamp(datetime(x.year, x.month, x.day))
    s = str(x).strip()
    if not s: return pd.NaT
    for fmt in ("%d-%m-%Y","%d/%m/%Y","%d.%m.%Y","%d-%m-%y","%d/%m/%y","%Y-%m-%d","%Y/%m/%d","%Y.%m.%d"):
        try: return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception: pass
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def _parse_time_safe(x) -> dt_time:
    if pd.isna(x): return dt_time(0,0,0)
    if isinstance(x, (pd.Timestamp, datetime)): return x.time()
    s = str(x).strip().replace("  "," ")
    for fmt in ("%H:%M:%S","%H:%M","%I:%M %p","%I:%M:%S %p"):
        try: return datetime.strptime(s, fmt).time()
        except Exception: pass
    try:
        t = pd.to_datetime(s, errors="coerce").time()
        return t or dt_time(0,0,0)
    except Exception:
        return dt_time(0,0,0)

def to_ts_ist(dates, times, tz=IST_TZ):
    ds, ts = pd.Series(dates, copy=False), pd.Series(times, copy=False)
    d_parsed, t_parsed = ds.apply(_parse_date_india_first), ts.apply(_parse_time_safe)

    def _combine(d, t):
        if pd.isna(d): return pd.NaT
        dt_naive = datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
        return pd.Timestamp(dt_naive).tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

    return pd.Series([_combine(d, t) for d, t in zip(d_parsed, t_parsed)],
                     dtype=f"datetime64[ns, {tz}]")

# ===================== Canon helpers ==========================
def canon_option_type(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): return "NA"
    s = str(x).strip().upper()
    if s in ("", "NONE", "NULL", "NAN"): return "NA"
    if s in ("CE","CALL"): return "CE"
    if s in ("PE","PUT"):  return "PE"
    return "NA"

def parse_option_type(s):
    if not isinstance(s, str): return None
    u = s.upper()
    if "CE" in u: return "CE"
    if "PE" in u: return "PE"
    return None

def canon_ids(df, cols=("account_id","strategy_id","user_id","strategy_variant_id")):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df

def canon_strike(s): return pd.to_numeric(s, errors="coerce").round(2)

def infer_instrument_type(option_type: str, instrument_name: str | None = None) -> str:
    ot = canon_option_type(option_type)
    if ot in {"CE", "PE"}:
        return "OPTION"
    return "UNKNOWN"

# ===================== FIFO PnL =================================
def fifo_realized_pnl(group_df):
    g = group_df.sort_values("timestamp_ist").copy()
    fills, realized = len(g), 0.0
    open_lots, exit_without_open = [], False
    for _, r in g.iterrows():
        qty = float(r["qty"]); price = float(r["trade_price"])
        if qty == 0 or np.isnan(qty) or np.isnan(price): continue
        sign, remaining = (1.0 if qty > 0 else -1.0), abs(qty)
        if not open_lots or all(l["sign"] == sign for l in open_lots):
            open_lots.append({"sign":sign,"qty":remaining,"price":price,"time":r["timestamp_ist"]}); continue
        i, touched = 0, False
        while remaining > 1e-12 and i < len(open_lots):
            lot = open_lots[i]
            if lot["sign"] == sign: i += 1; continue
            touched = True
            matched = min(remaining, lot["qty"])
            realized += matched * (price - lot["price"]) * lot["sign"]
            lot["qty"] -= matched; remaining -= matched
            if lot["qty"] <= 1e-12: open_lots.pop(i)
            else: i += 1
        if not touched and abs(qty) > 0 and sign != (open_lots[0]["sign"] if open_lots else sign):
            exit_without_open = True
        if remaining > 1e-12:
            open_lots.append({"sign":sign,"qty":remaining,"price":price,"time":r["timestamp_ist"]})
    if not open_lots: return realized, fills, 0.0, None, exit_without_open
    net_qty = sum(l["sign"]*l["qty"] for l in open_lots)
    if abs(net_qty) <= 1e-12: return realized, fills, 0.0, None, exit_without_open
    dom_sign = 1.0 if net_qty > 0 else -1.0
    dom_qty  = sum(l["qty"] for l in open_lots if l["sign"] == dom_sign)
    avg_entry = (sum(l["price"]*l["qty"] for l in open_lots if l["sign"]==dom_sign)/dom_qty) if dom_qty>0 else None
    return realized, fills, net_qty, avg_entry, exit_without_open

# ================== DDL ==================
DDL_DIM = """
CREATE SCHEMA IF NOT EXISTS dim;

CREATE TABLE IF NOT EXISTS dim.portfolios (
  id bigint PRIMARY KEY, portfolio text NOT NULL, mode text NOT NULL,
  updated_at timestamptz DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_dim_portfolios_name ON dim.portfolios(portfolio);

CREATE TABLE IF NOT EXISTS dim.strategies (
  id bigint PRIMARY KEY, strategy_name text NOT NULL, portfolio_id bigint NOT NULL REFERENCES dim.portfolios(id),
  updated_at timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_dim_strategies_portfolio ON dim.strategies(portfolio_id);

CREATE TABLE IF NOT EXISTS dim.strategy_variants (
  id bigint PRIMARY KEY, strategy_id bigint NOT NULL REFERENCES dim.strategies(id),
  variant_name text NOT NULL, updated_at timestamptz DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_dim_variants_strategy ON dim.strategy_variants(strategy_id);

CREATE TABLE IF NOT EXISTS dim.accounts (
  account_id bigint PRIMARY KEY, name text NOT NULL, updated_at timestamptz DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_dim_accounts_name ON dim.accounts(name);

CREATE TABLE IF NOT EXISTS dim.users (
  user_id bigint PRIMARY KEY, name text NOT NULL, updated_at timestamptz DEFAULT now()
);

CREATE TABLE IF NOT EXISTS dim.account_assignments (
  account_id bigint NOT NULL REFERENCES dim.accounts(account_id),
  user_id bigint NOT NULL REFERENCES dim.users(user_id),
  valid_from date NOT NULL DEFAULT DATE '1970-01-01', valid_to date,
  PRIMARY KEY (account_id, user_id, valid_from)
);
CREATE INDEX IF NOT EXISTS ix_dim_acc_assign_user ON dim.account_assignments(user_id);
CREATE INDEX IF NOT EXISTS ix_dim_acc_assign_acc  ON dim.account_assignments(account_id);
"""

DDL_TRADES_ALL = """
CREATE SCHEMA IF NOT EXISTS mart;

CREATE TABLE IF NOT EXISTS mart.trades_all (
  unique_id text PRIMARY KEY,
  timestamp_ist timestamptz,
  trade_day date,
  account_id text,
  strategy_id text,
  user_id text,
  strategy_variant_id text,
  instrument_name text,
  instrument_type text DEFAULT 'UNKNOWN' NOT NULL,
  type text,
  option_type text DEFAULT 'NA' NOT NULL,
  strike double precision,
  qty double precision,
  trade_price double precision,
  side text,
  entry_label text,
  theoretical_price double precision,
  exec_mode text,
  account_name text,
  strategy_name text,
  portfolio_name text,
  exit_reason text
);
"""

DDL_MART_OPS_TABLES = """
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS mart.trade_pnl (
  trade_day date,
  account_id text, strategy_id text, user_id text, strategy_variant_id text,
  instrument_name text,
  instrument_type text DEFAULT 'UNKNOWN' NOT NULL,
  option_type text DEFAULT 'NA' NOT NULL,
  strike double precision,
  account_name text, strategy_name text, portfolio_name text, exec_mode text,
  realized_pnl double precision, fills int,
  PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, instrument_type, option_type, strike)
);

CREATE TABLE IF NOT EXISTS mart.open_positions_eod (
  trade_day date,
  account_id text, strategy_id text, user_id text, strategy_variant_id text,
  instrument_name text,
  instrument_type text DEFAULT 'UNKNOWN' NOT NULL,
  option_type text DEFAULT 'NA' NOT NULL,
  strike double precision,
  account_name text, strategy_name text, portfolio_name text, exec_mode text,
  net_qty double precision, avg_entry_price double precision, last_trade_ts timestamptz,
  PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, instrument_type, option_type, strike)
);

CREATE TABLE IF NOT EXISTS mart.slippage_events (
  unique_key text PRIMARY KEY,
  timestamp_ist timestamptz, trade_day date,
  user_id text, account_name text, strategy_name text,
  instrument_name text,
  instrument_type text DEFAULT 'UNKNOWN' NOT NULL,
  option_type text DEFAULT 'NA' NOT NULL,
  side text, trade_price double precision, theoretical_price double precision,
  slip_abs double precision, slip_pct double precision,
  exec_mode text, exit_reason text
);

/* RAW error table — PK does NOT include option_type/strike; we store strikes_csv */
CREATE TABLE IF NOT EXISTS ops.erroneous_trades (
  trade_day date,
  account_id text, strategy_id text, user_id text, strategy_variant_id text,
  issue_code text,
  issue_detail text,
  affected_unique_ids text[],
  first_ts timestamptz, last_ts timestamptz,
  created_at_utc timestamptz,
  instrument_type text DEFAULT 'UNKNOWN' NOT NULL,
  option_type text DEFAULT 'NA' NOT NULL,
  strike double precision,
  strikes_csv text,
  PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
);

CREATE TABLE IF NOT EXISTS ops.erroneous_trades_clean (
  trade_day date,
  account_id text, account_name text,
  strategy_id text, strategy_name text, portfolio_name text, exec_mode text,
  user_id text, strategy_variant_id text,
  issue_code text, issue_detail text,
  wrong_fills int,
  strikes_csv text,
  affected_ids_csv text,
  first_ts timestamptz, last_ts timestamptz,
  created_at_utc timestamptz,
  instrument_type text DEFAULT 'UNKNOWN' NOT NULL,
  option_type text DEFAULT 'NA' NOT NULL
);
"""

DDL_MART_OPS_VIEWS_DROP = """
DROP VIEW IF EXISTS ops.v_erroneous_trades_clean CASCADE;
DROP VIEW IF EXISTS ops.v_erroneous_counts_by_bucket CASCADE;
DROP VIEW IF EXISTS mart.v_slippage_daily CASCADE;
DROP VIEW IF EXISTS mart.v_strategy_pnl CASCADE;
DROP VIEW IF EXISTS mart.v_open_positions_eod CASCADE;
"""

DDL_MART_OPS_VIEWS_CREATE = """
CREATE VIEW ops.v_erroneous_trades_clean AS
SELECT trade_day, portfolio_name, strategy_name, account_name, exec_mode,
       strategy_variant_id, issue_code,
       wrong_fills, strikes_csv, affected_ids_csv,
       first_ts, last_ts, issue_detail
FROM ops.erroneous_trades_clean
ORDER BY trade_day DESC, portfolio_name, strategy_name, issue_code;

CREATE VIEW ops.v_erroneous_counts_by_bucket AS
SELECT trade_day, portfolio_name, strategy_name, exec_mode,
       COUNT(*) AS bad_trades, SUM(wrong_fills) AS bad_fills
FROM ops.erroneous_trades_clean
GROUP BY 1,2,3,4
ORDER BY 1 DESC, 2,3,4;

CREATE VIEW mart.v_slippage_daily AS
SELECT trade_day, account_name, strategy_name,
       COUNT(*) AS fills, AVG(slip_pct) AS avg_slip_pct
FROM mart.slippage_events
GROUP BY 1,2,3;

CREATE VIEW mart.v_strategy_pnl AS
SELECT trade_day, portfolio_name, strategy_name, exec_mode,
       SUM(realized_pnl) AS realized_pnl, SUM(fills) AS fills
FROM mart.trade_pnl
GROUP BY 1,2,3,4;

CREATE VIEW mart.v_open_positions_eod AS
SELECT trade_day, portfolio_name, strategy_name, exec_mode,
       strategy_variant_id, instrument_type, option_type, strike,
       SUM(net_qty) AS net_qty, AVG(avg_entry_price) AS avg_entry_price
FROM mart.open_positions_eod
GROUP BY 1,2,3,4,5,6,7,8;
"""

DDL_INDEXES_PERF = """
CREATE INDEX IF NOT EXISTS ix_slip_day            ON mart.slippage_events(trade_day);
CREATE INDEX IF NOT EXISTS ix_slip_strategy_day   ON mart.slippage_events(strategy_name, trade_day);

DROP INDEX IF EXISTS mart.ix_pnl_all_keys;
CREATE INDEX IF NOT EXISTS ix_pnl_all_keys ON mart.trade_pnl
  (trade_day, account_id, strategy_id, user_id, strategy_variant_id, instrument_type, option_type, strike);

CREATE INDEX IF NOT EXISTS ix_pnl_day_strategy ON mart.trade_pnl(trade_day, strategy_id);
CREATE INDEX IF NOT EXISTS ix_pnl_day_account  ON mart.trade_pnl(trade_day, account_id);

DROP INDEX IF EXISTS mart.ix_pnl_day_user;
CREATE INDEX IF NOT EXISTS ix_pnl_day_user     ON mart.trade_pnl(trade_day, user_id);

CREATE INDEX IF NOT EXISTS ix_open_day         ON mart.open_positions_eod(trade_day);
CREATE INDEX IF NOT EXISTS ix_err_day          ON ops.erroneous_trades(trade_day);
"""

# ================== Migrations / constraints ==================
def ensure_columns_and_constraints(dst_engine):
    if MANAGE_SCHEMA_EXTERNALLY: return
    with dst_engine.begin() as con:
        targets = [
            ("mart","trades_all"),
            ("mart","trade_pnl"),
            ("mart","open_positions_eod"),
            ("mart","slippage_events"),
            ("ops","erroneous_trades"),
            ("ops","erroneous_trades_clean"),
        ]
        for schema, table in targets:
            con.execute(text(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS instrument_name text;"))
            con.execute(text(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS instrument_type text;"))
            con.execute(text(f"ALTER TABLE {schema}.{table} ALTER COLUMN instrument_type SET DEFAULT 'UNKNOWN';"))
            con.execute(text(f"UPDATE {schema}.{table} SET instrument_type='UNKNOWN' WHERE instrument_type IS NULL OR instrument_type=''"))

            con.execute(text(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS option_type text;"))
            con.execute(text(f"ALTER TABLE {schema}.{table} ALTER COLUMN option_type SET DEFAULT 'NA';"))
            con.execute(text(f"UPDATE {schema}.{table} SET option_type='NA' WHERE option_type IS NULL OR option_type=''"))

            con.execute(text(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS strike float8;"))

            # Check constraint for option_type
            con.execute(text(f"""
            DO $$
            BEGIN
              IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = '{schema}_{table}_chk_option_type'
              ) THEN
                ALTER TABLE {schema}.{table}
                ADD CONSTRAINT {schema}_{table}_chk_option_type
                CHECK (option_type IN ('CE','PE','NA'));
              END IF;
            END$$;"""))

            # NOT NULL for instrument_type/option_type (keep strike nullable)
            con.execute(text(f"ALTER TABLE {schema}.{table} ALTER COLUMN instrument_type SET NOT NULL;"))
            con.execute(text(f"ALTER TABLE {schema}.{table} ALTER COLUMN option_type SET NOT NULL;"))

            if schema == "ops" and table == "erroneous_trades":
                con.execute(text(f"ALTER TABLE {schema}.{table} ADD COLUMN IF NOT EXISTS strikes_csv text;"))

# ================== Schema ensure ==================
def ensure_dim_schema(dst_engine):
    if MANAGE_SCHEMA_EXTERNALLY: return
    with dst_engine.begin() as con:
        con.execute(text(DDL_DIM))

def ensure_mart_ops_schema(dst_engine):
    if MANAGE_SCHEMA_EXTERNALLY: return
    # 1) create tables
    with dst_engine.begin() as con:
        con.execute(text(DDL_TRADES_ALL))
        con.execute(text(DDL_MART_OPS_TABLES))
    # 2) ensure columns/constraints exist (so indexes can reference them)
    ensure_columns_and_constraints(dst_engine)
    # 3) drop views → create indexes → create views
    with dst_engine.begin() as con:
        con.execute(text(DDL_MART_OPS_VIEWS_DROP))
        con.execute(text(DDL_INDEXES_PERF))
        con.execute(text(DDL_MART_OPS_VIEWS_CREATE))

# ================== DIM sync (SOURCE → REPORT) ==================
def fetch_strategy_variants_df(engine):
    candidate_sql = [
        "SELECT id::bigint AS id, strategy_id::bigint AS strategy_id, variant::text AS variant_name FROM core.strategy_variants",
        "SELECT id::bigint AS id, strategy_id::bigint AS strategy_id, name::text    AS variant_name FROM core.strategy_variants",
        "SELECT id::bigint AS id, strategy_id::bigint AS strategy_id, variant_name::text AS variant_name FROM core.strategy_variants",
    ]
    last_err = None
    for sql in candidate_sql:
        try:
            with engine.connect() as con:
                return pd.read_sql(text(sql), con)
        except Exception as e:
            last_err = e
    raise last_err

def sync_dims(src_engine, dst_engine):
    def stage_and_upsert(df: pd.DataFrame, stage_fq: str, upsert_sql: str):
        if df is None or df.empty: return
        with dst_engine.begin() as con:
            schema, table = stage_fq.split(".")
            df.to_sql(table, con, schema=schema, if_exists="replace", index=False)
            con.execute(text(upsert_sql))
            con.execute(text(f"DROP TABLE {stage_fq};"))

    with src_engine.connect() as con:
        df_port = pd.read_sql(text("""
            SELECT id::bigint AS id, portfolio::text AS portfolio,
                   COALESCE(NULLIF(mode::text,''),'unknown') AS mode
            FROM core.portfolios
        """), con)
    stage_and_upsert(df_port, "dim._stage_portfolios", """
        INSERT INTO dim.portfolios (id, portfolio, mode, updated_at)
        SELECT id, portfolio, mode, now() FROM dim._stage_portfolios
        ON CONFLICT (id) DO UPDATE
        SET portfolio=EXCLUDED.portfolio, mode=EXCLUDED.mode, updated_at=now();
    """)

    with src_engine.connect() as con:
        df_strat = pd.read_sql(text("""
            SELECT id::bigint AS id, strategy_name::text AS strategy_name, portfolio_id::bigint AS portfolio_id
            FROM core.strategies
        """), con)
    stage_and_upsert(df_strat, "dim._stage_strategies", """
        INSERT INTO dim.strategies (id, strategy_name, portfolio_id, updated_at)
        SELECT id, strategy_name, portfolio_id, now() FROM dim._stage_strategies
        ON CONFLICT (id) DO UPDATE
        SET strategy_name=EXCLUDED.strategy_name, portfolio_id=EXCLUDED.portfolio_id, updated_at=now();
    """)

    df_var = fetch_strategy_variants_df(src_engine)
    stage_and_upsert(df_var, "dim._stage_strategy_variants", """
        INSERT INTO dim.strategy_variants (id, strategy_id, variant_name, updated_at)
        SELECT id, strategy_id, variant_name, now() FROM dim._stage_strategy_variants
        ON CONFLICT (id) DO UPDATE
        SET strategy_id=EXCLUDED.strategy_id, variant_name=EXCLUDED.variant_name, updated_at=now();
    """)

    with src_engine.connect() as con:
        df_acct = pd.read_sql(text("SELECT id::bigint AS account_id, name::text AS name FROM core.accounts"), con)
    stage_and_upsert(df_acct, "dim._stage_accounts", """
        INSERT INTO dim.accounts (account_id, name, updated_at)
        SELECT account_id, name, now() FROM dim._stage_accounts
        ON CONFLICT (account_id) DO UPDATE SET name=EXCLUDED.name, updated_at=now();
    """)

    with src_engine.connect() as con:
        df_users = pd.read_sql(text("SELECT id_no::bigint AS user_id, name::text AS name FROM core.users"), con)
    stage_and_upsert(df_users, "dim._stage_users", """
        INSERT INTO dim.users (user_id, name, updated_at)
        SELECT user_id, name, now() FROM dim._stage_users
        ON CONFLICT (user_id) DO UPDATE SET name=EXCLUDED.name, updated_at=now();
    """)

    with src_engine.connect() as con:
        df_assign = pd.read_sql(text("""
            SELECT DISTINCT account_id::bigint AS account_id, user_id::bigint AS user_id,
                            NULL::date AS valid_from, NULL::date AS valid_to
            FROM core.account_assignments
        """), con)
    stage_and_upsert(df_assign, "dim._stage_account_assignments", """
        INSERT INTO dim.account_assignments (account_id, user_id, valid_from, valid_to)
        SELECT account_id, user_id,
               COALESCE(NULLIF(valid_from::text,'')::date, DATE '1970-01-01'),
               NULLIF(valid_to::text,'')::date
        FROM dim._stage_account_assignments
        ON CONFLICT (account_id, user_id, valid_from) DO NOTHING;
    """)

# ================== CIN builder (per-day, from prior EOD) ==================
def build_carry_ins_for_day(dst_engine, trade_day_str: str) -> pd.DataFrame:
    d = pd.to_datetime(trade_day_str).date()
    prev_day = d - timedelta(days=1)
    with dst_engine.connect() as con:
        df_prev = pd.read_sql(text("""
            SELECT account_id::text, strategy_id::text, user_id::text,
                   strategy_variant_id::text,
                   instrument_name::text,
                   COALESCE(NULLIF(option_type,''),'NA') AS option_type,
                   strike::float8 AS strike,
                   net_qty::float8 AS net_qty,
                   avg_entry_price::float8 AS avg_entry_price,
                   exec_mode, account_name, strategy_name, portfolio_name
            FROM mart.open_positions_eod
            WHERE trade_day = :d AND COALESCE(net_qty,0) <> 0
        """), con, params={"d": prev_day})

    if df_prev.empty: return pd.DataFrame()

    pk = ["account_id","strategy_id","user_id","strategy_variant_id","option_type","strike"]
    df_prev = (df_prev.groupby(pk + ["instrument_name"], as_index=False, dropna=False)
               .agg({"net_qty":"sum","avg_entry_price":"last","exec_mode":"last",
                     "account_name":"last","strategy_name":"last","portfolio_name":"last"}))

    start_ts = pd.Timestamp(f"{d} 00:00:00").tz_localize(IST_TZ)

    def _strike_text(val: float) -> str:
        if pd.isna(val): return "NA"
        i = int(val); return str(i) if abs(val - i) < 1e-9 else str(val)

    cin = pd.DataFrame({
        "unique_id": [f"CIN:{d}:{r.account_id}:{r.strategy_id}:{r.user_id}:{r.strategy_variant_id}:{canon_option_type(r.option_type)}:{_strike_text(float(r.strike))}"
                      for r in df_prev.itertuples(index=False)],
        "instrument_name": df_prev["instrument_name"],
        "instrument_type": df_prev["option_type"].apply(lambda x: infer_instrument_type(x)),
        "type": "CARRY_IN",
        "option_type": df_prev["option_type"].apply(canon_option_type),
        "entry_exit_error": "CARRY_IN",
        "theoretical_price": np.nan,
        "theoretical_time": None,
        "timestamp_ist": start_ts,
        "trade_day": d,
        "account_id": df_prev["account_id"],
        "strategy_id": df_prev["strategy_id"],
        "user_id": df_prev["user_id"],
        "strategy_variant_id": df_prev["strategy_variant_id"],
        "strike": df_prev["strike"].astype(float),
        "qty": df_prev["net_qty"].astype(float),
        "trade_price": df_prev["avg_entry_price"].astype(float),
        "side": np.where(df_prev["net_qty"] > 0, "BUY", np.where(df_prev["net_qty"] < 0, "SELL", None)),
        "entry_label": "CARRY_IN",
        "exec_mode": df_prev["exec_mode"],
        "account_name": df_prev["account_name"],
        "strategy_name": df_prev["strategy_name"],
        "portfolio_name": df_prev["portfolio_name"],
        "exit_reason": "CARRY_IN",
    })
    cin = cin[(cin["qty"].notna()) & (cin["qty"] != 0)]
    return cin

# ================== Trade ETL (per-day) ==================
def run_trade_etl(src_engine, dst_engine, from_date: str, to_date: str):
    days = pd.date_range(pd.to_datetime(from_date).date(),
                         pd.to_datetime(to_date).date(), freq="D")

    for d_ts in days:
        day_date = d_ts.date()
        day_str = day_date.isoformat()

        # ---------- Pull trades for THIS DAY ----------
        with src_engine.connect() as con:
            df = pd.read_sql(text("""
                SELECT t.unique_id, t.instrument_name, t.trade_date, t.trade_time, t.trade_price,
                       t.type, t.qty, t.strike, t.strategy_variant_id, t.entry_exit_error,
                       t.theoretical_price, t.theoretical_time, t.reason_exit AS exit_reason,
                       t.account_id::text AS account_id, t.strategy_id::text AS strategy_id, t.user_id::text AS user_id,
                       a.name AS account_name, s.strategy_name, s.portfolio_id,
                       p.portfolio AS portfolio_name, p.mode AS exec_mode, u.name AS user_name
                FROM core.trades t
                LEFT JOIN core.accounts   a ON a.id = t.account_id
                LEFT JOIN core.strategies s ON s.id = t.strategy_id
                LEFT JOIN core.portfolios p ON p.id = s.portfolio_id
                LEFT JOIN core.users      u ON u.id_no = t.user_id
                WHERE t.trade_date = :d
            """), con, params={"d": day_date})

        if df.empty:
            df = pd.DataFrame(columns=[
                "unique_id","instrument_name","trade_date","trade_time","trade_price","type",
                "qty","strike","strategy_variant_id","entry_exit_error","theoretical_price",
                "theoretical_time","exit_reason","account_id","strategy_id","user_id",
                "account_name","strategy_name","portfolio_id","portfolio_name","exec_mode","user_name"
            ])

        # Normalize / canonicalize
        for c in ["qty","trade_price","theoretical_price","strike"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp_ist"] = to_ts_ist(df["trade_date"], df["trade_time"], tz=IST_TZ)
        df["trade_day"]     = df["timestamp_ist"].dt.tz_convert(IST_TZ).dt.date
        df["option_type"]   = df["type"].apply(parse_option_type).apply(canon_option_type)
        df["instrument_type"] = df.apply(lambda r: infer_instrument_type(r["option_type"], r.get("instrument_name")), axis=1)
        df = canon_ids(df, ["account_id","strategy_id","user_id","strategy_variant_id"])
        df["strike"] = canon_strike(df["strike"])
        sign = np.sign(df["qty"].astype(float))
        df["side"] = np.where(sign>0,"BUY", np.where(sign<0,"SELL", None))
        df["entry_label"] = df.get("entry_label", df["entry_exit_error"])
        df["entry_label"] = df["entry_label"].astype(str).str.strip().replace({"": None})

        # ---------- CARRY-IN (from EOD D-1) ----------
        cin_df = build_carry_ins_for_day(dst_engine, day_str)
        if not cin_df.empty:
            for col in df.columns:
                if col not in cin_df.columns: cin_df[col] = None
            for col in cin_df.columns:
                if col not in df.columns: df[col] = None
            cols_union = list(df.columns)
            df = pd.concat([cin_df[cols_union], df[cols_union]], ignore_index=True, sort=False)

        # ---------- Slippage per fill ----------
        theo_ok = df["theoretical_price"].notna() & (df["theoretical_price"] > 0)
        sign    = np.sign(df["qty"].astype(float))
        df["slip_abs"] = np.where(theo_ok, (df["trade_price"] - df["theoretical_price"]) * sign, np.nan)
        df["slip_pct"] = np.where(theo_ok, 100.0 * df["slip_abs"] / df["theoretical_price"], np.nan)
        exit_reason_lc = df.get("exit_reason").astype(str).str.strip().str.lower()
        manual_mask = theo_ok & (exit_reason_lc == "manual")
        df.loc[manual_mask, ["slip_abs","slip_pct"]] = 0.0

        # ---------- Write transformed trades (stage → casted INSERT) ----------
        with dst_engine.begin() as con:
            df_raw = df.copy()
            cols = ["unique_id","timestamp_ist","trade_day","account_id","strategy_id","user_id",
                    "strategy_variant_id","instrument_name","instrument_type","type","option_type","strike","qty",
                    "trade_price","side","entry_label","theoretical_price","exec_mode",
                    "account_name","strategy_name","portfolio_name","exit_reason"]
            for c in cols:
                if c not in df_raw.columns: df_raw[c] = None
            df_raw = df_raw[cols]
            df_raw = df_raw[df_raw["unique_id"].notna()]
            df_raw["trade_day"] = pd.to_datetime(df_raw["trade_day"], errors="coerce").dt.date
            for c in ["account_id","strategy_id","user_id","strategy_variant_id","instrument_type","option_type","instrument_name","type","side","entry_label","exec_mode","account_name","strategy_name","portfolio_name","exit_reason"]:
                df_raw[c] = df_raw[c].astype(object)
            for c in ["strike","qty","trade_price","theoretical_price"]:
                df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
            df_raw.to_sql("_trades_all_stage", con, schema="mart", if_exists="replace", index=False)
            con.execute(text("""
                INSERT INTO mart.trades_all
                (unique_id, timestamp_ist, trade_day, account_id, strategy_id, user_id,
                 strategy_variant_id, instrument_name, instrument_type, type, option_type, strike, qty,
                 trade_price, side, entry_label, theoretical_price, exec_mode,
                 account_name, strategy_name, portfolio_name, exit_reason)
                SELECT
                  unique_id::text,
                  timestamp_ist::timestamptz,
                  trade_day::date,
                  account_id::text,
                  strategy_id::text,
                  user_id::text,
                  strategy_variant_id::text,
                  instrument_name::text,
                  instrument_type::text,
                  type::text,
                  option_type::text,
                  strike::float8,
                  qty::float8,
                  trade_price::float8,
                  side::text,
                  entry_label::text,
                  theoretical_price::float8,
                  exec_mode::text,
                  account_name::text,
                  strategy_name::text,
                  portfolio_name::text,
                  exit_reason::text
                FROM mart._trades_all_stage
                ON CONFLICT (unique_id) DO UPDATE SET
                  timestamp_ist     = EXCLUDED.timestamp_ist,
                  trade_day         = EXCLUDED.trade_day,
                  instrument_name   = EXCLUDED.instrument_name,
                  instrument_type   = EXCLUDED.instrument_type,
                  qty               = EXCLUDED.qty,
                  trade_price       = EXCLUDED.trade_price,
                  side              = EXCLUDED.side,
                  theoretical_price = EXCLUDED.theoretical_price,
                  exec_mode         = EXCLUDED.exec_mode,
                  account_name      = EXCLUDED.account_name,
                  strategy_name     = EXCLUDED.strategy_name,
                  portfolio_name    = EXCLUDED.portfolio_name,
                  exit_reason       = EXCLUDED.exit_reason,
                  option_type       = EXCLUDED.option_type,
                  strike            = EXCLUDED.strike;
                DROP TABLE mart._trades_all_stage;
            """))

        # ---------- CLUBBING (VWAP) ----------
        club_keys = ["timestamp_ist","trade_day",
                     "account_id","strategy_id","user_id","strategy_variant_id",
                     "instrument_name","instrument_type","type","option_type","strike","exec_mode",
                     "account_name","strategy_name","portfolio_name"]
        if not df.empty:
            tmp = df.copy()
            tmp["abs_qty"] = tmp["qty"].abs()
            tmp["px_w"]    = tmp["trade_price"] * tmp["abs_qty"]
            df_agg = (tmp.groupby(club_keys, dropna=False, as_index=False)
                          .agg(qty=("qty","sum"), w=("abs_qty","sum"), pxw=("px_w","sum")))
            df_agg["trade_price"] = np.where(df_agg["w"]>0, df_agg["pxw"]/df_agg["w"], np.nan)
            df_agg.drop(columns=["w","pxw"], inplace=True)
            df_agg["side"] = np.where(df_agg["qty"]>0, "BUY", np.where(df_agg["qty"]<0,"SELL",None))
        else:
            df_agg = df

        # ---------- Error detection (EXIT_WITHOUT_OPEN) ----------
        err_rows, now_ts = [], datetime.now(timezone.utc)
        group_keys_err = ["trade_day","account_id","strategy_id","user_id","strategy_variant_id","instrument_type","option_type","strike"]
        for _, gdf in df_agg.groupby(group_keys_err, dropna=False):
            realized, fills, net_qty, avg_entry, exit_without_open = fifo_realized_pnl(gdf)
            if exit_without_open:
                err_rows.append({
                    "trade_day": gdf["trade_day"].iloc[0],
                    "account_id": gdf["account_id"].iloc[0],
                    "strategy_id": gdf["strategy_id"].iloc[0],
                    "user_id": gdf["user_id"].iloc[0],
                    "strategy_variant_id": gdf["strategy_variant_id"].iloc[0],
                    "issue_code": "EXIT_WITHOUT_OPEN_POSITION",
                    "issue_detail": "Detected a closing direction when no opposite open lots existed (FIFO).",
                    "affected_unique_ids": list(map(str, gdf.get("unique_id", pd.Series(dtype=str)).tolist())),
                    "first_ts": pd.to_datetime(gdf["timestamp_ist"]).min(),
                    "last_ts":  pd.to_datetime(gdf["timestamp_ist"]).max(),
                    "created_at_utc": now_ts,
                    "instrument_type": gdf["instrument_type"].iloc[0],
                    "option_type": canon_option_type(gdf["option_type"].iloc[0]),
                    "strike": float(gdf["strike"].iloc[0]) if pd.notna(gdf["strike"].iloc[0]) else None,
                    "strikes_csv": None
                })
        errs_out = pd.DataFrame(err_rows)

        # ---------- PnL & Open positions ----------
        pnl_keys = ["trade_day","account_id","strategy_id","user_id",
                    "strategy_variant_id","instrument_name","instrument_type","option_type","strike",
                    "account_name","strategy_name","portfolio_name","exec_mode"]

        results, openpos = [], []
        for keys, sub in df_agg.groupby(pnl_keys, dropna=False):
            (trade_day, account_id, strategy_id, user_id,
             variant_id, instrument_name, instrument_type, option_type, strike,
             account_name, strategy_name, portfolio_name, exec_mode) = keys
            realized, fills, net_qty, avg_entry, _ = fifo_realized_pnl(sub)
            results.append({
                "trade_day": trade_day, "account_id": account_id, "strategy_id": strategy_id, "user_id": user_id,
                "strategy_variant_id": variant_id, "instrument_name": instrument_name,
                "instrument_type": instrument_type, "option_type": option_type, "strike": strike,
                "account_name": account_name, "strategy_name": strategy_name,
                "portfolio_name": portfolio_name, "exec_mode": exec_mode,
                "realized_pnl": realized, "fills": fills
            })
            if net_qty and abs(net_qty) > 1e-12:
                tmax = pd.to_datetime(sub["timestamp_ist"]).max()
                openpos.append({
                    "trade_day": trade_day, "account_id": account_id, "strategy_id": strategy_id, "user_id": user_id,
                    "strategy_variant_id": variant_id, "instrument_name": instrument_name,
                    "instrument_type": instrument_type, "option_type": option_type, "strike": strike,
                    "account_name": account_name, "strategy_name": strategy_name,
                    "portfolio_name": portfolio_name, "exec_mode": exec_mode,
                    "net_qty": net_qty, "avg_entry_price": avg_entry, "last_trade_ts": tmax
                })

        pnl_out, open_out = pd.DataFrame(results), pd.DataFrame(openpos)
        if not pnl_out.empty:
            _pkeys = ["trade_day","account_id","strategy_id","user_id","strategy_variant_id","instrument_name","instrument_type","option_type","strike"]
            agg_map = {"account_name":"max","strategy_name":"max","portfolio_name":"max",
                       "exec_mode":"max","realized_pnl":"sum","fills":"sum"}
            pnl_out = pnl_out.groupby(_pkeys, as_index=False, dropna=False).agg(agg_map)
            pnl_out["option_type"] = pnl_out["option_type"].apply(canon_option_type)
        if not open_out.empty:
            open_out["option_type"] = open_out["option_type"].apply(canon_option_type)

        # Slippage subset
        slip_out = df.loc[theo_ok, ["unique_id","timestamp_ist","trade_day","user_id","account_name","strategy_name",
                                    "instrument_name","instrument_type","option_type","side","trade_price","theoretical_price",
                                    "slip_abs","slip_pct","exec_mode","exit_reason"]].copy()
        slip_out.rename(columns={"unique_id":"unique_key"}, inplace=True)

        # ---------- Upserts / Inserts ----------
        with dst_engine.begin() as con:
            # Slippage
            if not slip_out.empty:
                slip_out["trade_day"] = pd.to_datetime(slip_out["trade_day"], errors="coerce").dt.date
                for c in ["theoretical_price","trade_price","slip_abs","slip_pct"]:
                    slip_out[c] = pd.to_numeric(slip_out[c], errors="coerce")
                slip_out.to_sql("_slip_stage", con, schema="mart", if_exists="replace", index=False)
                con.execute(text("""
                    INSERT INTO mart.slippage_events AS t
                    (unique_key, timestamp_ist, trade_day, user_id, account_name, strategy_name,
                     instrument_name, instrument_type, option_type, side, trade_price, theoretical_price,
                     slip_abs, slip_pct, exec_mode, exit_reason)
                    SELECT unique_key::text, timestamp_ist::timestamptz, trade_day::date,
                           user_id::text, account_name::text, strategy_name::text,
                           instrument_name::text, instrument_type::text, option_type::text, side::text,
                           trade_price::float8, theoretical_price::float8,
                           slip_abs::float8, slip_pct::float8, exec_mode::text, exit_reason::text
                    FROM mart._slip_stage
                    ON CONFLICT (unique_key) DO UPDATE SET
                      timestamp_ist=EXCLUDED.timestamp_ist, trade_day=EXCLUDED.trade_day,
                      user_id=EXCLUDED.user_id, account_name=EXCLUDED.account_name,
                      strategy_name=EXCLUDED.strategy_name, instrument_name=EXCLUDED.instrument_name,
                      instrument_type=EXCLUDED.instrument_type, option_type=EXCLUDED.option_type, side=EXCLUDED.side,
                      trade_price=EXCLUDED.trade_price, theoretical_price=EXCLUDED.theoretical_price,
                      slip_abs=EXCLUDED.slip_abs, slip_pct=EXCLUDED.slip_pct,
                      exec_mode=EXCLUDED.exec_mode, exit_reason=EXCLUDED.exit_reason;
                    DROP TABLE mart._slip_stage;
                """))

        # ---------- Build day_flow (for QC and mismatch logic) ----------
        if not df.empty:
            day_flow = (df.groupby(
                ["trade_day","account_id","strategy_id","user_id","strategy_variant_id","instrument_type","option_type","strike"],
                dropna=False, as_index=False)["qty"].sum()
                .rename(columns={"qty":"day_net_qty"}))
        else:
            day_flow = pd.DataFrame(columns=["trade_day","account_id","strategy_id","user_id","strategy_variant_id","instrument_type","option_type","strike","day_net_qty"])

        # ---------- Flag and EXCLUDE mismatched EOD buckets BEFORE writing EOD ----------
        if not open_out.empty:
            # Fetch already-flagged keys
            with dst_engine.connect() as con:
                flagged = pd.read_sql(text("""
                    SELECT DISTINCT trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts
                    FROM ops.erroneous_trades
                    WHERE trade_day = :d AND issue_code = 'EOD_MANUAL_MISMATCH'
                """), con, params={"d": day_date})

            # Join candidate EOD vs day_flow on bucket keys
            keys = ["trade_day","account_id","strategy_id","user_id","strategy_variant_id","instrument_type","option_type","strike"]
            eod_df = open_out[keys + ["instrument_name","net_qty"]].rename(columns={"net_qty": "eod_net_qty"}).copy()
            merged = pd.merge(day_flow, eod_df, on=keys, how="outer", validate="one_to_one")
            mism = merged[(merged["day_net_qty"].fillna(0.0) != merged["eod_net_qty"].fillna(0.0))].copy()

            if not mism.empty:
                now_utc = datetime.now(timezone.utc)
                # Aggregate to conflict key of ops.erroneous_trades (no option_type/strike in PK)
                pk_cols = ["trade_day","account_id","strategy_id","user_id","strategy_variant_id"]
                mism["option_type"] = mism["option_type"].apply(canon_option_type)
                mism["instrument_type"] = mism["instrument_type"].apply(lambda x: x if x else "UNKNOWN")
                mism["strike_txt"] = mism["strike"].map(lambda s: (str(int(s)) if pd.notna(s) and abs(s-int(s))<1e-9 else str(s)))

                agg = (mism.groupby(pk_cols, dropna=False)
                       .agg(
                           strikes_csv=("strike_txt", lambda s: ",".join(sorted({x for x in s if x not in (None, 'nan')}))),
                           instrument_type=("instrument_type", "max"),
                           option_type=("option_type", "max"),
                           sample_day_qty=("day_net_qty", "max"),
                           sample_eod_qty=("eod_net_qty", "max"),
                        ).reset_index())

                # remove if already flagged (by PK+issue_code+first_ts@00:00)
                if not flagged.empty:
                    flagged["first_ts"] = pd.to_datetime(flagged["first_ts"])
                first_ts = pd.Timestamp(day_str + " 00:00:00", tz=IST_TZ)
                if not flagged.empty:
                    fkey = flagged[flagged["issue_code"]=="EOD_MANUAL_MISMATCH"].copy()
                    fkey["key"] = fkey[pk_cols].astype(str).agg("|".join, axis=1)
                    already = set(fkey["key"].tolist())
                    agg["key"] = agg[pk_cols].astype(str).agg("|".join, axis=1)
                    agg = agg[~agg["key"].isin(already)].drop(columns=["key"])

                if not agg.empty:
                    # RAW errors rows (one per PK)
                    errs_qc = pd.DataFrame([{
                        "trade_day":           r.trade_day,
                        "account_id":          str(r.account_id),
                        "strategy_id":         str(r.strategy_id),
                        "user_id":             str(r.user_id),
                        "strategy_variant_id": str(r.strategy_variant_id),
                        "issue_code":          "EOD_MANUAL_MISMATCH",
                        "issue_detail":        f"Day flow {r.sample_day_qty} vs EOD {r.sample_eod_qty} differ for strikes [{r.strikes_csv}] — excluded from EOD.",
                        "affected_unique_ids": None,
                        "first_ts":            first_ts,
                        "last_ts":             pd.Timestamp(day_str + " 23:59:59", tz=IST_TZ),
                        "created_at_utc":      now_utc,
                        "instrument_type":     r.instrument_type if pd.notna(r.instrument_type) else "UNKNOWN",
                        "option_type":         canon_option_type(r.option_type),
                        "strike":              None,
                        "strikes_csv":         r.strikes_csv
                    } for r in agg.itertuples(index=False)])

                    # CLEAN errors rows
                    errs_clean_qc = pd.DataFrame([{
                        "trade_day":           r.trade_day,
                        "account_id":          str(r.account_id),
                        "account_name":        None,
                        "strategy_id":         str(r.strategy_id),
                        "strategy_name":       None,
                        "portfolio_name":      None,
                        "exec_mode":           None,
                        "user_id":             str(r.user_id),
                        "strategy_variant_id": str(r.strategy_variant_id),
                        "issue_code":          "EOD_MANUAL_MISMATCH",
                        "issue_detail":        f"Day flow {r.sample_day_qty} vs EOD {r.sample_eod_qty} differ for strikes [{r.strikes_csv}].",
                        "wrong_fills":         0,
                        "strikes_csv":         r.strikes_csv,
                        "affected_ids_csv":    None,
                        "first_ts":            first_ts,
                        "last_ts":             pd.Timestamp(day_str + " 23:59:59", tz=IST_TZ),
                        "created_at_utc":      now_utc,
                        "instrument_type":     r.instrument_type if pd.notna(r.instrument_type) else "UNKNOWN",
                        "option_type":         canon_option_type(r.option_type)
                    } for r in agg.itertuples(index=False)])

                    with dst_engine.begin() as con:
                        if not errs_qc.empty:
                            errs_qc.to_sql("_err_stage_qc", con, schema="ops", if_exists="replace", index=False)
                            con.execute(text("""
                                WITH dedup AS (
                                  SELECT DISTINCT
                                    trade_day::date, account_id::text, strategy_id::text, user_id::text,
                                    strategy_variant_id::text, issue_code::text, first_ts::timestamptz
                                  FROM ops._err_stage_qc
                                )
                                INSERT INTO ops.erroneous_trades
                                (trade_day, account_id, strategy_id, user_id, strategy_variant_id,
                                 issue_code, issue_detail, affected_unique_ids, first_ts, last_ts, created_at_utc,
                                 instrument_type, option_type, strike, strikes_csv)
                                SELECT s.trade_day::date, s.account_id::text, s.strategy_id::text, s.user_id::text, s.strategy_variant_id::text,
                                       s.issue_code::text, s.issue_detail::text, NULL::text[],
                                       s.first_ts::timestamptz, s.last_ts::timestamptz, s.created_at_utc::timestamptz,
                                       s.instrument_type::text, s.option_type::text, s.strike::float8, s.strikes_csv::text
                                FROM ops._err_stage_qc s
                                JOIN dedup d USING (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
                                ON CONFLICT (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
                                DO UPDATE SET
                                  issue_detail     = EXCLUDED.issue_detail,
                                  last_ts          = EXCLUDED.last_ts,
                                  created_at_utc   = EXCLUDED.created_at_utc,
                                  instrument_type  = EXCLUDED.instrument_type,
                                  option_type      = EXCLUDED.option_type,
                                  strike           = EXCLUDED.strike,
                                  strikes_csv      = EXCLUDED.strikes_csv;
                                DROP TABLE ops._err_stage_qc;
                            """))

                        if not errs_clean_qc.empty:
                            errs_clean_qc.to_sql("_err_clean_stage_qc", con, schema="ops", if_exists="replace", index=False)
                            con.execute(text("""
                                WITH dedup AS (
                                  SELECT DISTINCT
                                    trade_day::date, account_id::text, strategy_id::text, user_id::text,
                                    strategy_variant_id::text, issue_code::text, first_ts::timestamptz
                                  FROM ops._err_clean_stage_qc
                                )
                                INSERT INTO ops.erroneous_trades_clean
                                (trade_day, account_id, account_name, strategy_id, strategy_name, portfolio_name, exec_mode,
                                 user_id, strategy_variant_id, issue_code, issue_detail, wrong_fills, strikes_csv, affected_ids_csv,
                                 first_ts, last_ts, created_at_utc, instrument_type, option_type)
                                SELECT s.trade_day::date, s.account_id::text, s.account_name::text, s.strategy_id::text, s.strategy_name::text,
                                       s.portfolio_name::text, s.exec_mode::text, s.user_id::text, s.strategy_variant_id::text,
                                       s.issue_code::text, s.issue_detail::text, s.wrong_fills::int, s.strikes_csv::text, s.affected_ids_csv::text,
                                       s.first_ts::timestamptz, s.last_ts::timestamptz, s.created_at_utc::timestamptz,
                                       s.instrument_type::text, s.option_type::text
                                FROM ops._err_clean_stage_qc s
                                JOIN dedup d USING (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
                                ON CONFLICT (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
                                DO UPDATE SET
                                  issue_detail     = EXCLUDED.issue_detail,
                                  wrong_fills      = EXCLUDED.wrong_fills,
                                  strikes_csv      = EXCLUDED.strikes_csv,
                                  affected_ids_csv = EXCLUDED.affected_ids_csv,
                                  last_ts          = EXCLUDED.last_ts,
                                  created_at_utc   = EXCLUDED.created_at_utc,
                                  instrument_type  = EXCLUDED.instrument_type,
                                  option_type      = EXCLUDED.option_type;
                                DROP TABLE ops._err_clean_stage_qc;
                            """))

                    # Exclude mismatched keys from open_out
                    mismatched_keys = set(
                        (str(r.account_id), str(r.strategy_id), str(r.user_id), str(r.strategy_variant_id),
                         str(r.instrument_type if r.instrument_type else "UNKNOWN"),
                         str(canon_option_type(r.option_type)), float(r.strike))
                        for r in mism.itertuples(index=False) if pd.notna(r.strike)
                    )
                    if mismatched_keys:
                        key_cols = ["account_id","strategy_id","user_id","strategy_variant_id","instrument_type","option_type","strike"]
                        open_out["instrument_type"] = open_out["instrument_type"].fillna("UNKNOWN")
                        open_out["option_type"] = open_out["option_type"].apply(canon_option_type)
                        mask_keep = ~open_out[key_cols].apply(
                            lambda s: (str(s["account_id"]), str(s["strategy_id"]), str(s["user_id"]),
                                       str(s["strategy_variant_id"]), str(s["instrument_type"]),
                                       str(s["option_type"]), float(s["strike"])) in mismatched_keys,
                            axis=1
                        )
                        open_out = open_out.loc[mask_keep].copy()

        # ---------- Upsert PnL & EOD AFTER exclusions ----------
        with dst_engine.begin() as con:
            # PnL (delete-by-day then insert)
            if not pnl_out.empty:
                pnl_out.to_sql("_pnl_stage", con, schema="mart", if_exists="replace", index=False)
                con.execute(text("DELETE FROM mart.trade_pnl WHERE trade_day = :d"), {"d": day_date})
                con.execute(text("""
                    WITH dedup AS (
                      SELECT trade_day::date AS trade_day,
                             account_id::text AS account_id,
                             strategy_id::text AS strategy_id,
                             user_id::text AS user_id,
                             strategy_variant_id::text AS strategy_variant_id,
                             instrument_name::text AS instrument_name,
                             instrument_type::text AS instrument_type,
                             option_type::text AS option_type,
                             strike::float8 AS strike,
                             MAX(account_name)::text AS account_name,
                             MAX(strategy_name)::text AS strategy_name,
                             MAX(portfolio_name)::text AS portfolio_name,
                             MAX(exec_mode)::text AS exec_mode,
                             SUM(realized_pnl)::float8 AS realized_pnl,
                             SUM(fills)::int AS fills
                      FROM mart._pnl_stage
                      GROUP BY 1,2,3,4,5,6,7,8,9
                    )
                    INSERT INTO mart.trade_pnl
                    (trade_day, account_id, strategy_id, user_id, strategy_variant_id,
                     instrument_name, instrument_type, option_type, strike,
                     account_name, strategy_name, portfolio_name, exec_mode, realized_pnl, fills)
                    SELECT trade_day, account_id, strategy_id, user_id, strategy_variant_id,
                           instrument_name, instrument_type, option_type, strike,
                           account_name, strategy_name, portfolio_name, exec_mode, realized_pnl, fills
                    FROM dedup;
                    DROP TABLE mart._pnl_stage;
                """))

            # EOD (delete-by-day then insert)
            if not open_out.empty:
                open_out.to_sql("_open_stage", con, schema="mart", if_exists="replace", index=False)
                con.execute(text("DELETE FROM mart.open_positions_eod WHERE trade_day = :d"), {"d": day_date})
                con.execute(text("""
                    WITH dedup AS (
                      SELECT DISTINCT ON (trade_day, account_id, strategy_id, user_id, strategy_variant_id, instrument_type, option_type, strike)
                             trade_day::date AS trade_day,
                             account_id::text AS account_id,
                             strategy_id::text AS strategy_id,
                             user_id::text AS user_id,
                             strategy_variant_id::text AS strategy_variant_id,
                             instrument_name::text AS instrument_name,
                             instrument_type::text AS instrument_type,
                             option_type::text AS option_type,
                             strike::float8 AS strike,
                             account_name::text AS account_name,
                             strategy_name::text AS strategy_name,
                             portfolio_name::text AS portfolio_name,
                             exec_mode::text AS exec_mode,
                             net_qty::float8 AS net_qty,
                             avg_entry_price::float8 AS avg_entry_price,
                             last_trade_ts::timestamptz AS last_trade_ts
                      FROM mart._open_stage
                      ORDER BY trade_day, account_id, strategy_id, user_id, strategy_variant_id, instrument_type, option_type, strike, last_trade_ts DESC
                    )
                    INSERT INTO mart.open_positions_eod
                      (trade_day, account_id, strategy_id, user_id, strategy_variant_id,
                       instrument_name, instrument_type, option_type, strike,
                       account_name, strategy_name, portfolio_name, exec_mode,
                       net_qty, avg_entry_price, last_trade_ts)
                    SELECT trade_day, account_id, strategy_id, user_id, strategy_variant_id,
                           instrument_name, instrument_type, option_type, strike,
                           account_name, strategy_name, portfolio_name, exec_mode,
                           net_qty, avg_entry_price, last_trade_ts
                    FROM dedup
                    ON CONFLICT (trade_day, account_id, strategy_id, user_id, strategy_variant_id, instrument_type, option_type, strike)
                    DO UPDATE SET
                      instrument_name=EXCLUDED.instrument_name,
                      account_name=EXCLUDED.account_name,
                      strategy_name=EXCLUDED.strategy_name,
                      portfolio_name=EXCLUDED.portfolio_name,
                      exec_mode=EXCLUDED.exec_mode,
                      net_qty=EXCLUDED.net_qty,
                      avg_entry_price=EXCLUDED.avg_entry_price,
                      last_trade_ts=GREATEST(mart.open_positions_eod.last_trade_ts, EXCLUDED.last_trade_ts),
                      instrument_type=EXCLUDED.instrument_type,
                      option_type=EXCLUDED.option_type,
                      strike=EXCLUDED.strike;
                    DROP TABLE mart._open_stage;
                """))

        # ---------- Per-day QC ----------
        expected_cin = 0 if cin_df is None else len(cin_df)
        with dst_engine.connect() as con:
            prev_day_date = day_date - timedelta(days=1)
            prev_cnt_db = pd.read_sql(text("""
                SELECT COUNT(*) AS c
                FROM (
                  SELECT DISTINCT
                          account_id::text,
                          strategy_id::text,
                          user_id::text,
                          strategy_variant_id::text,
                          COALESCE(NULLIF(option_type,''),'NA') AS option_type,
                          strike::float8
                  FROM mart.open_positions_eod
                  WHERE trade_day = :prev_day
                    AND COALESCE(net_qty,0) <> 0
                ) x
            """), con, params={"prev_day": prev_day_date})["c"].iloc[0]

        if expected_cin != prev_cnt_db:
            print(f"QC FAILED (day {day_str}): prev_eod_rows={prev_cnt_db} cin_rows={expected_cin}")
            raise RuntimeError("Carry-in count mismatch detected (per-day)")

        #  day_flow vs EOD for this day — must match after exclusions
        with dst_engine.connect() as con:
            qc = pd.read_sql(text("""
                WITH flagged AS (
                  SELECT DISTINCT
                    trade_day, account_id, strategy_id, user_id, strategy_variant_id
                  FROM ops.erroneous_trades
                  WHERE trade_day = :d AND issue_code = 'EOD_MANUAL_MISMATCH'
                ),
                day_flow AS (
                  SELECT
                    t.trade_day,
                    t.account_id, t.strategy_id, t.user_id, t.strategy_variant_id,
                    COALESCE(NULLIF(t.instrument_type,''),'UNKNOWN') AS instrument_type,
                    COALESCE(NULLIF(t.option_type,''),'NA')          AS option_type,
                    t.strike::float8                                 AS strike,
                    SUM(t.qty)::float8                               AS day_net_qty
                  FROM mart.trades_all t
                  LEFT JOIN flagged f
                    ON f.trade_day=t.trade_day
                   AND f.account_id=t.account_id
                   AND f.strategy_id=t.strategy_id
                   AND f.user_id=t.user_id
                   AND f.strategy_variant_id=t.strategy_variant_id
                  WHERE t.trade_day = :d
                    AND f.trade_day IS NULL
                  GROUP BY 1,2,3,4,5,6,7,8
                ),
                eod AS (
                  SELECT
                    e.trade_day,
                    e.account_id, e.strategy_id, e.user_id, e.strategy_variant_id,
                    COALESCE(NULLIF(e.instrument_type,''),'UNKNOWN') AS instrument_type,
                    COALESCE(NULLIF(e.option_type,''),'NA')          AS option_type,
                    e.strike::float8                                 AS strike,
                    e.net_qty::float8                                AS eod_net_qty
                  FROM mart.open_positions_eod e
                  LEFT JOIN flagged f
                    ON f.trade_day=e.trade_day
                   AND f.account_id=e.account_id
                   AND f.strategy_id=e.strategy_id
                   AND f.user_id=e.user_id
                   AND f.strategy_variant_id=e.strategy_variant_id
                  WHERE e.trade_day = :d
                    AND f.trade_day IS NULL
                )
                SELECT
                  COALESCE(d.trade_day, e.trade_day)      AS trade_day,
                  COALESCE(d.account_id, e.account_id)    AS account_id,
                  COALESCE(d.strategy_id, e.strategy_id)  AS strategy_id,
                  COALESCE(d.user_id, e.user_id)          AS user_id,
                  COALESCE(d.strategy_variant_id, e.strategy_variant_id) AS strategy_variant_id,
                  COALESCE(d.instrument_type, e.instrument_type) AS instrument_type,
                  COALESCE(d.option_type, e.option_type)   AS option_type,
                  COALESCE(d.strike, e.strike)             AS strike,
                  d.day_net_qty,
                  e.eod_net_qty
                FROM day_flow d
                FULL OUTER JOIN eod e
                  ON e.trade_day=d.trade_day
                 AND e.account_id=d.account_id
                 AND e.strategy_id=d.strategy_id
                 AND e.user_id=d.user_id
                 AND e.strategy_variant_id=d.strategy_variant_id
                 AND e.instrument_type=d.instrument_type
                 AND e.option_type=d.option_type
                 AND e.strike=d.strike
                WHERE COALESCE(d.day_net_qty,0) <> COALESCE(e.eod_net_qty,0)
                ORDER BY 1,2,3,4,5,6,7,8
            """), con, params={"d": day_date})

        if not qc.empty:
            print("QC FAILED (day_flow vs EOD) for", day_str, ":\n", qc)
            raise RuntimeError("Day-flow vs EOD mismatch detected (per-day)")

        print(f"✅ ETL day {day_str} complete")

    print(f"✅ ETL finished for {from_date}..{to_date}")

# ================== Glue ==================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="from_date", help="YYYY-MM-DD (inclusive)")
    ap.add_argument("--to",   dest="to_date",   help="YYYY-MM-DD (inclusive)")
    args = ap.parse_args()

    if not args.from_date and not args.to_date:
        y = date.today() - timedelta(days=1)
        args.from_date = args.to_date = y.isoformat()

    src = create_engine(SOURCE_DB_URL, pool_pre_ping=True)
    dst = create_engine(REPORT_DB_URL, pool_pre_ping=True)

    ensure_dim_schema(dst)
    ensure_mart_ops_schema(dst)
    sync_dims(src, dst)
    run_trade_etl(src, dst, args.from_date, args.to_date)

    print("Tip: ops.v_erroneous_trades_clean, ops.v_erroneous_counts_by_bucket, "
          "mart.v_slippage_daily, mart.v_strategy_pnl, mart.v_open_positions_eod.")

if __name__ == "__main__":
    main()
