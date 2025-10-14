#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end: DIM sync (from algo_department.core) + Slippage/PnL ETL + Error flags
- Creates schemas/tables: dim, mart, ops (idempotent, safe with existing tables)
- Upserts DIMs from source DB (no FDW required)
- Runs trade ETL with IST-safe date parsing
- Adds helpful views + indexes for Metabase
- Handles cross-day realization via carry-in positions from prior EOD
- FIX: carry-in rows now get a synthetic unique_id so inserts into mart.trades_all never violate NOT NULL/PK
- FIX: PnL dedup both in pandas and SQL to avoid PK collisions in mart.trade_pnl

Run examples:
  python reporting_system.py --from 2025-10-03 --to 2025-10-03
  python reporting_system.py   # defaults to yesterday
"""

import argparse
from datetime import date, timedelta, datetime, timezone, time as dt_time
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ----------------- CONFIG -----------------
SOURCE_DB_URL = "postgresql+psycopg2://disha:zxcvbnm@192.168.18.18:5432/algo_department"
REPORT_DB_URL = "postgresql+psycopg2://disha:zxcvbnm@192.168.18.18:5432/report"

# If you manage schema via DBeaver or migrations, set to True to skip all DDL.
MANAGE_SCHEMA_EXTERNALLY = False
# -----------------------------------------

IST_TZ = "Asia/Kolkata"

# ===================== Day-first parsing helpers (IST-safe) =====================
def _parse_date_india_first(x) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.Timestamp(x)
    if isinstance(x, date):
        return pd.Timestamp(datetime(x.year, x.month, x.day))
    s = str(x).strip()
    if not s:
        return pd.NaT
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y"):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            pass
    return pd.to_datetime(s, errors="coerce", dayfirst=True)

def _parse_time_safe(x) -> dt_time | None:
    if pd.isna(x):
        return dt_time(0, 0, 0)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.time()
    s = str(x).strip()
    if not s:
        return dt_time(0, 0, 0)
    s = s.replace("  ", " ")
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
        try:
            return datetime.strptime(s, fmt).time()
        except Exception:
            pass
    try:
        t = pd.to_datetime(s, errors="coerce").time()
        return t if t is not None else dt_time(0, 0, 0)
    except Exception:
        return dt_time(0, 0, 0)

def to_ts_ist(dates, times, tz=IST_TZ):
    ds = pd.Series(dates, copy=False)
    ts = pd.Series(times, copy=False)
    d_parsed = ds.apply(_parse_date_india_first)
    t_parsed = ts.apply(_parse_time_safe)

    def _combine(d, t):
        if pd.isna(d):
            return pd.NaT
        dt_naive = datetime(d.year, d.month, d.day, t.hour, t.minute, t.second)
        return pd.Timestamp(dt_naive).tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT")

    out = [_combine(d, t) for d, t in zip(d_parsed, t_parsed)]
    return pd.Series(out, dtype=f"datetime64[ns, {tz}]")
# ==============================================================================

def parse_option_type(s):
    if not isinstance(s, str): return None
    u = s.upper()
    if "CE" in u: return "CE"
    if "PE" in u: return "PE"
    return None

def fifo_realized_pnl(group_df):
    """
    FIFO (First-In, First-Out) within bucket (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike)
    Returns: realized_pnl, fills, net_qty, avg_entry, exit_without_open
    """
    g = group_df.sort_values("timestamp_ist").copy()
    fills = len(g)
    realized = 0.0
    open_lots = []  # {"sign": +1/-1, "qty": >0, "price": float, "time": ts}
    exit_without_open = False

    for _, r in g.iterrows():
        qty = float(r["qty"])
        price = float(r["trade_price"])
        if qty == 0 or np.isnan(qty) or np.isnan(price):
            continue
        sign = 1.0 if qty > 0 else -1.0
        remaining = abs(qty)

        if not open_lots or all(lot["sign"] == sign for lot in open_lots):
            open_lots.append({"sign": sign, "qty": remaining, "price": price, "time": r["timestamp_ist"]})
            continue

        i = 0
        touched_opposite = False
        while remaining > 1e-12 and i < len(open_lots):
            lot = open_lots[i]
            if lot["sign"] == sign:
                i += 1
                continue
            touched_opposite = True
            matched = min(remaining, lot["qty"])
            entry_price = lot["price"]
            exit_price  = price
            realized += matched * (exit_price - entry_price) * lot["sign"]
            lot["qty"] -= matched
            remaining  -= matched
            if lot["qty"] <= 1e-12:
                open_lots.pop(i)
            else:
                i += 1

        if not touched_opposite and abs(qty) > 0 and sign != (open_lots[0]["sign"] if open_lots else sign):
            exit_without_open = True
        if remaining > 1e-12:
            open_lots.append({"sign": sign, "qty": remaining, "price": price, "time": r["timestamp_ist"]})

    if not open_lots:
        return realized, fills, 0.0, None, exit_without_open

    net_qty = sum(lot["sign"] * lot["qty"] for lot in open_lots)
    if abs(net_qty) <= 1e-12:
        return realized, fills, 0.0, None, exit_without_open

    dominant_sign = 1.0 if net_qty > 0 else -1.0
    dom_qty = sum(lot["qty"] for lot in open_lots if lot["sign"] == dominant_sign)
    avg_entry = (
        sum(lot["price"] * lot["qty"] for lot in open_lots if lot["sign"] == dominant_sign) / dom_qty
        if dom_qty > 0 else None
    )
    return realized, fills, net_qty, avg_entry, exit_without_open

# ================== Schema DDL ==================
DDL_DIM = """
CREATE SCHEMA IF NOT EXISTS dim;

-- portfolios
CREATE TABLE IF NOT EXISTS dim.portfolios (
  id         bigint PRIMARY KEY,
  portfolio  text NOT NULL,
  mode       text NOT NULL,
  updated_at timestamptz DEFAULT now()
);
ALTER TABLE dim.portfolios
  ADD COLUMN IF NOT EXISTS id         bigint,
  ADD COLUMN IF NOT EXISTS portfolio  text,
  ADD COLUMN IF NOT EXISTS mode       text,
  ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();
CREATE UNIQUE INDEX IF NOT EXISTS ux_dim_portfolios_name ON dim.portfolios(portfolio);

-- strategies
CREATE TABLE IF NOT EXISTS dim.strategies (
  id            bigint PRIMARY KEY,
  strategy_name text   NOT NULL,
  portfolio_id  bigint NOT NULL REFERENCES dim.portfolios(id),
  updated_at    timestamptz DEFAULT now()
);
ALTER TABLE dim.strategies
  ADD COLUMN IF NOT EXISTS id            bigint,
  ADD COLUMN IF NOT EXISTS strategy_name text,
  ADD COLUMN IF NOT EXISTS portfolio_id  bigint,
  ADD COLUMN IF NOT EXISTS updated_at    timestamptz DEFAULT now();
CREATE INDEX IF NOT EXISTS ix_dim_strategies_portfolio ON dim.strategies(portfolio_id);

-- strategy_variants
CREATE TABLE IF NOT EXISTS dim.strategy_variants (
  id           bigint PRIMARY KEY,
  strategy_id  bigint NOT NULL REFERENCES dim.strategies(id),
  variant_name text   NOT NULL,
  updated_at   timestamptz DEFAULT now()
);
ALTER TABLE dim.strategy_variants
  ADD COLUMN IF NOT EXISTS id           bigint,
  ADD COLUMN IF NOT EXISTS strategy_id  bigint,
  ADD COLUMN IF NOT EXISTS variant_name text,
  ADD COLUMN IF NOT EXISTS updated_at   timestamptz DEFAULT now();
CREATE INDEX IF NOT EXISTS ix_dim_variants_strategy ON dim.strategy_variants(strategy_id);

-- accounts
CREATE TABLE IF NOT EXISTS dim.accounts (
  account_id bigint PRIMARY KEY,
  name       text NOT NULL,
  updated_at timestamptz DEFAULT now()
);
ALTER TABLE dim.accounts
  ADD COLUMN IF NOT EXISTS account_id bigint,
  ADD COLUMN IF NOT EXISTS name       text,
  ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();
CREATE UNIQUE INDEX IF NOT EXISTS ux_dim_accounts_name ON dim.accounts(name);

-- users
CREATE TABLE IF NOT EXISTS dim.users (
  user_id    bigint PRIMARY KEY,
  name       text NOT NULL,
  updated_at timestamptz DEFAULT now()
);
ALTER TABLE dim.users
  ADD COLUMN IF NOT EXISTS user_id    bigint,
  ADD COLUMN IF NOT EXISTS name       text,
  ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

-- account_assignments
CREATE TABLE IF NOT EXISTS dim.account_assignments (
  account_id bigint NOT NULL REFERENCES dim.accounts(account_id),
  user_id    bigint NOT NULL REFERENCES dim.users(user_id),
  valid_from date   NOT NULL DEFAULT DATE '1970-01-01',
  valid_to   date,
  PRIMARY KEY (account_id, user_id, valid_from)
);
ALTER TABLE dim.account_assignments
  ADD COLUMN IF NOT EXISTS account_id bigint,
  ADD COLUMN IF NOT EXISTS user_id    bigint,
  ADD COLUMN IF NOT EXISTS valid_from date   DEFAULT DATE '1970-01-01',
  ADD COLUMN IF NOT EXISTS valid_to   date;
CREATE INDEX IF NOT EXISTS ix_dim_acc_assign_user ON dim.account_assignments(user_id);
CREATE INDEX IF NOT EXISTS ix_dim_acc_assign_acc  ON dim.account_assignments(account_id);
"""

# ---------- TABLES ONLY (final shapes; no trade_number anywhere) ----------
DDL_MART_OPS_TABLES = """
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS ops;

CREATE TABLE IF NOT EXISTS mart.trades_all (
  unique_id           text PRIMARY KEY,
  timestamp_ist       timestamptz,
  trade_day           date,
  account_id          text,
  strategy_id         text,
  user_id             text,
  strategy_variant_id text,
  instrument_name     text,
  type                text,
  option_type         text,
  strike              double precision,
  qty                 double precision,
  trade_price         double precision,
  side                text,
  entry_label         text,
  theoretical_price   double precision,
  exec_mode           text,
  account_name        text,
  strategy_name       text,
  portfolio_name      text
);

CREATE TABLE IF NOT EXISTS mart.slippage_events (
  unique_key         text PRIMARY KEY,
  timestamp_ist      timestamptz,
  trade_day          date,
  user_id            text,
  account_name       text,
  strategy_name      text,
  instrument_name    text,
  option_type        text,
  side               text,
  trade_price        double precision,
  theoretical_price  double precision,
  slip_abs           double precision,
  slip_pct           double precision,
  exec_mode          text
);

CREATE TABLE IF NOT EXISTS mart.trade_pnl (
  trade_day            date,
  account_id           text,
  strategy_id          text,
  user_id              text,
  strategy_variant_id  text,
  strike               double precision,
  account_name         text,
  strategy_name        text,
  portfolio_name       text,
  exec_mode            text,
  realized_pnl         double precision,
  fills                int,
  PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike)
);

CREATE TABLE IF NOT EXISTS mart.open_positions_eod (
  trade_day            date,
  account_id           text,
  strategy_id          text,
  user_id              text,
  strategy_variant_id  text,
  strike               double precision,
  account_name         text,
  strategy_name        text,
  portfolio_name       text,
  exec_mode            text,
  net_qty              double precision,
  avg_entry_price      double precision,
  last_trade_ts        timestamptz,
  PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike)
);

CREATE TABLE IF NOT EXISTS ops.erroneous_trades (
  trade_day            date,
  account_id           text,
  strategy_id          text,
  user_id              text,
  strategy_variant_id  text,
  issue_code           text,
  issue_detail         text,
  affected_unique_ids  text[] ,
  first_ts             timestamptz,
  last_ts              timestamptz,
  created_at_utc       timestamptz,
  PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
);

CREATE TABLE IF NOT EXISTS ops.erroneous_trades_clean (
  trade_day            date,
  account_id           text,
  account_name         text,
  strategy_id          text,
  strategy_name        text,
  portfolio_name       text,
  exec_mode            text,
  user_id              text,
  strategy_variant_id  text,
  issue_code           text,
  issue_detail         text,
  wrong_fills          int,
  strikes_csv          text,
  affected_ids_csv     text,
  first_ts             timestamptz,
  last_ts              timestamptz,
  created_at_utc       timestamptz,
  PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
);
"""

# ---------- VIEWS (drop first, then create) ----------
DDL_MART_OPS_VIEWS_DROP = """
DROP VIEW IF EXISTS ops.v_erroneous_trades_clean CASCADE;
DROP VIEW IF EXISTS ops.v_erroneous_counts_by_bucket CASCADE;
DROP VIEW IF EXISTS mart.v_slippage_daily CASCADE;
DROP VIEW IF EXISTS mart.v_strategy_pnl CASCADE;
DROP VIEW IF EXISTS mart.v_open_positions_eod CASCADE;
"""

DDL_MART_OPS_VIEWS_CREATE = """
CREATE VIEW ops.v_erroneous_trades_clean AS
SELECT
  trade_day, portfolio_name, strategy_name, account_name, exec_mode,
  strategy_variant_id, issue_code,
  wrong_fills, strikes_csv, affected_ids_csv,
  first_ts, last_ts, issue_detail
FROM ops.erroneous_trades_clean
ORDER BY trade_day DESC, portfolio_name, strategy_name, issue_code;

CREATE VIEW ops.v_erroneous_counts_by_bucket AS
SELECT
  trade_day, portfolio_name, strategy_name, exec_mode,
  COUNT(*)         AS bad_trades,
  SUM(wrong_fills) AS bad_fills
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
       strategy_variant_id, strike,
       SUM(net_qty) AS net_qty,
       AVG(avg_entry_price) AS avg_entry_price
FROM mart.open_positions_eod
GROUP BY 1,2,3,4,5,6;
"""

DDL_INDEXES_PERF = """
CREATE INDEX IF NOT EXISTS ix_slip_day            ON mart.slippage_events(trade_day);
CREATE INDEX IF NOT EXISTS ix_slip_strategy_day   ON mart.slippage_events(strategy_name, trade_day);
CREATE INDEX IF NOT EXISTS ix_slip_account_day    ON mart.slippage_events(account_name, trade_day);

CREATE INDEX IF NOT EXISTS ix_pnl_all_keys ON mart.trade_pnl
  (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike);
CREATE INDEX IF NOT EXISTS ix_pnl_day_strategy ON mart.trade_pnl(trade_day, strategy_id);
CREATE INDEX IF NOT EXISTS ix_pnl_day_account  ON mart.trade_pnl(trade_day, account_id);
CREATE INDEX IF NOT EXISTS ix_pnl_day_user     ON mart.trade_pnl(trade_day, user_id);

CREATE INDEX IF NOT EXISTS ix_open_day         ON mart.open_positions_eod(trade_day);
CREATE INDEX IF NOT EXISTS ix_err_day          ON ops.erroneous_trades(trade_day);
"""

# ================== Migration helper (drop trade_number; dedup PnL) ==================
def migrate_drop_trade_number(dst_engine):
    with dst_engine.begin() as con:
        con.execute(text(DDL_MART_OPS_VIEWS_DROP))

        def col_exists(schema, table, column):
            q = text("""
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema=:s AND table_name=:t AND column_name=:c
                LIMIT 1
            """)
            return con.execute(q, {"s": schema, "t": table, "c": column}).first() is not None

        def drop_all_pks(schema, table):
            sql = text("""
                SELECT conname
                FROM pg_constraint
                WHERE conrelid = (:schema || '.' || :table)::regclass
                  AND contype = 'p'
            """)
            for (conname,) in con.execute(sql, {"schema": schema, "table": table}).fetchall():
                con.execute(text(f'ALTER TABLE {schema}.{table} DROP CONSTRAINT IF EXISTS "{conname}";'))

        # 1) mart.trade_pnl: drop legacy column, dedup, add new PK
        if col_exists("mart", "trade_pnl", "trade_number"):
            drop_all_pks("mart", "trade_pnl")
            con.execute(text("ALTER TABLE mart.trade_pnl DROP COLUMN IF EXISTS trade_number"))

        con.execute(text("""
            CREATE TEMP TABLE IF NOT EXISTS _tmp_pnl_dedup AS
            SELECT
              trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike,
              MAX(account_name)   AS account_name,
              MAX(strategy_name)  AS strategy_name,
              MAX(portfolio_name) AS portfolio_name,
              MAX(exec_mode)      AS exec_mode,
              SUM(realized_pnl)   AS realized_pnl,
              SUM(fills)          AS fills
            FROM mart.trade_pnl
            GROUP BY 1,2,3,4,5,6;
        """))
        con.execute(text("TRUNCATE TABLE mart.trade_pnl;"))
        con.execute(text("""
            INSERT INTO mart.trade_pnl
            (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike,
             account_name, strategy_name, portfolio_name, exec_mode, realized_pnl, fills)
            SELECT * FROM _tmp_pnl_dedup;
        """))
        con.execute(text("DROP TABLE IF EXISTS _tmp_pnl_dedup;"))
        drop_all_pks("mart", "trade_pnl")
        con.execute(text("""
            ALTER TABLE mart.trade_pnl
            ADD PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike)
        """))

        # 2) ops tables: remove legacy trade_number if present
        if col_exists("ops", "erroneous_trades", "trade_number"):
            drop_all_pks("ops", "erroneous_trades")
            con.execute(text("ALTER TABLE ops.erroneous_trades DROP COLUMN IF EXISTS trade_number"))
            con.execute(text("""
                ALTER TABLE ops.erroneous_trades
                ADD PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
            """))

        if col_exists("ops", "erroneous_trades_clean", "trade_number"):
            drop_all_pks("ops", "erroneous_trades_clean")
            con.execute(text("ALTER TABLE ops.erroneous_trades_clean DROP COLUMN IF EXISTS trade_number"))
            con.execute(text("""
                ALTER TABLE ops.erroneous_trades_clean
                ADD PRIMARY KEY (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
            """))

# ================== DIM schema creation helpers ==================
def ensure_dim_schema(dst_engine):
    if MANAGE_SCHEMA_EXTERNALLY:
        return
    with dst_engine.begin() as con:
        con.execute(text(DDL_DIM))

def ensure_mart_ops_schema(dst_engine):
    if MANAGE_SCHEMA_EXTERNALLY:
        return
    migrate_drop_trade_number(dst_engine)
    with dst_engine.begin() as con:
        con.execute(text(DDL_MART_OPS_TABLES))
        con.execute(text(DDL_INDEXES_PERF))
        con.execute(text(DDL_MART_OPS_VIEWS_CREATE))

# ------------------ Helpers to fetch source DFs robustly ------------------
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
            continue
    raise last_err

# ================== DIM syncs (from SOURCE to REPORT) ==================
def sync_dims(src_engine, dst_engine):
    def stage_and_upsert(df: pd.DataFrame, stage_fq: str, upsert_sql: str):
        if df is None or df.empty:
            return
        with dst_engine.begin() as con:
            schema, table = stage_fq.split(".")
            df.to_sql(table, con, schema=schema, if_exists="replace", index=False)
            con.execute(text(upsert_sql))
            con.execute(text(f"DROP TABLE {stage_fq};"))

    with src_engine.connect() as con:
        df_port = pd.read_sql(text("""
            SELECT
                id::bigint                                  AS id,
                portfolio::text                             AS portfolio,
                CASE WHEN mode IS NULL OR mode::text = '' THEN 'unknown'
                     ELSE mode::text END                    AS mode
            FROM core.portfolios
        """), con)
    df_port["mode"] = df_port["mode"].fillna("").astype(str).str.strip().replace({"": "unknown"})
    stage_and_upsert(
        df_port, "dim._stage_portfolios",
        """
        INSERT INTO dim.portfolios (id, portfolio, mode, updated_at)
        SELECT id, portfolio, mode, now()
        FROM dim._stage_portfolios
        ON CONFLICT (id) DO UPDATE
        SET portfolio  = EXCLUDED.portfolio,
            mode       = EXCLUDED.mode,
            updated_at = now();
        """
    )

    with src_engine.connect() as con:
        df_strat = pd.read_sql(text("""
            SELECT id::bigint AS id, strategy_name::text AS strategy_name, portfolio_id::bigint AS portfolio_id
            FROM core.strategies
        """), con)
    stage_and_upsert(
        df_strat, "dim._stage_strategies",
        """
        INSERT INTO dim.strategies (id, strategy_name, portfolio_id, updated_at)
        SELECT id, strategy_name, portfolio_id, now()
        FROM dim._stage_strategies
        ON CONFLICT (id) DO UPDATE
        SET strategy_name = EXCLUDED.strategy_name,
            portfolio_id  = EXCLUDED.portfolio_id,
            updated_at    = now();
        """
    )

    df_var = fetch_strategy_variants_df(src_engine)
    stage_and_upsert(
        df_var, "dim._stage_strategy_variants",
        """
        INSERT INTO dim.strategy_variants (id, strategy_id, variant_name, updated_at)
        SELECT id, strategy_id, variant_name, now()
        FROM dim._stage_strategy_variants
        ON CONFLICT (id) DO UPDATE
        SET strategy_id  = EXCLUDED.strategy_id,
            variant_name = EXCLUDED.variant_name,
            updated_at   = now();
        """
    )

    with src_engine.connect() as con:
        df_acct = pd.read_sql(text("""
            SELECT id::bigint AS account_id, name::text AS name
            FROM core.accounts
        """), con)
    stage_and_upsert(
        df_acct, "dim._stage_accounts",
        """
        INSERT INTO dim.accounts (account_id, name, updated_at)
        SELECT account_id, name, now()
        FROM dim._stage_accounts
        ON CONFLICT (account_id) DO UPDATE
        SET name       = EXCLUDED.name,
            updated_at = now();
        """
    )

    with src_engine.connect() as con:
        df_users = pd.read_sql(text("""
            SELECT id_no::bigint AS user_id, name::text AS name
            FROM core.users
        """), con)
    stage_and_upsert(
        df_users, "dim._stage_users",
        """
        INSERT INTO dim.users (user_id, name, updated_at)
        SELECT user_id, name, now()
        FROM dim._stage_users
        ON CONFLICT (user_id) DO UPDATE
        SET name       = EXCLUDED.name,
            updated_at = now();
        """
    )

    with src_engine.connect() as con:
        df_assign = pd.read_sql(text("""
            SELECT DISTINCT account_id::bigint AS account_id,
                            user_id::bigint    AS user_id,
                            NULL::date         AS valid_from,
                            NULL::date         AS valid_to
            FROM core.account_assignments
        """), con)
    stage_and_upsert(
        df_assign, "dim._stage_account_assignments",
        """
        INSERT INTO dim.account_assignments (account_id, user_id, valid_from, valid_to)
        SELECT
          account_id,
          user_id,
          COALESCE(NULLIF(valid_from::text, '')::date, DATE '1970-01-01') AS valid_from,
          NULLIF(valid_to::text, '')::date                                 AS valid_to
        FROM dim._stage_account_assignments
        ON CONFLICT (account_id, user_id, valid_from) DO NOTHING;
        """
    )

# ================== Carry-in from previous EOD ==================
def fetch_carry_in_positions(dst_engine, from_date):
    """
    Return prior-day open positions to seed FIFO for cross-day realization.
    Produces synthetic fills dated at the start of from_date (IST).
    """
    prev_day = (pd.to_datetime(from_date).date() - timedelta(days=1)).isoformat()
    q = text("""
        SELECT trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike,
               account_name, strategy_name, portfolio_name, exec_mode,
               net_qty, avg_entry_price, last_trade_ts
        FROM mart.open_positions_eod
        WHERE trade_day = :d
    """)
    with dst_engine.connect() as con:
        df_prev = pd.read_sql(q, con, params={"d": prev_day})
    if df_prev.empty:
        return df_prev

    start_ts = pd.Timestamp(f"{from_date} 00:00:00").tz_localize(IST_TZ)
    carry = pd.DataFrame({
        "unique_id": None,  # will be set below
        "instrument_name": None,
        "type": None,
        "option_type": None,
        "entry_exit_error": "CARRY_IN",
        "theoretical_price": np.nan,
        "theoretical_time": None,
        "timestamp_ist": start_ts,
        "trade_day": pd.to_datetime(from_date).date(),
        "account_id": df_prev["account_id"].astype(str),
        "strategy_id": df_prev["strategy_id"].astype(str),
        "user_id": df_prev["user_id"].astype(str),
        "strategy_variant_id": df_prev["strategy_variant_id"].astype(str),
        "strike": df_prev["strike"].astype(float),
        "qty": df_prev["net_qty"].astype(float),
        "trade_price": df_prev["avg_entry_price"].astype(float),
        "side": np.where(df_prev["net_qty"] > 0, "BUY",
                         np.where(df_prev["net_qty"] < 0, "SELL", None)),
        "entry_label": "CARRY_IN",
        "exec_mode": df_prev["exec_mode"],
        "account_name": df_prev["account_name"],
        "strategy_name": df_prev["strategy_name"],
        "portfolio_name": df_prev["portfolio_name"],
    })

    def _strike_text(val: float) -> str:
        if pd.isna(val):
            return "NA"
        i = int(val)
        return str(i) if abs(val - i) < 1e-9 else str(val)

    carry["unique_id"] = carry.apply(
        lambda r: f"CIN:{r['trade_day']}:{str(r['account_id'])}:{str(r['strategy_id'])}:{str(r['user_id'])}:{str(r['strategy_variant_id'])}:{_strike_text(float(r['strike']))}",
        axis=1,
    )
    carry["instrument_name"] = carry["instrument_name"].fillna("CARRY_IN")
    carry["type"] = carry["type"].fillna("CARRY_IN")
    return carry

# ================== Trade ETL ==================
def run_trade_etl(src_engine, dst_engine, from_date: str, to_date: str):
    # ---------- Pull slice ----------
    q = text("""
        SELECT
          t.unique_id,
          t.instrument_name,
          t.trade_date,
          t.trade_time,
          t.trade_price,
          t.type,
          t.qty,
          t.strike,
          t.strategy_variant_id,
          t.entry_exit_error,
          t.theoretical_price,
          t.theoretical_time,

          t.account_id::text  AS account_id,
          t.strategy_id::text AS strategy_id,
          t.user_id::text     AS user_id,

          a.name                AS account_name,
          s.strategy_name       AS strategy_name,
          s.portfolio_id,
          p.portfolio           AS portfolio_name,
          p.mode                AS exec_mode,
          u.name                AS user_name
        FROM core.trades t
        LEFT JOIN core.accounts   a ON a.id    = t.account_id
        LEFT JOIN core.strategies s ON s.id    = t.strategy_id
        LEFT JOIN core.portfolios p ON p.id    = s.portfolio_id
        LEFT JOIN core.users      u ON u.id_no = t.user_id
        WHERE t.trade_date BETWEEN :d1 AND :d2
    """)
    with src_engine.connect() as con:
        df = pd.read_sql(q, con, params={"d1": from_date, "d2": to_date})

    if df.empty:
        print(f"No trades found {from_date}..{to_date}")
        return

    for c in ["qty", "trade_price", "theoretical_price", "strike"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["timestamp_ist"] = to_ts_ist(df["trade_date"], df["trade_time"], tz=IST_TZ)
    df["trade_day"]     = df["timestamp_ist"].dt.tz_convert(IST_TZ).dt.date
    df["option_type"]   = df["type"].apply(parse_option_type)
    sign                = np.sign(df["qty"].astype(float))
    df["side"]          = np.where(sign > 0, "BUY", np.where(sign < 0, "SELL", None))
    df["entry_exit"]    = df["entry_exit_error"].astype(str).str.strip().str.title()

    # ---------- CARRY-IN from previous EOD ----------
    carry = fetch_carry_in_positions(dst_engine, from_date)
    if carry is not None and not carry.empty:
        for col in df.columns:
            if col not in carry.columns:
                carry[col] = None
        for col in carry.columns:
            if col not in df.columns:
                df[col] = None
        cols_union = list(df.columns)
        carry = carry.reindex(columns=cols_union)
        df = pd.concat([carry, df[cols_union]], ignore_index=True)

    # ---------- Slippage per fill ----------
    theo_ok = df["theoretical_price"].notna() & (df["theoretical_price"] > 0)
    sign    = np.sign(df["qty"].astype(float))
    df["slip_abs"] = np.where(theo_ok, (df["trade_price"] - df["theoretical_price"]) * sign, np.nan)
    df["slip_pct"] = np.where(theo_ok, 100.0 * df["slip_abs"] / df["theoretical_price"], np.nan)

    # ---------- Write ALL transformed trades (pre-clubbing) ----------
    with dst_engine.begin() as con:
        df_raw = df.copy()
        df_raw["entry_label"] = df_raw.get("entry_label", df_raw["entry_exit_error"])
        cols = ["unique_id","timestamp_ist","trade_day","account_id","strategy_id","user_id",
                "strategy_variant_id","instrument_name","type","option_type","strike","qty",
                "trade_price","side","entry_label","theoretical_price","exec_mode",
                "account_name","strategy_name","portfolio_name"]
        for c in cols:
            if c not in df_raw.columns:
                df_raw[c] = None
        df_raw = df_raw[cols]
        df_raw = df_raw[df_raw["unique_id"].notna()]  # safety
        df_raw.to_sql("_trades_all_stage", con, schema="mart", if_exists="replace", index=False)
        con.execute(text("""
            INSERT INTO mart.trades_all
            (unique_id, timestamp_ist, trade_day, account_id, strategy_id, user_id,
             strategy_variant_id, instrument_name, type, option_type, strike, qty,
             trade_price, side, entry_label, theoretical_price, exec_mode,
             account_name, strategy_name, portfolio_name)
            SELECT
             unique_id, timestamp_ist, trade_day, account_id, strategy_id, user_id,
             strategy_variant_id, instrument_name, type, option_type, strike, qty,
             trade_price, side, entry_label, theoretical_price, exec_mode,
             account_name, strategy_name, portfolio_name
            FROM mart._trades_all_stage
            ON CONFLICT (unique_id) DO UPDATE SET
              timestamp_ist     = EXCLUDED.timestamp_ist,
              trade_day         = EXCLUDED.trade_day,
              qty               = EXCLUDED.qty,
              trade_price       = EXCLUDED.trade_price,
              side              = EXCLUDED.side,
              theoretical_price = EXCLUDED.theoretical_price,
              exec_mode         = EXCLUDED.exec_mode,
              account_name      = EXCLUDED.account_name,
              strategy_name     = EXCLUDED.strategy_name,
              portfolio_name    = EXCLUDED.portfolio_name;
            DROP TABLE mart._trades_all_stage;
        """))

    # ---------- CLUBBING (VWAP) ----------
    club_keys = [
        "timestamp_ist","trade_day",
        "account_id","strategy_id","user_id","strategy_variant_id",
        "instrument_name","type","option_type","strike","exec_mode",
        "account_name","strategy_name","portfolio_name"
    ]
    if not df.empty:
        df["abs_qty"] = df["qty"].abs()
        tmp = df.copy()
        tmp["px_w"] = tmp["trade_price"] * tmp["abs_qty"]
        vwap_df = tmp.groupby(club_keys, dropna=False, as_index=False).agg(
            qty=("qty","sum"),
            w=("abs_qty","sum"),
            pxw=("px_w","sum")
        )
        vwap_df["trade_price"] = np.where(vwap_df["w"] > 0, vwap_df["pxw"] / vwap_df["w"], np.nan)
        vwap_df.drop(columns=["w","pxw"], inplace=True)
        df_agg = vwap_df
        df_agg["side"] = np.where(df_agg["qty"] > 0, "BUY", np.where(df_agg["qty"] < 0, "SELL", None))
    else:
        df_agg = df

    # ---------- Error detection (FIFO-only) ----------
    err_rows = []
    now_ts = datetime.now(timezone.utc)
    group_keys_err = ["trade_day","account_id","strategy_id","user_id","strategy_variant_id","strike"]
    for gkeys, gdf in df_agg.groupby(group_keys_err, dropna=False):
        trade_day, account_id, strategy_id, user_id, variant_id, strike = gkeys
        _, _, _, _, exit_without_open = fifo_realized_pnl(gdf)
        if exit_without_open:
            err_rows.append({
                "trade_day": trade_day,
                "account_id": account_id, "strategy_id": strategy_id, "user_id": user_id,
                "strategy_variant_id": variant_id,
                "issue_code": "EXIT_WITHOUT_OPEN_POSITION",
                "issue_detail": "Detected a closing direction when no opposite open lots existed (FIFO).",
                "affected_unique_ids": list(map(str, gdf.get("unique_id", pd.Series(dtype=str)).tolist())),
                "first_ts": pd.to_datetime(gdf["timestamp_ist"]).min(),
                "last_ts":  pd.to_datetime(gdf["timestamp_ist"]).max(),
                "created_at_utc": now_ts
            })

    errs_out = pd.DataFrame(err_rows)

    # ---------- PnL & Open positions ----------
    pnl_keys = ["trade_day","account_id","strategy_id","user_id",
                "strategy_variant_id","strike",
                "account_name","strategy_name","portfolio_name","exec_mode"]

    results, openpos = [], []
    for keys, sub in df_agg.groupby(pnl_keys, dropna=False):
        (trade_day, account_id, strategy_id, user_id,
         variant_id, strike,
         account_name, strategy_name, portfolio_name, exec_mode) = keys

        realized, fills, net_qty, avg_entry, _ = fifo_realized_pnl(sub)
        results.append({
            "trade_day": trade_day,
            "account_id": account_id, "strategy_id": strategy_id, "user_id": user_id,
            "strategy_variant_id": variant_id, "strike": strike,
            "account_name": account_name, "strategy_name": strategy_name,
            "portfolio_name": portfolio_name, "exec_mode": exec_mode,
            "realized_pnl": realized, "fills": fills
        })

        if net_qty and abs(net_qty) > 1e-12:
            tmax = pd.to_datetime(sub["timestamp_ist"]).max()
            openpos.append({
                "trade_day": trade_day,
                "account_id": account_id, "strategy_id": strategy_id, "user_id": user_id,
                "strategy_variant_id": variant_id, "strike": strike,
                "account_name": account_name, "strategy_name": strategy_name,
                "portfolio_name": portfolio_name, "exec_mode": exec_mode,
                "net_qty": net_qty, "avg_entry_price": avg_entry,
                "last_trade_ts": tmax
            })

    pnl_out  = pd.DataFrame(results)
    open_out = pd.DataFrame(openpos)

    # ---- GUARDRAIL: ensure one row per PnL PK in pandas before staging ----
    if not pnl_out.empty:
        _pnl_keys = ["trade_day","account_id","strategy_id","user_id","strategy_variant_id","strike"]
        agg_map = {
            "account_name": "max",
            "strategy_name": "max",
            "portfolio_name": "max",
            "exec_mode": "max",
            "realized_pnl": "sum",
            "fills": "sum",
        }
        pnl_out = pnl_out.groupby(_pnl_keys, as_index=False, dropna=False).agg(agg_map)

    # Slippage rows (only where theoretical available)
    slip_out = df.loc[theo_ok, [
        "unique_id","timestamp_ist","trade_day","user_id","account_name","strategy_name",
        "instrument_name","option_type","side","trade_price","theoretical_price",
        "slip_abs","slip_pct","exec_mode"
    ]].copy()
    slip_out.rename(columns={"unique_id": "unique_key"}, inplace=True)

    # ---------- Upserts / Inserts ----------
    with dst_engine.begin() as con:
        # Slippage
        if not slip_out.empty:
            slip_out.to_sql("_slip_stage", con, schema="mart", if_exists="replace", index=False)
            con.execute(text("""
                INSERT INTO mart.slippage_events AS t
                (unique_key, timestamp_ist, trade_day, user_id, account_name, strategy_name,
                 instrument_name, option_type, side, trade_price, theoretical_price,
                 slip_abs, slip_pct, exec_mode)
                SELECT unique_key, timestamp_ist, trade_day, user_id, account_name, strategy_name,
                       instrument_name, option_type, side, trade_price, theoretical_price,
                       slip_abs, slip_pct, exec_mode
                FROM mart._slip_stage
                ON CONFLICT (unique_key) DO UPDATE SET
                  timestamp_ist     = EXCLUDED.timestamp_ist,
                  trade_day         = EXCLUDED.trade_day,
                  user_id           = EXCLUDED.user_id,
                  account_name      = EXCLUDED.account_name,
                  strategy_name     = EXCLUDED.strategy_name,
                  instrument_name   = EXCLUDED.instrument_name,
                  option_type       = EXCLUDED.option_type,
                  side              = EXCLUDED.side,
                  trade_price       = EXCLUDED.trade_price,
                  theoretical_price = EXCLUDED.theoretical_price,
                  slip_abs          = EXCLUDED.slip_abs,
                  slip_pct          = EXCLUDED.slip_pct,
                  exec_mode         = EXCLUDED.exec_mode;
                DROP TABLE mart._slip_stage;
            """))

        # PnL (delete-by-day, then INSERT from a dedup CTE)
        if not pnl_out.empty:
            pnl_out.to_sql("_pnl_stage", con, schema="mart", if_exists="replace", index=False)
            con.execute(text("""
                DELETE FROM mart.trade_pnl
                WHERE trade_day IN (SELECT DISTINCT trade_day FROM mart._pnl_stage);

                WITH dedup AS (
                  SELECT
                    trade_day,
                    account_id,
                    strategy_id,
                    user_id,
                    strategy_variant_id,
                    strike,
                    MAX(account_name)   AS account_name,
                    MAX(strategy_name)  AS strategy_name,
                    MAX(portfolio_name) AS portfolio_name,
                    MAX(exec_mode)      AS exec_mode,
                    SUM(realized_pnl)   AS realized_pnl,
                    SUM(fills)          AS fills
                  FROM mart._pnl_stage
                  GROUP BY 1,2,3,4,5,6
                )
                INSERT INTO mart.trade_pnl
                (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike,
                 account_name, strategy_name, portfolio_name, exec_mode, realized_pnl, fills)
                SELECT
                  trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike,
                  account_name, strategy_name, portfolio_name, exec_mode, realized_pnl, fills
                FROM dedup;

                DROP TABLE mart._pnl_stage;
            """))

        # Open positions (DEDUPE + UPSERT)
        if not open_out.empty:
            open_out.to_sql("_open_stage", con, schema="mart", if_exists="replace", index=False)
            con.execute(text("""
                DELETE FROM mart.open_positions_eod
                WHERE trade_day IN (SELECT DISTINCT trade_day FROM mart._open_stage);

                WITH dedup AS (
                  SELECT DISTINCT ON (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike)
                         trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike,
                         account_name, strategy_name, portfolio_name, exec_mode,
                         net_qty, avg_entry_price, last_trade_ts
                  FROM mart._open_stage
                  ORDER BY trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike, last_trade_ts DESC
                )
                INSERT INTO mart.open_positions_eod
                  (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike,
                   account_name, strategy_name, portfolio_name, exec_mode,
                   net_qty, avg_entry_price, last_trade_ts)
                SELECT trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike,
                       account_name, strategy_name, portfolio_name, exec_mode,
                       net_qty, avg_entry_price, last_trade_ts
                FROM dedup
                ON CONFLICT (trade_day, account_id, strategy_id, user_id, strategy_variant_id, strike)
                DO UPDATE SET
                  account_name    = EXCLUDED.account_name,
                  strategy_name   = EXCLUDED.strategy_name,
                  portfolio_name  = EXCLUDED.portfolio_name,
                  exec_mode       = EXCLUDED.exec_mode,
                  net_qty         = EXCLUDED.net_qty,
                  avg_entry_price = EXCLUDED.avg_entry_price,
                  last_trade_ts   = GREATEST(mart.open_positions_eod.last_trade_ts, EXCLUDED.last_trade_ts);

                DROP TABLE mart._open_stage;
            """))

        # Errors (raw)
        if not errs_out.empty:
            stage = errs_out.copy()
            stage["affected_unique_ids"] = stage["affected_unique_ids"].apply(
                lambda v: ",".join(v) if isinstance(v, (list, tuple)) else (None if pd.isna(v) else str(v))
            )
            stage.to_sql("_err_stage", con, schema="ops", if_exists="replace", index=False)
            con.execute(text("""
                INSERT INTO ops.erroneous_trades AS t
                (trade_day, account_id, strategy_id, user_id, strategy_variant_id,
                 issue_code, issue_detail, affected_unique_ids, first_ts, last_ts, created_at_utc)
                SELECT
                  trade_day, account_id, strategy_id, user_id, strategy_variant_id,
                  issue_code, issue_detail,
                  CASE
                    WHEN affected_unique_ids IS NULL OR affected_unique_ids = ''
                      THEN NULL::text[]
                    ELSE string_to_array(affected_unique_ids, ',')::text[]
                  END AS affected_unique_ids,
                  first_ts, last_ts, created_at_utc
                FROM ops._err_stage
                ON CONFLICT (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
                DO UPDATE SET
                  issue_detail        = EXCLUDED.issue_detail,
                  affected_unique_ids = EXCLUDED.affected_unique_ids,
                  last_ts             = EXCLUDED.last_ts,
                  created_at_utc      = EXCLUDED.created_at_utc;
                DROP TABLE ops._err_stage;
            """))

        # Errors (clean)
        if not errs_out.empty:
            clean_err_rows = []
            for e in err_rows:
                mask = (
                    (df_agg["trade_day"] == e["trade_day"]) &
                    (df_agg["account_id"].astype(str) == str(e["account_id"])) &
                    (df_agg["strategy_id"].astype(str) == str(e["strategy_id"])) &
                    (df_agg["user_id"].astype(str) == str(e["user_id"])) &
                    (df_agg["strategy_variant_id"].astype(str) == str(e["strategy_variant_id"]))
                )
                sub = df_agg.loc[mask]

                affected_ids = list(map(str, e.get("affected_unique_ids", []))) if isinstance(e.get("affected_unique_ids"), (list, tuple)) else []
                wrong_fills = len(affected_ids)

                strikes = sorted({float(s) for s in sub["strike"].dropna().unique().tolist()})
                strikes_csv = ",".join(str(int(s)) if abs(s - int(s)) < 1e-9 else str(s) for s in strikes)

                clean_err_rows.append({
                    "trade_day":           e["trade_day"],
                    "account_id":          str(e["account_id"]),
                    "account_name":        sub["account_name"].dropna().iloc[0] if "account_name" in sub and not sub["account_name"].dropna().empty else None,
                    "strategy_id":         str(e["strategy_id"]),
                    "strategy_name":       sub["strategy_name"].dropna().iloc[0] if "strategy_name" in sub and not sub["strategy_name"].dropna().empty else None,
                    "portfolio_name":      sub["portfolio_name"].dropna().iloc[0] if "portfolio_name" in sub and not sub["portfolio_name"].dropna().empty else None,
                    "exec_mode":           sub["exec_mode"].dropna().iloc[0] if "exec_mode" in sub and not sub["exec_mode"].dropna().empty else None,
                    "user_id":             str(e["user_id"]),
                    "strategy_variant_id": str(e["strategy_variant_id"]),
                    "issue_code":          e["issue_code"],
                    "issue_detail":        e["issue_detail"],
                    "wrong_fills":         wrong_fills,
                    "strikes_csv":         strikes_csv,
                    "affected_ids_csv":    ",".join(affected_ids),
                    "first_ts":            e["first_ts"],
                    "last_ts":             e["last_ts"],
                    "created_at_utc":      e.get("created_at_utc"),
                })

            errs_clean_out = pd.DataFrame(clean_err_rows)
            if not errs_clean_out.empty:
                errs_clean_out.to_sql("_err_clean_stage", con, schema="ops", if_exists="replace", index=False)
                con.execute(text("""
                    INSERT INTO ops.erroneous_trades_clean AS t
                    (trade_day, account_id, account_name, strategy_id, strategy_name, portfolio_name, exec_mode,
                     user_id, strategy_variant_id,
                     issue_code, issue_detail, wrong_fills, strikes_csv, affected_ids_csv,
                     first_ts, last_ts, created_at_utc)
                    SELECT trade_day, account_id, account_name, strategy_id, strategy_name, portfolio_name, exec_mode,
                           user_id, strategy_variant_id,
                           issue_code, issue_detail, wrong_fills, strikes_csv, affected_ids_csv,
                           first_ts, last_ts, created_at_utc
                    FROM ops._err_clean_stage
                    ON CONFLICT (trade_day, account_id, strategy_id, user_id, strategy_variant_id, issue_code, first_ts)
                    DO UPDATE SET
                      account_name      = EXCLUDED.account_name,
                      strategy_name     = EXCLUDED.strategy_name,
                      portfolio_name    = EXCLUDED.portfolio_name,
                      exec_mode         = EXCLUDED.exec_mode,
                      issue_detail      = EXCLUDED.issue_detail,
                      wrong_fills       = EXCLUDED.wrong_fills,
                      strikes_csv       = EXCLUDED.strikes_csv,
                      affected_ids_csv  = EXCLUDED.affected_ids_csv,
                      last_ts           = EXCLUDED.last_ts,
                      created_at_utc    = EXCLUDED.created_at_utc;
                    DROP TABLE ops._err_clean_stage;
                """))

    print(f"✅ slippage rows written: {len(slip_out)}")
    print(f"✅ trade rows (realized PnL) written: {len(pnl_out)}")
    print(f"✅ open positions rows written: {len(open_out)}")
    print(f"✅ error flags written: {0 if 'errs_out' not in locals() or errs_out is None else len(errs_out)}")

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

    print("Tip: In Metabase, use ops.v_erroneous_trades_clean, ops.v_erroneous_counts_by_bucket, "
          "mart.v_slippage_daily, mart.v_strategy_pnl, and mart.v_open_positions_eod.")

if __name__ == "__main__":
    main()
