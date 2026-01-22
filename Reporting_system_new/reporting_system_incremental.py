#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reporting_system_incremental.py

INCREMENTAL (single-day) runner.

✅ Same logic as your range rebuild code, but runs ONLY one trade_day:
- Purges ONLY that day from reports tables (NORMAL + ERROR pipelines)
- Seeds FIFO for that day from previous trading day open positions (carry-forward)
- Applies today's trades on top of that seed (ticket-scoped FIFO)
- Builds EOD positions (same mark logic + expiry-close QA fix + prev-day fallback)
- Applies auto-close at/after expiry (same)
- Upserts pnl_daily for that day (same)

✅ Two requested changes added (and nothing else):
1) Safety check: warning if prev_day missing OR prev_day FIFO rows = 0 while prev_day EOD has open positions.
2) Optional CLI flag: --day YYYY-MM-DD (otherwise runs for "today"/latest trading day via trading_calendar).
"""

import argparse
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional, Iterable, Dict, List
from collections import defaultdict
from itertools import groupby

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, URL
from sqlalchemy.exc import OperationalError


# ---------------------------------------------------------------------
# DB CONFIG (TEST DB)  (same style as your file)
# ---------------------------------------------------------------------
#test db
# REPORT_DB_URL = URL.create(
#     drivername="postgresql+psycopg2",
#     username="postgres",
#     password="New@121",      #test DB
#     host="localhost",
#     port=5432,
#     database="postgres",
# )
#main db
REPORT_DB_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password="New@1234",
    host="192.168.18.23",
    port=5432,
    database="postgres",
)



# -------------------------
# Config / CLI
# -------------------------
@dataclass
class Config:
    src_schema: str = "core"
    src_table: str = "trades"
    rpt_schema: str = "reports"
    db_url: URL = REPORT_DB_URL


def parse_args():
    p = argparse.ArgumentParser(description="Run reports pipeline incrementally for a single day.")
    p.add_argument("--day", dest="day", required=False, help="Run for a specific trade day YYYY-MM-DD (optional).")
    return p.parse_args()


# ---------------------------------------------------------------------
# QA tracker (unchanged fields)
# ---------------------------------------------------------------------
class QAStats:
    def __init__(self):
        # NORMAL
        self.missing_marks_by_day_normal = defaultdict(int)
        self.autoclose_by_day_normal = defaultdict(int)
        self.missing_marks_total_normal = 0
        self.autoclose_total_normal = 0
        self.expiry_mark_used_total_normal = 0
        self.fallback_prev_used_total_normal = 0  # prev_mv/prev_qty used

        # ERROR
        self.missing_marks_by_day_error = defaultdict(int)
        self.autoclose_by_day_error = defaultdict(int)
        self.missing_marks_total_error = 0
        self.autoclose_total_error = 0
        self.expiry_mark_used_total_error = 0
        self.fallback_prev_used_total_error = 0  # prev_mv/prev_qty used

        # Trades processed stats
        self.trades_rows_fetched_normal = 0
        self.trades_rows_fetched_error = 0
        self.distinct_tickets_normal = 0
        self.distinct_tickets_error = 0

        # Purge QA (single day now)
        self.purged_fifo_rows_normal = 0
        self.purged_eod_rows_normal = 0
        self.purged_pnl_rows_normal = 0
        self.purged_fifo_rows_error = 0
        self.purged_eod_rows_error = 0
        self.purged_pnl_rows_error = 0

        # Carry-forward inserts
        self.carry_forward_inserts_normal = 0
        self.carry_forward_inserts_error = 0

        # End-of-run open positions counts
        self.open_positions_end_normal = 0
        self.open_tickets_end_normal = 0
        self.stuck_open_after_expiry_end_normal = 0

        self.open_positions_end_error = 0
        self.open_tickets_end_error = 0
        self.stuck_open_after_expiry_end_error = 0

        # Unparsable dates (kept)
        self.unparsable_trade_date_total = 0
        self.unparsable_trade_date_in_window = 0

        # Detailed list of missing marks (no mark and no fallback) (both pipelines)
        self.missing_mark_details: List[Dict] = []


# ---------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------
def run_sql(conn: Connection, sql: str, **params):
    return conn.execute(text(sql), params)


def get_conn(db_url: URL):
    print("[DB] Using DSN:", db_url.render_as_string(hide_password=True))
    eng = create_engine(db_url, future=True, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    try:
        return eng.begin()
    except OperationalError as e:
        print("\n[DB CONNECT ERROR]")
        print(" →", getattr(e, "orig", e))
        raise


# ---------------------------------------------------------------------
# TRADING CALENDAR HELPERS (same logic style)
# ---------------------------------------------------------------------
def get_prev_trading_day(conn: Connection, day: date) -> Optional[date]:
    row = run_sql(
        conn,
        """
        SELECT max(trade_date)
        FROM core.trading_calendar
        WHERE trade_date < :day
          AND is_trading_day = true
        """,
        day=day,
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def is_trading_day(conn: Connection, day: date) -> bool:
    row = run_sql(
        conn,
        """
        SELECT is_trading_day
        FROM core.trading_calendar
        WHERE trade_date = :day
        """,
        day=day,
    ).fetchone()
    return bool(row and row[0] is True)


def get_latest_trading_day_on_or_before(conn: Connection, day: date) -> Optional[date]:
    row = run_sql(
        conn,
        """
        SELECT max(trade_date)
        FROM core.trading_calendar
        WHERE trade_date <= :day
          AND is_trading_day = true
        """,
        day=day,
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def resolve_run_day(conn: Connection, forced_day: Optional[str]) -> date:
    if forced_day:
        return date.fromisoformat(forced_day)

    today = date.today()
    if is_trading_day(conn, today):
        return today

    latest = get_latest_trading_day_on_or_before(conn, today)
    if latest:
        return latest

    # fallback if trading_calendar missing/empty: use today
    print("[WARN] trading_calendar missing/empty; using today as run_day.")
    return today


# ---------------------------------------------------------------------
# TABLE NAME MAP (unchanged)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Tables:
    fifo: str
    eod: str
    pnl: str


def tables_for(mode: str) -> Tables:
    if mode == "ERROR":
        return Tables(
            fifo="errors_fifo_inventory_today",
            eod="errors_eod_positions",
            pnl="errors_pnl_daily",
        )
    return Tables(
        fifo="fifo_inventory_today",
        eod="eod_positions",
        pnl="pnl_daily",
    )


# ---------------------------------------------------------------------
# SCHEMA: FIFO WORK TABLES (unchanged)
# ---------------------------------------------------------------------
def sql_create_fifo_work(cfg: Config, t: Tables) -> str:
    return f"""
create table if not exists "{cfg.rpt_schema}"."{t.fifo}" (
  position_key_trade text not null,
  account_id int not null,
  strategy_id int not null,
  strategy_variant_id int,
  user_id int,
  instrument_name text not null,
  option_type text not null,
  strike numeric not null,
  expiry_date date not null,
  trade_number bigint,
  trade_day date not null,

  eod_net_qty numeric not null,
  avg_cost_fifo numeric,
  realized_pnl_fifo_today numeric default 0,

  primary key (position_key_trade, trade_day)
);
"""


def sql_create_errors_eod_positions(cfg: Config) -> str:
    return f"""
create table if not exists "{cfg.rpt_schema}"."errors_eod_positions" (
  trade_day date not null,
  position_key_trade text not null,
  account_id int not null,
  strategy_id int not null,
  strategy_variant_id int,
  user_id int,
  instrument_name text not null,
  option_type text not null,
  strike numeric not null,
  expiry_date date not null,
  eod_net_qty numeric not null,
  avg_cost_fifo numeric,
  eod_price numeric,
  mv_eod numeric,
  realized_pnl_fifo_today numeric default 0,
  carry_in boolean default false,
  prev_eod_qty numeric,
  prev_mv_eod numeric,
  unrealized_pnl numeric,
  prev_unrealized_pnl numeric,
  primary key (trade_day, position_key_trade)
);
"""


def sql_create_errors_pnl_daily(cfg: Config) -> str:
    return f"""
create table if not exists "{cfg.rpt_schema}"."errors_pnl_daily" (
  trade_day date not null,
  account_id int not null,
  strategy_id int not null,
  strategy_variant_id int,
  user_id int,
  realized_cash_today numeric,
  unrealized_change numeric,
  total_pnl_day numeric,
  cum_total_pnl numeric,
  primary key (trade_day, account_id, strategy_id, strategy_variant_id, user_id)
);
"""


# ---------------------------------------------------------------------
# SELECT TRADES (unchanged query; caller will pass d_from=d_to=run_day)
# ---------------------------------------------------------------------
def sql_select_trades(cfg: Config, mode: str) -> str:
    if mode == "ERROR":
        where_mode = """
          and upper(coalesce(t.entry_exit_error::text,'')) = 'ERROR'
        """
    else:
        where_mode = """
          and (
                t.entry_exit_error is null
             or upper(t.entry_exit_error::text) in ('ENTRY','EXIT')
          )
        """

    return f"""
select
  t.account_id,
  t.strategy_id,
  t.strategy_variant_id,
  t.user_id,
  t.instrument_name,
  upper((t.type)::text)                      as option_type,
  (t.strike)::numeric                        as strike,
  (t.expiry_date)::date                      as expiry_date,
  (t.trade_number)::bigint                   as trade_number,
  t.trade_time                               as trade_ts,
  to_date((t.trade_date)::text,'YYYY-MM-DD') as trade_day,
  (t.qty)::numeric                           as qty,
  (t.trade_price)::numeric                   as trade_price,
  upper(coalesce(t.entry_exit_error::text,'')) as entry_exit_error,
  concat_ws('||',
    t.account_id,
    t.strategy_id,
    coalesce(t.strategy_variant_id,0),
    t.user_id,
    t.instrument_name,
    upper((t.type)::text),
    coalesce((t.strike)::text,''),
    coalesce((t.expiry_date)::text,''),
    coalesce((t.trade_number)::text,'0')
  ) as position_key_trade
from "{cfg.src_schema}"."{cfg.src_table}" t
where to_date((t.trade_date)::text,'YYYY-MM-DD') between :d_from and :d_to
{where_mode}
order by
  position_key_trade,
  to_date((t.trade_date)::text,'YYYY-MM-DD'),
  t.trade_time,
  t.trade_number;
"""


# ---------------------------------------------------------------------
# UPSERT FIFO DAY (unchanged)
# ---------------------------------------------------------------------
def sql_upsert_fifo_day(cfg: Config, t: Tables) -> str:
    return f"""
insert into "{cfg.rpt_schema}"."{t.fifo}" as f(
  position_key_trade,
  account_id,
  strategy_id,
  strategy_variant_id,
  user_id,
  instrument_name,
  option_type,
  strike,
  expiry_date,
  trade_number,
  trade_day,
  eod_net_qty,
  avg_cost_fifo,
  realized_pnl_fifo_today
)
values (
  :position_key_trade,
  :account_id,
  :strategy_id,
  :strategy_variant_id,
  :user_id,
  :instrument_name,
  :option_type,
  :strike,
  :expiry_date,
  :trade_number,
  :trade_day,
  :eod_net_qty,
  :avg_cost_fifo,
  :realized_pnl_fifo_today
)
on conflict (position_key_trade, trade_day) do update set
  eod_net_qty             = excluded.eod_net_qty,
  avg_cost_fifo           = excluded.avg_cost_fifo,
  realized_pnl_fifo_today = excluded.realized_pnl_fifo_today;
"""


# ---------------------------------------------------------------------
# PRICE LOOKUPS (unchanged)
# ---------------------------------------------------------------------
def sql_best_mark(cfg: Config) -> str:
    return """
select price
from core.option_eod_prices_all
where trade_date      = :day
  and instrument_name = :instr
  and option_type     = :otype
  and strike          = :strike
  and expiry_date     = :expiry
order by retrieved_at desc
limit 1;
"""


def sql_best_expiry_close(cfg: Config) -> str:
    return """
select price
from core.option_eod_prices_all
where instrument_name = :instr
  and option_type     = :otype
  and strike          = :strike
  and expiry_date     = :expiry
  and trade_date      = :expiry
order by retrieved_at desc
limit 1;
"""


def sql_best_expiry_fallback(cfg: Config) -> str:
    return """
select price
from core.option_eod_prices_all
where instrument_name = :instr
  and option_type     = :otype
  and strike          = :strike
  and expiry_date     = :expiry
  and trade_date     <= :expiry
order by trade_date desc, retrieved_at desc
limit 1;
"""


def get_best_mark(conn: Connection, cfg: Config, day: date,
                  instr: str, otype: str, strike: float, expiry) -> Optional[float]:
    row = run_sql(conn, sql_best_mark(cfg),
                  day=day, instr=instr, otype=otype, strike=strike, expiry=expiry).fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])


def get_best_expiry_close(conn: Connection, cfg: Config,
                          instr: str, otype: str, strike: float, expiry) -> Optional[float]:
    row = run_sql(conn, sql_best_expiry_close(cfg),
                  instr=instr, otype=otype, strike=strike, expiry=expiry).fetchone()
    if row and row[0] is not None:
        return float(row[0])
    row = run_sql(conn, sql_best_expiry_fallback(cfg),
                  instr=instr, otype=otype, strike=strike, expiry=expiry).fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])


# ---------------------------------------------------------------------
# UPSERT EOD POSITIONS (unchanged)
# ---------------------------------------------------------------------
def sql_upsert_eod_pos(cfg: Config, t: Tables) -> str:
    return f"""
insert into "{cfg.rpt_schema}"."{t.eod}" as x(
  trade_day,
  position_key_trade,
  account_id,
  strategy_id,
  strategy_variant_id,
  user_id,
  instrument_name,
  option_type,
  strike,
  expiry_date,
  eod_net_qty,
  avg_cost_fifo,
  eod_price,
  mv_eod,
  realized_pnl_fifo_today,
  carry_in,
  prev_eod_qty,
  prev_mv_eod,
  unrealized_pnl,
  prev_unrealized_pnl
)
values (
  :trade_day,
  :position_key_trade,
  :account_id,
  :strategy_id,
  :strategy_variant_id,
  :user_id,
  :instrument_name,
  :option_type,
  :strike,
  :expiry_date,
  :eod_net_qty,
  :avg_cost_fifo,
  :eod_price,
  :mv_eod,
  :realized_pnl_fifo_today,
  :carry_in,
  :prev_eod_qty,
  :prev_mv_eod,
  :unrealized_pnl,
  :prev_unrealized_pnl
)
on conflict (trade_day, position_key_trade)
do update set
  eod_net_qty             = excluded.eod_net_qty,
  avg_cost_fifo           = excluded.avg_cost_fifo,
  eod_price               = excluded.eod_price,
  mv_eod                  = excluded.mv_eod,
  realized_pnl_fifo_today = excluded.realized_pnl_fifo_today,
  carry_in                = excluded.carry_in,
  prev_eod_qty            = excluded.prev_eod_qty,
  prev_mv_eod             = excluded.prev_mv_eod,
  unrealized_pnl          = excluded.unrealized_pnl,
  prev_unrealized_pnl     = excluded.prev_unrealized_pnl;
"""


# ---------------------------------------------------------------------
# UPSERT DAILY PNL (unchanged)
# ---------------------------------------------------------------------
def sql_upsert_pnl_daily(cfg: Config, t: Tables) -> str:
    return f"""
insert into "{cfg.rpt_schema}"."{t.pnl}" as d(
  trade_day,
  account_id,
  strategy_id,
  strategy_variant_id,
  user_id,
  realized_cash_today,
  unrealized_change,
  total_pnl_day,
  cum_total_pnl
)
select
  :day as trade_day,
  account_id,
  strategy_id,
  strategy_variant_id,
  user_id,
  sum(realized_pnl_fifo_today)                                       as realized_cash_today,
  sum(coalesce(unrealized_pnl,0) - coalesce(prev_unrealized_pnl,0))  as unrealized_change,
  sum(
      realized_pnl_fifo_today
      + (coalesce(unrealized_pnl,0) - coalesce(prev_unrealized_pnl,0))
  )                                                                  as total_pnl_day,
  null::numeric
from "{cfg.rpt_schema}"."{t.eod}"
where trade_day = :day
group by 2,3,4,5
on conflict (trade_day, account_id, strategy_id, strategy_variant_id, user_id)
do update set
  realized_cash_today = excluded.realized_cash_today,
  unrealized_change   = excluded.unrealized_change,
  total_pnl_day       = excluded.total_pnl_day;
"""


def upsert_pnl_daily(conn: Connection, cfg: Config, t: Tables, day: date):
    run_sql(conn, sql_upsert_pnl_daily(cfg, t), day=day)


# ---------------------------------------------------------------------
# FIFO ENGINE (same logic; now supports seeding)
# ---------------------------------------------------------------------
class FifoState:
    __slots__ = ("q_running", "cost_total", "avg_cost", "realized_today", "last_day")

    def __init__(self):
        self.q_running: float = 0.0
        self.cost_total: float = 0.0
        self.avg_cost: Optional[float] = None
        self.realized_today: float = 0.0
        self.last_day: Optional[date] = None


def sql_upsert_fifo_day_call(conn: Connection, cfg: Config, t: Tables, r: Dict, st: FifoState):
    run_sql(conn, sql_upsert_fifo_day(cfg, t),
        position_key_trade      = r["position_key_trade"],
        account_id              = r["account_id"],
        strategy_id             = r["strategy_id"],
        strategy_variant_id     = r["strategy_variant_id"],
        user_id                 = r["user_id"],
        instrument_name         = r["instrument_name"],
        option_type             = r["option_type"],
        strike                  = r["strike"],
        expiry_date             = r["expiry_date"],
        trade_number            = r["trade_number"],
        trade_day               = r["trade_day"],
        eod_net_qty             = st.q_running,
        avg_cost_fifo           = st.avg_cost,
        realized_pnl_fifo_today = st.realized_today
    )


def load_seed_state_from_prev_day(conn: Connection, cfg: Config, t: Tables, prev_day: Optional[date], pk: str) -> FifoState:
    st = FifoState()
    if prev_day is None:
        return st

    row = run_sql(conn, f"""
        select eod_net_qty, avg_cost_fifo
        from "{cfg.rpt_schema}"."{t.fifo}"
        where trade_day = :d
          and position_key_trade = :pk
        limit 1
    """, d=prev_day, pk=pk).fetchone()

    if not row:
        row = run_sql(conn, f"""
            select eod_net_qty, avg_cost_fifo
            from "{cfg.rpt_schema}"."{t.eod}"
            where trade_day = :d
              and position_key_trade = :pk
            limit 1
        """, d=prev_day, pk=pk).fetchone()

    if row and row[0] is not None:
        st.q_running = float(row[0])
        st.avg_cost = float(row[1]) if row[1] is not None else None
        if st.q_running != 0.0 and st.avg_cost is not None:
            st.cost_total = st.avg_cost * abs(st.q_running)
        else:
            st.cost_total = 0.0

    return st


def process_bucket_seeded(conn: Connection, cfg: Config, t: Tables, rows: Iterable[Dict], seed: FifoState):
    st = seed
    for r in rows:
        if st.last_day != r["trade_day"]:
            st.realized_today = 0.0
            st.last_day = r["trade_day"]

        q = float(r["qty"])
        p = float(r["trade_price"])

        if q > 0:  # buy
            if st.q_running >= 0:
                st.cost_total += q * p
                st.q_running  += q
            else:
                use_qty = min(abs(st.q_running), q)
                entry_px = st.avg_cost or 0.0
                st.realized_today += (entry_px - p) * use_qty
                st.cost_total     -= entry_px * use_qty
                st.q_running      += use_qty
                leftover = q - use_qty
                if leftover > 0:
                    st.cost_total += leftover * p
                    st.q_running  += leftover

        elif q < 0:  # sell
            if st.q_running <= 0:
                st.cost_total += abs(q) * p
                st.q_running  += q
            else:
                use_qty = min(st.q_running, abs(q))
                entry_px = st.avg_cost or 0.0
                st.realized_today += (p - entry_px) * use_qty
                st.cost_total     -= entry_px * use_qty
                st.q_running      -= use_qty
                leftover = abs(q) - use_qty
                if leftover > 0:
                    st.cost_total += leftover * p
                    st.q_running  -= leftover

        if st.q_running != 0:
            st.avg_cost = st.cost_total / abs(st.q_running)
        else:
            st.avg_cost = None
            st.cost_total = 0.0

        sql_upsert_fifo_day_call(conn, cfg, t, r, st)


# ---------------------------------------------------------------------
# CARRY-FORWARD (unchanged SQL + behavior; used to seed run_day)
# ---------------------------------------------------------------------
def sql_fifo_carry_forward(cfg: Config, t: Tables) -> str:
    return f"""
insert into "{cfg.rpt_schema}"."{t.fifo}" (
    position_key_trade,
    account_id,
    strategy_id,
    strategy_variant_id,
    user_id,
    instrument_name,
    option_type,
    strike,
    expiry_date,
    trade_number,
    trade_day,
    eod_net_qty,
    avg_cost_fifo,
    realized_pnl_fifo_today
)
select
    e.position_key_trade,
    e.account_id,
    e.strategy_id,
    e.strategy_variant_id,
    e.user_id,
    e.instrument_name,
    e.option_type,
    e.strike,
    e.expiry_date,
    null::bigint as trade_number,
    :day         as trade_day,
    e.eod_net_qty,
    e.avg_cost_fifo,
    0::numeric   as realized_pnl_fifo_today
from "{cfg.rpt_schema}"."{t.eod}" e
where e.trade_day = :prev_day
  and e.eod_net_qty <> 0
  and e.expiry_date > :prev_day
  and not exists (
        select 1
        from "{cfg.rpt_schema}"."{t.fifo}" f
        where f.trade_day = :day
          and f.position_key_trade = e.position_key_trade
  );
"""


def ensure_fifo_carry_forward(conn: Connection, cfg: Config, t: Tables, day: date) -> int:
    prev_trading_day = get_prev_trading_day(conn, day)
    if not prev_trading_day:
        return 0
    res = run_sql(conn, sql_fifo_carry_forward(cfg, t), day=day, prev_day=prev_trading_day)
    return int(res.rowcount or 0)


# ---------------------------------------------------------------------
# BUILD EOD POS (unchanged; includes QA fix)
# ---------------------------------------------------------------------
def build_eod_positions_for_day(conn: Connection, cfg: Config, t: Tables, mode: str, day: date, qa: QAStats) -> int:
    missing_marks = 0

    fifo_rows = run_sql(conn, f"""
        select *
        from "{cfg.rpt_schema}"."{t.fifo}"
        where trade_day = :day
        order by position_key_trade
    """, day=day).mappings().all()

    for r in fifo_rows:
        pk = r["position_key_trade"]

        prev = run_sql(conn, f"""
            select eod_net_qty, mv_eod, unrealized_pnl
            from "{cfg.rpt_schema}"."{t.eod}"
            where position_key_trade = :pk
              and trade_day = (
                select max(trade_day)
                from "{cfg.rpt_schema}"."{t.eod}"
                where position_key_trade = :pk
                  and trade_day < :day
              )
        """, pk=pk, day=day).fetchone()

        prev_qty = float(prev[0]) if prev and prev[0] is not None else None
        prev_mv  = float(prev[1]) if prev and prev[1] is not None else None
        prev_unreal = float(prev[2]) if prev and prev[2] is not None else 0.0
        carry_in = (prev_qty is not None and prev_qty != 0.0)

        instr = r["instrument_name"]
        otype = r["option_type"]
        strike = float(r["strike"])
        expiry = r["expiry_date"]

        eod_price = get_best_mark(conn, cfg, day, instr, otype, strike, expiry)

        used_expiry_mark = False
        if eod_price is None and day >= expiry:
            expiry_px = get_best_expiry_close(conn, cfg, instr, otype, strike, expiry)
            if expiry_px is not None:
                eod_price = expiry_px
                used_expiry_mark = True
                if mode == "ERROR":
                    qa.expiry_mark_used_total_error += 1
                else:
                    qa.expiry_mark_used_total_normal += 1

        used_prev_fallback = False
        if eod_price is None:
            if prev_qty is not None and prev_mv is not None and prev_qty != 0.0:
                eod_price = prev_mv / prev_qty
                used_prev_fallback = True
                if mode == "ERROR":
                    qa.fallback_prev_used_total_error += 1
                else:
                    qa.fallback_prev_used_total_normal += 1
            else:
                missing_marks += 1
                qa.missing_mark_details.append({
                    "mode": mode,
                    "trade_day": day,
                    "instrument_name": instr,
                    "option_type": otype,
                    "strike": strike,
                    "expiry_date": expiry,
                    "account_id": r["account_id"],
                    "strategy_id": r["strategy_id"],
                    "strategy_variant_id": r["strategy_variant_id"],
                    "user_id": r["user_id"],
                    "position_key_trade": pk,
                    "is_expiry_or_after": bool(day >= expiry),
                    "attempted_expiry_close": bool(day >= expiry),
                    "expiry_close_used": used_expiry_mark,
                    "prev_fallback_used": used_prev_fallback,
                })

        qty = float(r["eod_net_qty"])
        avg_cost = float(r["avg_cost_fifo"]) if r["avg_cost_fifo"] is not None else None

        mv_eod = (eod_price * qty) if (eod_price is not None) else 0.0

        if eod_price is not None and avg_cost is not None and qty != 0.0:
            unrealized_pnl = (eod_price - avg_cost) * qty
        else:
            unrealized_pnl = 0.0

        run_sql(conn, sql_upsert_eod_pos(cfg, t),
            trade_day               = day,
            position_key_trade      = pk,
            account_id              = r["account_id"],
            strategy_id             = r["strategy_id"],
            strategy_variant_id     = r["strategy_variant_id"],
            user_id                 = r["user_id"],
            instrument_name         = instr,
            option_type             = otype,
            strike                  = r["strike"],
            expiry_date             = expiry,
            eod_net_qty             = r["eod_net_qty"],
            avg_cost_fifo           = r["avg_cost_fifo"],
            eod_price               = eod_price,
            mv_eod                  = mv_eod,
            realized_pnl_fifo_today = r["realized_pnl_fifo_today"],
            carry_in                = carry_in,
            prev_eod_qty            = prev_qty,
            prev_mv_eod             = prev_mv,
            unrealized_pnl          = unrealized_pnl,
            prev_unrealized_pnl     = prev_unreal
        )

    return missing_marks


# ---------------------------------------------------------------------
# AUTO-CLOSE AFTER / AT EXPIRY (unchanged)
# ---------------------------------------------------------------------
def apply_autoclose_after_expiry(conn: Connection, cfg: Config, t: Tables, day: date) -> int:
    rows = run_sql(conn, f"""
        select
          position_key_trade,
          instrument_name,
          option_type,
          strike,
          expiry_date,
          eod_net_qty,
          avg_cost_fifo
        from "{cfg.rpt_schema}"."{t.eod}"
        where trade_day = :day
          and eod_net_qty <> 0
          and trade_day >= expiry_date
    """, day=day).mappings().all()

    applied = 0

    for r in rows:
        expiry_px = get_best_expiry_close(
            conn, cfg,
            r["instrument_name"],
            r["option_type"],
            float(r["strike"]),
            r["expiry_date"]
        )
        if expiry_px is None:
            continue

        qty = float(r["eod_net_qty"])
        ac = float(r["avg_cost_fifo"] or 0.0)

        if qty > 0:
            realized_bump = (expiry_px - ac) * qty
        else:
            realized_bump = (ac - expiry_px) * abs(qty)

        run_sql(conn, f"""
            update "{cfg.rpt_schema}"."{t.eod}"
               set realized_pnl_fifo_today = coalesce(realized_pnl_fifo_today, 0) + :add_real,
                   eod_price               = :px,
                   mv_eod                  = 0,
                   eod_net_qty             = 0,
                   avg_cost_fifo           = null,
                   unrealized_pnl          = 0
             where trade_day = :day
               and position_key_trade = :pk
        """, add_real=realized_bump, px=expiry_px, day=day, pk=r["position_key_trade"])

        applied += 1

    return applied


# ---------------------------------------------------------------------
# QA: unparsable trade_date (kept; for incremental we treat run_day as window)
# ---------------------------------------------------------------------
def sql_count_unparsable_trade_date_total(cfg: Config) -> str:
    return f"""
select count(*)::bigint
from "{cfg.src_schema}"."{cfg.src_table}" t
where to_date(t.trade_date::text, 'YYYY-MM-DD') is null;
"""


def sql_count_unparsable_trade_date_in_window(cfg: Config) -> str:
    return f"""
select count(*)::bigint
from "{cfg.src_schema}"."{cfg.src_table}" t
where to_date(t.trade_date::text, 'YYYY-MM-DD') is null
  and t.trade_date::text >= :d_from_text
  and t.trade_date::text <= :d_to_text;
"""


def count_unparsable_trade_dates(conn: Connection, cfg: Config, qa: QAStats, d_from: date, d_to: date):
    qa.unparsable_trade_date_total = run_sql(
        conn, sql_count_unparsable_trade_date_total(cfg)
    ).scalar_one()
    qa.unparsable_trade_date_in_window = run_sql(
        conn,
        sql_count_unparsable_trade_date_in_window(cfg),
        d_from_text=d_from.isoformat(),
        d_to_text=d_to.isoformat()
    ).scalar_one()


# ---------------------------------------------------------------------
# PURGE SINGLE DAY (same dependency order; only that day)
# ---------------------------------------------------------------------
def purge_reports_day(conn: Connection, cfg: Config, t: Tables, day: date) -> Dict[str, int]:
    res_pnl = run_sql(conn, f'DELETE FROM "{cfg.rpt_schema}"."{t.pnl}" WHERE trade_day = :d', d=day)
    res_eod = run_sql(conn, f'DELETE FROM "{cfg.rpt_schema}"."{t.eod}" WHERE trade_day = :d', d=day)
    res_fifo = run_sql(conn, f'DELETE FROM "{cfg.rpt_schema}"."{t.fifo}" WHERE trade_day = :d', d=day)
    return {
        "pnl": int(res_pnl.rowcount or 0),
        "eod": int(res_eod.rowcount or 0),
        "fifo": int(res_fifo.rowcount or 0),
    }


# ---------------------------------------------------------------------
# EXTRA QA QUERIES (unchanged)
# ---------------------------------------------------------------------
def count_open_positions_end(conn: Connection, cfg: Config, eod_table: str, run_to: date) -> Dict[str, int]:
    open_rows = int(run_sql(
        conn,
        f'SELECT count(*)::bigint FROM "{cfg.rpt_schema}"."{eod_table}" WHERE trade_day = :d AND eod_net_qty <> 0',
        d=run_to
    ).scalar_one())

    open_tickets = int(run_sql(
        conn,
        f'SELECT count(distinct position_key_trade)::bigint FROM "{cfg.rpt_schema}"."{eod_table}" WHERE trade_day = :d AND eod_net_qty <> 0',
        d=run_to
    ).scalar_one())

    stuck = int(run_sql(
        conn,
        f'''
        SELECT count(*)::bigint
        FROM "{cfg.rpt_schema}"."{eod_table}"
        WHERE trade_day = :d
          AND eod_net_qty <> 0
          AND trade_day > expiry_date
        ''',
        d=run_to
    ).scalar_one())

    return {"open_rows": open_rows, "open_tickets": open_tickets, "stuck_after_expiry": stuck}


# ---------------------------------------------------------------------
# CHANGE #1 (requested): safety warning helpers
# ---------------------------------------------------------------------
def count_fifo_rows_day(conn: Connection, cfg: Config, fifo_table: str, day: date) -> int:
    return int(run_sql(
        conn,
        f'SELECT count(*)::bigint FROM "{cfg.rpt_schema}"."{fifo_table}" WHERE trade_day = :d',
        d=day
    ).scalar_one())


def print_seed_safety_warnings(conn: Connection, cfg: Config, run_day: date, prev_day: Optional[date]):
    if prev_day is None:
        print(f"[WARN][SEED] prev_trading_day is missing for run_day={run_day}. "
              f"FIFO seeding cannot happen; carry positions may be wrong.")
        return

    fifo_n = count_fifo_rows_day(conn, cfg, "fifo_inventory_today", prev_day)
    open_prev_eod = int(run_sql(
        conn,
        f'''
        SELECT count(*)::bigint
        FROM "{cfg.rpt_schema}"."eod_positions"
        WHERE trade_day = :d
          AND eod_net_qty <> 0
        ''',
        d=prev_day
    ).scalar_one())

    if fifo_n == 0 and open_prev_eod > 0:
        print(f"[WARN][SEED] Seed looks missing: fifo_inventory_today has 0 rows on {prev_day} "
              f"but eod_positions has {open_prev_eod} open rows. "
              f"Incremental results for {run_day} may be wrong until you run the range rebuild.")


# ---------------------------------------------------------------------
# RUN ONE PIPELINE (NORMAL or ERROR) for a SINGLE DAY
# ---------------------------------------------------------------------
def run_one_mode_incremental(conn: Connection, cfg: Config, qa: QAStats, mode: str, run_day: date, prev_day: Optional[date]):
    t = tables_for(mode)

    # Ensure tables exist (same)
    run_sql(conn, sql_create_fifo_work(cfg, t))
    if mode == "ERROR":
        run_sql(conn, sql_create_errors_eod_positions(cfg))
        run_sql(conn, sql_create_errors_pnl_daily(cfg))

    # Purge only run_day (idempotent intra-day reruns)
    purged = purge_reports_day(conn, cfg, t, run_day)
    if mode == "ERROR":
        qa.purged_fifo_rows_error += purged["fifo"]
        qa.purged_eod_rows_error += purged["eod"]
        qa.purged_pnl_rows_error += purged["pnl"]
    else:
        qa.purged_fifo_rows_normal += purged["fifo"]
        qa.purged_eod_rows_normal += purged["eod"]
        qa.purged_pnl_rows_normal += purged["pnl"]

    # Seed FIFO for run_day from prev_day open positions (same carry-forward logic)
    inserted = ensure_fifo_carry_forward(conn, cfg, t, run_day)
    if mode == "ERROR":
        qa.carry_forward_inserts_error += inserted
    else:
        qa.carry_forward_inserts_normal += inserted

    # Pull trades only for run_day (same SQL)
    rows = run_sql(
        conn,
        sql_select_trades(cfg, mode),
        d_from=run_day,
        d_to=run_day
    ).mappings().all()

    if mode == "ERROR":
        qa.trades_rows_fetched_error += len(rows)
        qa.distinct_tickets_error += len(set(r["position_key_trade"] for r in rows))
    else:
        qa.trades_rows_fetched_normal += len(rows)
        qa.distinct_tickets_normal += len(set(r["position_key_trade"] for r in rows))

    # Apply ticket-scoped FIFO for today's trades, seeded from prev_day state
    for pk, group in groupby(rows, key=lambda r: r["position_key_trade"]):
        seed = load_seed_state_from_prev_day(conn, cfg, t, prev_day, pk)
        process_bucket_seeded(conn, cfg, t, group, seed)

    # Build EOD for run_day (same)
    missing = build_eod_positions_for_day(conn, cfg, t, mode, run_day, qa)
    if mode == "ERROR":
        qa.missing_marks_by_day_error[run_day] += missing
        qa.missing_marks_total_error += missing
    else:
        qa.missing_marks_by_day_normal[run_day] += missing
        qa.missing_marks_total_normal += missing

    # Auto-close (same)
    closed = apply_autoclose_after_expiry(conn, cfg, t, run_day)
    if mode == "ERROR":
        qa.autoclose_by_day_error[run_day] += closed
        qa.autoclose_total_error += closed
    else:
        qa.autoclose_by_day_normal[run_day] += closed
        qa.autoclose_total_normal += closed

    # PnL for run_day (same)
    upsert_pnl_daily(conn, cfg, t, run_day)


# ---------------------------------------------------------------------
# DRIVER (incremental)
# ---------------------------------------------------------------------
def run_pipeline_incremental(cfg: Config, forced_day: Optional[str] = None):
    qa = QAStats()

    with get_conn(cfg.db_url) as conn:
        run_day = resolve_run_day(conn, forced_day)
        prev_day = get_prev_trading_day(conn, run_day)

        print(f"[INCREMENTAL] run_day={run_day} | prev_trading_day={prev_day}")

        # CHANGE #1: tiny safety warning check (no logic change)
        print_seed_safety_warnings(conn, cfg, run_day, prev_day)

        # QA counts (kept)
        count_unparsable_trade_dates(conn, cfg, qa, run_day, run_day)

        # NORMAL
        run_one_mode_incremental(conn, cfg, qa, mode="NORMAL", run_day=run_day, prev_day=prev_day)

        # ERROR
        run_one_mode_incremental(conn, cfg, qa, mode="ERROR", run_day=run_day, prev_day=prev_day)

        # End-of-run open positions QA (same style)
        tn = tables_for("NORMAL")
        te = tables_for("ERROR")

        end_n = count_open_positions_end(conn, cfg, tn.eod, run_day)
        qa.open_positions_end_normal = end_n["open_rows"]
        qa.open_tickets_end_normal = end_n["open_tickets"]
        qa.stuck_open_after_expiry_end_normal = end_n["stuck_after_expiry"]

        end_e = count_open_positions_end(conn, cfg, te.eod, run_day)
        qa.open_positions_end_error = end_e["open_rows"]
        qa.open_tickets_end_error = end_e["open_tickets"]
        qa.stuck_open_after_expiry_end_error = end_e["stuck_after_expiry"]

    # QA summary (same style)
    print("\n====== QA SUMMARY (INCREMENTAL) ======")
    print(f"Run trade_day                : {run_day}")
    print(f"Prev trading day (seed)      : {prev_day}")
    print(f"Unparsable trade_date rows   : total={qa.unparsable_trade_date_total} | in_day={qa.unparsable_trade_date_in_window}")

    print("\n--- NORMAL (ENTRY/EXIT + NULL treated as normal) ---")
    print(f"Trades fetched               : {qa.trades_rows_fetched_normal}")
    print(f"Distinct tickets (keys)      : {qa.distinct_tickets_normal}")
    print(f"Purged rows (fifo/eod/pnl)   : {qa.purged_fifo_rows_normal} / {qa.purged_eod_rows_normal} / {qa.purged_pnl_rows_normal}")
    print(f"Carry-forward inserts        : {qa.carry_forward_inserts_normal}")
    print(f"Auto-closes applied          : {qa.autoclose_total_normal}")
    print(f"Expiry-close marks used      : {qa.expiry_mark_used_total_normal}")
    print(f"Prev-day fallback marks used : {qa.fallback_prev_used_total_normal}")
    print(f"Missing EOD marks (final)    : {qa.missing_marks_total_normal}")
    print(f"Open positions @ end         : rows={qa.open_positions_end_normal} | tickets={qa.open_tickets_end_normal}")
    print(f"Stuck open after expiry @ end: {qa.stuck_open_after_expiry_end_normal}")

    print("\n--- ERROR (ERROR-only; isolated tables) ---")
    print(f"Trades fetched               : {qa.trades_rows_fetched_error}")
    print(f"Distinct tickets (keys)      : {qa.distinct_tickets_error}")
    print(f"Purged rows (fifo/eod/pnl)   : {qa.purged_fifo_rows_error} / {qa.purged_eod_rows_error} / {qa.purged_pnl_rows_error}")
    print(f"Carry-forward inserts        : {qa.carry_forward_inserts_error}")
    print(f"Auto-closes applied          : {qa.autoclose_total_error}")
    print(f"Expiry-close marks used      : {qa.expiry_mark_used_total_error}")
    print(f"Prev-day fallback marks used : {qa.fallback_prev_used_total_error}")
    print(f"Missing EOD marks (final)    : {qa.missing_marks_total_error}")
    print(f"Open positions @ end         : rows={qa.open_positions_end_error} | tickets={qa.open_tickets_end_error}")
    print(f"Stuck open after expiry @ end: {qa.stuck_open_after_expiry_end_error}")

    total_missing = qa.missing_marks_total_normal + qa.missing_marks_total_error
    if total_missing > 0:
        print("\n--- SAMPLE MISSING MARKS (up to 20) ---")
        for rec in qa.missing_mark_details[:20]:
            print(
                f"[{rec['mode']}] day={rec['trade_day']} "
                f"instr={rec['instrument_name']} {rec['option_type']} K={rec['strike']} exp={rec['expiry_date']} "
                f"| acct={rec['account_id']} strat={rec['strategy_id']} var={rec['strategy_variant_id']} user={rec['user_id']} "
                f"| key={rec['position_key_trade']} "
                f"| expiry_or_after={rec['is_expiry_or_after']}"
            )

    print("\n[OK] Incremental pipeline completed.")


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()
    run_pipeline_incremental(cfg, forced_day=args.day)
