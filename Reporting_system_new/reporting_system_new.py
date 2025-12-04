#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reporting_system_new.py (current behavior)

Source: core.trades

Outputs (schema: reports)
NORMAL pipeline:
  - fifo_inventory_today  (PK: position_key_trade, trade_day)
  - eod_positions         (PK: trade_day, position_key_trade)
  - pnl_daily
ERROR pipeline (fully isolated):
  - errors_fifo_inventory_today
  - errors_eod_positions
  - errors_pnl_daily

Key points implemented
1) Ticket-scoped FIFO per position_key_trade (includes trade_number in the key)
   - Produces eod_net_qty, avg_cost_fifo, realized_pnl_fifo_today per day.

2) Two pipelines split by core.trades.entry_exit_error
   - NORMAL: entry_exit_error IN ('ENTRY','EXIT') or NULL.
   - ERROR : entry_exit_error = 'ERROR'.
   - ERROR never flows into pnl_daily (it stays in errors_* tables).

3) Rebuild is idempotent every run
   - rebuild_from = prev trading day of requested start_date (or start_date).
   - Deletes and rebuilds FIFO/EOD/PNL for [rebuild_from .. end_date] for BOTH pipelines.

4) Trading-calendar driven daily loop + carry-forward
   - Uses core.trading_calendar trading days (fallback: all calendar days).
   - Carries forward open positions from prior trading day (only if expiry_date > prev_day).

5) EOD marking + missing-mark logic
   - Primary: core.option_eod_prices_all for that trade_day (latest retrieved_at).
   - QA fix: if day >= expiry_date and primary missing → try expiry-close mark first.
   - Fallback: previous implied price (prev_mv / prev_qty).
   - If still missing: mv_eod=0, unrealized_pnl=0, logged in QA.

6) Auto-close at/after expiry (realize on expiry mark)
   - If trade_day >= expiry_date and expiry mark exists → realize PnL and set net qty to 0.

7) Daily PnL aggregation
   - realized_cash_today = sum(realized_pnl_fifo_today)
   - unrealized_change   = sum(unrealized_pnl - prev_unrealized_pnl)
   - total_pnl_day        = realized + unrealized_change
   - No fees/charges applied.

QA at end
- Trades fetched + distinct tickets (NORMAL/ERROR), purge + window row counts, carry-forward inserts,
  expiry-mark usage, fallback usage, missing marks, open/stuck positions at end, and sample missing marks
  with acct/strategy/variant/user + position_key_trade.
"""


import argparse
from dataclasses import dataclass
from datetime import date
from typing import Optional, Iterable, Dict, List
from collections import defaultdict
from itertools import groupby

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, URL
from sqlalchemy.exc import OperationalError

# -------------------------
# HARD-CODED DB URL
# -------------------------
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
    start_date: date
    end_date: date
    src_schema: str = "core"
    src_table: str = "trades"
    rpt_schema: str = "reports"
    db_url: URL = REPORT_DB_URL


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Run reports pipeline (FIFO + MTM + auto-close + P&L).")
    p.add_argument("--from", dest="dfrom", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--to",   dest="dto",   required=True, help="End date YYYY-MM-DD")
    a = p.parse_args()
    return Config(
        start_date=date.fromisoformat(a.dfrom),
        end_date=date.fromisoformat(a.dto),
    )


# ---------------------------------------------------------------------
# QA tracker
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

        # Rebuild/purge QA
        self.purged_fifo_rows_normal = 0
        self.purged_eod_rows_normal = 0
        self.purged_pnl_rows_normal = 0
        self.purged_fifo_rows_error = 0
        self.purged_eod_rows_error = 0
        self.purged_pnl_rows_error = 0

        # Carry-forward inserts
        self.carry_forward_inserts_normal = 0
        self.carry_forward_inserts_error = 0

        # Window row counts (post-run)
        self.window_fifo_rows_normal = 0
        self.window_eod_rows_normal = 0
        self.window_pnl_rows_normal = 0
        self.window_fifo_rows_error = 0
        self.window_eod_rows_error = 0
        self.window_pnl_rows_error = 0

        # End-of-run open positions counts
        self.open_positions_end_normal = 0
        self.open_tickets_end_normal = 0
        self.stuck_open_after_expiry_end_normal = 0

        self.open_positions_end_error = 0
        self.open_tickets_end_error = 0
        self.stuck_open_after_expiry_end_error = 0

        # Unparsable dates (unchanged)
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
# TRADING CALENDAR HELPERS
# ---------------------------------------------------------------------
def get_trading_days(conn: Connection, cfg: Config, d_from: date, d_to: date) -> List[date]:
    rows = run_sql(
        conn,
        """
        SELECT trade_date
        FROM core.trading_calendar
        WHERE trade_date BETWEEN :d_from AND :d_to
          AND is_trading_day = true
        ORDER BY trade_date
        """,
        d_from=d_from,
        d_to=d_to,
    ).fetchall()

    if not rows:
        print("[WARN] trading_calendar empty or missing; using all calendar days.")
        cur = d_from
        out = []
        while cur <= d_to:
            out.append(cur)
            cur = cur.fromordinal(cur.toordinal() + 1)
        return out

    return [r[0] for r in rows]


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


# ---------------------------------------------------------------------
# TABLE NAME MAP
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
# SCHEMA: FIFO WORK TABLES
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
# SELECT TRADES (NORMAL vs ERROR via entry_exit_error)
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
# UPSERT FIFO DAY
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
# PRICE LOOKUPS (core.option_eod_prices_all)
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
# UPSERT EOD POSITIONS
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
# UPSERT DAILY PNL
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
# FIFO ENGINE
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


def process_bucket(conn: Connection, cfg: Config, t: Tables, rows: Iterable[Dict]):
    st = FifoState()
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
# CARRY-FORWARD (OPEN POSITIONS -> fifo_inventory_today)
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
    # rowcount is best-effort (should work for INSERT..SELECT)
    return int(res.rowcount or 0)


# ---------------------------------------------------------------------
# BUILD EOD POS (WITH MTM)  [QA FIX INCLUDED]
# ---------------------------------------------------------------------
def build_eod_positions_for_day(conn: Connection, cfg: Config, t: Tables, mode: str, day: date, qa: QAStats) -> int:
    """
    Returns count of positions where NO mark and NO fallback were available.
    QA fix: if day >= expiry_date and an expiry-close mark exists, use it (do not count missing).
    """
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

        # 1) Primary mark for the day
        eod_price = get_best_mark(conn, cfg, day, instr, otype, strike, expiry)

        # 2) QA FIX: if expiry day or after, try expiry-close mark before counting missing
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

        # 3) Fallback: previous day's implied price if any
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

                    # Contract identifiers
                    "instrument_name": instr,
                    "option_type": otype,
                    "strike": strike,
                    "expiry_date": expiry,

                    # Troubleshoot identifiers
                    "account_id": r["account_id"],
                    "strategy_id": r["strategy_id"],
                    "strategy_variant_id": r["strategy_variant_id"],
                    "user_id": r["user_id"],
                    "position_key_trade": pk,

                    # QA flags
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
# AUTO-CLOSE AFTER / AT EXPIRY
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
# QA: unparsable trade_date (unchanged)
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


def count_unparsable_trade_dates(conn: Connection, cfg: Config, qa: QAStats):
    qa.unparsable_trade_date_total = run_sql(
        conn, sql_count_unparsable_trade_date_total(cfg)
    ).scalar_one()
    qa.unparsable_trade_date_in_window = run_sql(
        conn,
        sql_count_unparsable_trade_date_in_window(cfg),
        d_from_text=cfg.start_date.isoformat(),
        d_to_text=cfg.end_date.isoformat()
    ).scalar_one()


# ---------------------------------------------------------------------
# PURGE WINDOW (delete + rebuild every run)  [NEW]
# ---------------------------------------------------------------------
def purge_reports_window(conn: Connection, cfg: Config, t: Tables, d_from: date, d_to: date) -> Dict[str, int]:
    # delete in dependency order
    res_pnl = run_sql(conn, f'DELETE FROM "{cfg.rpt_schema}"."{t.pnl}" WHERE trade_day BETWEEN :d1 AND :d2', d1=d_from, d2=d_to)
    res_eod = run_sql(conn, f'DELETE FROM "{cfg.rpt_schema}"."{t.eod}" WHERE trade_day BETWEEN :d1 AND :d2', d1=d_from, d2=d_to)
    res_fifo = run_sql(conn, f'DELETE FROM "{cfg.rpt_schema}"."{t.fifo}" WHERE trade_day BETWEEN :d1 AND :d2', d1=d_from, d2=d_to)
    return {
        "pnl": int(res_pnl.rowcount or 0),
        "eod": int(res_eod.rowcount or 0),
        "fifo": int(res_fifo.rowcount or 0),
    }


# ---------------------------------------------------------------------
# EXTRA QA QUERIES (end-of-run)
# ---------------------------------------------------------------------
def count_rows_window(conn: Connection, cfg: Config, table_name: str, d_from: date, d_to: date) -> int:
    return int(run_sql(
        conn,
        f'SELECT count(*)::bigint FROM "{cfg.rpt_schema}"."{table_name}" WHERE trade_day BETWEEN :d1 AND :d2',
        d1=d_from, d2=d_to
    ).scalar_one())


def count_open_positions_end(conn: Connection, cfg: Config, eod_table: str, run_to: date) -> Dict[str, int]:
    # open rows, open distinct tickets, and "stuck after expiry" open count
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
# RUN ONE PIPELINE (NORMAL or ERROR)
# ---------------------------------------------------------------------
def run_one_mode(conn: Connection, cfg: Config, qa: QAStats, mode: str, run_from: date, run_to: date):
    t = tables_for(mode)

    # Ensure tables exist
    run_sql(conn, sql_create_fifo_work(cfg, t))
    if mode == "ERROR":
        run_sql(conn, sql_create_errors_eod_positions(cfg))
        run_sql(conn, sql_create_errors_pnl_daily(cfg))

    # Purge window each run (idempotent rebuild)
    purged = purge_reports_window(conn, cfg, t, run_from, run_to)
    if mode == "ERROR":
        qa.purged_fifo_rows_error += purged["fifo"]
        qa.purged_eod_rows_error += purged["eod"]
        qa.purged_pnl_rows_error += purged["pnl"]
    else:
        qa.purged_fifo_rows_normal += purged["fifo"]
        qa.purged_eod_rows_normal += purged["eod"]
        qa.purged_pnl_rows_normal += purged["pnl"]

    # Build FIFO from trades (only for this mode)
    rows = run_sql(
        conn,
        sql_select_trades(cfg, mode),
        d_from=run_from,
        d_to=run_to
    ).mappings().all()

    if mode == "ERROR":
        qa.trades_rows_fetched_error += len(rows)
        qa.distinct_tickets_error += len(set(r["position_key_trade"] for r in rows))
    else:
        qa.trades_rows_fetched_normal += len(rows)
        qa.distinct_tickets_normal += len(set(r["position_key_trade"] for r in rows))

    for _, group in groupby(rows, key=lambda r: r["position_key_trade"]):
        process_bucket(conn, cfg, t, group)

    # Trading days from calendar for window
    trading_days = get_trading_days(conn, cfg, run_from, run_to)

    # Per-day pipeline
    for cur in trading_days:
        inserted = ensure_fifo_carry_forward(conn, cfg, t, cur)
        if mode == "ERROR":
            qa.carry_forward_inserts_error += inserted
        else:
            qa.carry_forward_inserts_normal += inserted

        missing = build_eod_positions_for_day(conn, cfg, t, mode, cur, qa)
        if mode == "ERROR":
            qa.missing_marks_by_day_error[cur] += missing
            qa.missing_marks_total_error += missing
        else:
            qa.missing_marks_by_day_normal[cur] += missing
            qa.missing_marks_total_normal += missing

        closed = apply_autoclose_after_expiry(conn, cfg, t, cur)
        if mode == "ERROR":
            qa.autoclose_by_day_error[cur] += closed
            qa.autoclose_total_error += closed
        else:
            qa.autoclose_by_day_normal[cur] += closed
            qa.autoclose_total_normal += closed

        upsert_pnl_daily(conn, cfg, t, cur)


# ---------------------------------------------------------------------
# DRIVER
# ---------------------------------------------------------------------
def run_pipeline(cfg: Config):
    qa = QAStats()
    rebuild_from = None

    with get_conn(cfg.db_url) as conn:
        # 1) QA (unchanged)
        count_unparsable_trade_dates(conn, cfg, qa)

        # 2) rebuild_from = prev trading day (for correct carry-forward)
        rebuild_from = get_prev_trading_day(conn, cfg.start_date) or cfg.start_date
        print(f"[REBUILD] Window: {rebuild_from} .. {cfg.end_date}  (requested: {cfg.start_date} .. {cfg.end_date})")

        # 3) NORMAL pipeline
        run_one_mode(conn, cfg, qa, mode="NORMAL", run_from=rebuild_from, run_to=cfg.end_date)

        # 4) ERROR pipeline
        run_one_mode(conn, cfg, qa, mode="ERROR", run_from=rebuild_from, run_to=cfg.end_date)

        # 5) Post-run window counts + end-of-run open positions (QA only)
        tn = tables_for("NORMAL")
        te = tables_for("ERROR")

        qa.window_fifo_rows_normal = count_rows_window(conn, cfg, tn.fifo, rebuild_from, cfg.end_date)
        qa.window_eod_rows_normal = count_rows_window(conn, cfg, tn.eod, rebuild_from, cfg.end_date)
        qa.window_pnl_rows_normal = count_rows_window(conn, cfg, tn.pnl, rebuild_from, cfg.end_date)

        qa.window_fifo_rows_error = count_rows_window(conn, cfg, te.fifo, rebuild_from, cfg.end_date)
        qa.window_eod_rows_error = count_rows_window(conn, cfg, te.eod, rebuild_from, cfg.end_date)
        qa.window_pnl_rows_error = count_rows_window(conn, cfg, te.pnl, rebuild_from, cfg.end_date)

        end_n = count_open_positions_end(conn, cfg, tn.eod, cfg.end_date)
        qa.open_positions_end_normal = end_n["open_rows"]
        qa.open_tickets_end_normal = end_n["open_tickets"]
        qa.stuck_open_after_expiry_end_normal = end_n["stuck_after_expiry"]

        end_e = count_open_positions_end(conn, cfg, te.eod, cfg.end_date)
        qa.open_positions_end_error = end_e["open_rows"]
        qa.open_tickets_end_error = end_e["open_tickets"]
        qa.stuck_open_after_expiry_end_error = end_e["stuck_after_expiry"]

    # -------------------------
    # QA summary (clean + actionable)
    # -------------------------
    print("\n====== QA SUMMARY ======")
    print(f"Requested date range         : {cfg.start_date} .. {cfg.end_date}")
    print(f"Rebuild window (incl carry)  : {rebuild_from} .. {cfg.end_date}")
    print(f"Unparsable trade_date rows   : total={qa.unparsable_trade_date_total} | in_window={qa.unparsable_trade_date_in_window}")

    print("\n--- NORMAL (ENTRY/EXIT + NULL treated as normal) ---")
    print(f"Trades fetched               : {qa.trades_rows_fetched_normal}")
    print(f"Distinct tickets (keys)      : {qa.distinct_tickets_normal}")
    print(f"Purged rows (fifo/eod/pnl)   : {qa.purged_fifo_rows_normal} / {qa.purged_eod_rows_normal} / {qa.purged_pnl_rows_normal}")
    print(f"Window rows (fifo/eod/pnl)   : {qa.window_fifo_rows_normal} / {qa.window_eod_rows_normal} / {qa.window_pnl_rows_normal}")
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
    print(f"Window rows (fifo/eod/pnl)   : {qa.window_fifo_rows_error} / {qa.window_eod_rows_error} / {qa.window_pnl_rows_error}")
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

    print("\n[OK] Pipeline completed.")


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
