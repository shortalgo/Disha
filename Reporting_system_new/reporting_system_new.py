#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reporting_system_new.py

Source: core.trades
Targets (schema: reports):
  - fifo_inventory_today  : ticket-scoped FIFO state per trade_day
  - fetched_eod_prices    : EOD marks per (date, instrument, type, strike, expiry)
  - eod_positions         : ticket-scoped EOD snapshot, PK (trade_day, position_key_trade)
  - pnl_daily             : daily P&L aggregated from eod_positions

Implements:
- Ticket-scoped FIFO (partition includes trade_number via position_key_trade).
- EOD marking using fetched_eod_prices (latest retrieved_at).
- Auto-close after expiry at expiry-day mark.
- Daily aggregation into reports.pnl_daily.
- No fees.
- QA summary of missing marks and auto-closes.
- Enqueue of open positions into reports.fetched_eod_prices as NULL-price rows.
"""

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional, Iterable, Dict
from collections import defaultdict

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
    p = argparse.ArgumentParser(description="Run reports pipeline (FIFO + EOD + auto-close + P&L).")
    p.add_argument("--from", dest="dfrom", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--to",   dest="dto",   required=True, help="End date YYYY-MM-DD")
    a = p.parse_args()
    return Config(
        start_date=date.fromisoformat(a.dfrom),
        end_date=date.fromisoformat(a.dto),
    )


# -------------------------
# QA tracker
# -------------------------
class QAStats:
    def __init__(self):
        self.missing_marks_by_day = defaultdict(int)
        self.autoclose_by_day = defaultdict(int)
        self.missing_marks_total = 0
        self.autoclose_total = 0
        self.unparsable_trade_date_total = 0
        self.unparsable_trade_date_in_window = 0


# -------------------------
# DB helpers
# -------------------------
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


# -------------------------
# SQL helpers
# -------------------------
def sql_create_fifo_work(cfg: Config) -> str:
    return f"""
create table if not exists "{cfg.rpt_schema}"."fifo_inventory_today" (
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


def sql_select_trades(cfg: Config) -> str:
    return f"""
select
  t.account_id,
  t.strategy_id,
  t.strategy_variant_id,
  t.user_id,
  t.instrument_name,
  upper((t.type)::text)                    as option_type,
  (t.strike)::numeric                      as strike,
  (t.expiry_date)::date                    as expiry_date,
  (t.trade_number)::bigint                 as trade_number,
  t.trade_time                             as trade_ts,
  to_date((t.trade_date)::text,'YYYY-MM-DD') as trade_day,
  (t.qty)::numeric                         as qty,
  (t.trade_price)::numeric                 as trade_price,
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
order by
  position_key_trade,
  to_date((t.trade_date)::text,'YYYY-MM-DD'),
  t.trade_time,
  t.trade_number;
"""


def sql_upsert_fifo_day(cfg: Config) -> str:
    return f"""
insert into "{cfg.rpt_schema}"."fifo_inventory_today" as f(
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


def sql_best_mark(cfg: Config) -> str:
    return f"""
select price
from "{cfg.rpt_schema}"."fetched_eod_prices"
where date = :day
  and instrument_name = :instr
  and option_type = :otype
  and strike = :strike
  and expiry_date = :expiry
order by retrieved_at desc
limit 1;
"""


def sql_best_expiry_close(cfg: Config) -> str:
    return f"""
select price
from "{cfg.rpt_schema}"."fetched_eod_prices"
where instrument_name = :instr
  and option_type = :otype
  and strike = :strike
  and expiry_date = :expiry
  and date = :expiry
order by retrieved_at desc
limit 1;
"""


def sql_best_expiry_fallback(cfg: Config) -> str:
    return f"""
select price
from "{cfg.rpt_schema}"."fetched_eod_prices"
where instrument_name = :instr
  and option_type = :otype
  and strike = :strike
  and expiry_date = :expiry
  and date <= :expiry
order by date desc, retrieved_at desc
limit 1;
"""


def sql_upsert_eod_pos(cfg: Config) -> str:
    # Ticket-scoped EOD, PK (trade_day, position_key_trade)
    return f"""
insert into "{cfg.rpt_schema}"."eod_positions" as t(
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
  prev_mv_eod
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
  :prev_mv_eod
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
  prev_mv_eod             = excluded.prev_mv_eod;
"""


def sql_upsert_pnl_daily(cfg: Config) -> str:
    return f"""
insert into "{cfg.rpt_schema}"."pnl_daily" as d(
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
  sum(realized_pnl_fifo_today) as realized_cash_today,
  sum(coalesce(mv_eod,0) - coalesce(prev_mv_eod,0)) as unrealized_change,
  sum(realized_pnl_fifo_today + (coalesce(mv_eod,0) - coalesce(prev_mv_eod,0))) as total_pnl_day,
  null::numeric
from "{cfg.rpt_schema}"."eod_positions"
where trade_day = :day
group by 2,3,4,5
on conflict (trade_day, account_id, strategy_id, strategy_variant_id, user_id)
do update set
  realized_cash_today = excluded.realized_cash_today,
  unrealized_change   = excluded.unrealized_change,
  total_pnl_day       = excluded.total_pnl_day;
"""


# ---- QA SQL helpers ----
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
    qa.unparsable_trade_date_total = run_sql(conn, sql_count_unparsable_trade_date_total(cfg)).scalar_one()
    qa.unparsable_trade_date_in_window = run_sql(
        conn,
        sql_count_unparsable_trade_date_in_window(cfg),
        d_from_text=cfg.start_date.isoformat(),
        d_to_text=cfg.end_date.isoformat()
    ).scalar_one()


# -------------------------
# FIFO engine (ticket-scoped)
# -------------------------
class FifoState:
    __slots__ = ("q_running", "cost_total", "avg_cost", "realized_today", "last_day")

    def __init__(self):
        self.q_running: float = 0.0
        self.cost_total: float = 0.0
        self.avg_cost: Optional[float] = None
        self.realized_today: float = 0.0
        self.last_day: Optional[date] = None


def sql_upsert_fifo_day_call(conn: Connection, cfg: Config, r: Dict, st: FifoState):
    run_sql(conn, sql_upsert_fifo_day(cfg),
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


def process_bucket(conn: Connection, cfg: Config, rows: Iterable[Dict]):
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

        sql_upsert_fifo_day_call(conn, cfg, r, st)


# -------------------------
# ENQUEUE step: open positions needing marks
# -------------------------
def sql_enqueue_eod_marks(cfg: Config) -> str:
    return f"""
insert into "{cfg.rpt_schema}"."fetched_eod_prices" (
    date,
    instrument_name,
    option_type,
    strike,
    expiry_date,
    price,
    retrieved_at
)
select distinct
    cast(:day as date)      as date,
    f.instrument_name::text,
    f.option_type::text,
    f.strike::numeric,
    f.expiry_date::date,
    null::numeric           as price,
    now()                   as retrieved_at
from "{cfg.rpt_schema}"."fifo_inventory_today" f
where f.trade_day = cast(:day as date)
  and f.eod_net_qty <> 0
  and not exists (
      select 1
      from "{cfg.rpt_schema}"."fetched_eod_prices" p
      where p.date            = cast(:day as date)
        and p.instrument_name = f.instrument_name
        and p.option_type     = f.option_type
        and p.strike          = f.strike
        and p.expiry_date     = f.expiry_date
  );
"""


def enqueue_eod_marks(conn: Connection, cfg: Config, day: date) -> int:
    res = run_sql(conn, sql_enqueue_eod_marks(cfg), day=day)
    return res.rowcount or 0


# -------------------------
# EOD + auto-close + P&L
# -------------------------
def get_best_mark(conn: Connection, cfg: Config, day: date,
                  instr: str, otype: str, strike: float, expiry) -> Optional[float]:
    row = run_sql(conn, sql_best_mark(cfg),
                  day=day, instr=instr, otype=otype, strike=strike, expiry=expiry).fetchone()
    return None if row is None else row[0]


def get_best_expiry_close(conn: Connection, cfg: Config,
                          instr: str, otype: str, strike: float, expiry) -> Optional[float]:
    row = run_sql(conn, sql_best_expiry_close(cfg),
                  instr=instr, otype=otype, strike=strike, expiry=expiry).fetchone()
    if row and row[0] is not None:
        return row[0]
    row = run_sql(conn, sql_best_expiry_fallback(cfg),
                  instr=instr, otype=otype, strike=strike, expiry=expiry).fetchone()
    return None if row is None else row[0]


def build_eod_positions_for_day(conn: Connection, cfg: Config, day: date) -> int:
    """Per-ticket EOD snapshot. Returns count of rows missing marks."""
    missing_marks = 0

    fifo_rows = run_sql(conn, f"""
        select *
        from "{cfg.rpt_schema}"."fifo_inventory_today"
        where trade_day = :day
        order by position_key_trade
    """, day=day).mappings().all()

    for r in fifo_rows:
        pk = r["position_key_trade"]

        eod_price = get_best_mark(
            conn, cfg,
            day,
            r["instrument_name"],
            r["option_type"],
            float(r["strike"]),
            r["expiry_date"]
        )
        if eod_price is None:
            missing_marks += 1

        mv_eod = float(eod_price or 0.0) * float(r["eod_net_qty"])

        prev = run_sql(conn, f"""
            select eod_net_qty, mv_eod
            from "{cfg.rpt_schema}"."eod_positions"
            where position_key_trade = :pk
              and trade_day = (
                select max(trade_day)
                from "{cfg.rpt_schema}"."eod_positions"
                where position_key_trade = :pk
                  and trade_day < :day
              )
        """, pk=pk, day=day).fetchone()

        prev_qty = prev[0] if prev else None
        prev_mv  = prev[1] if prev else None
        carry_in = (prev_qty is not None and float(prev_qty) != 0.0)

        run_sql(conn, sql_upsert_eod_pos(cfg),
            trade_day          = day,
            position_key_trade = pk,
            account_id         = r["account_id"],
            strategy_id        = r["strategy_id"],
            strategy_variant_id= r["strategy_variant_id"],
            user_id            = r["user_id"],
            instrument_name    = r["instrument_name"],
            option_type        = r["option_type"],
            strike             = r["strike"],
            expiry_date        = r["expiry_date"],
            eod_net_qty        = r["eod_net_qty"],
            avg_cost_fifo      = r["avg_cost_fifo"],
            eod_price          = eod_price,
            mv_eod             = mv_eod,
            realized_pnl_fifo_today = r["realized_pnl_fifo_today"],
            carry_in           = carry_in,
            prev_eod_qty       = prev_qty,
            prev_mv_eod        = prev_mv
        )

    return missing_marks


def apply_autoclose_after_expiry(conn: Connection, cfg: Config, day: date) -> int:
    """Auto-close tickets still open after expiry using expiry mark."""
    rows = run_sql(conn, f"""
        select
          position_key_trade,
          instrument_name,
          option_type,
          strike,
          expiry_date,
          eod_net_qty,
          avg_cost_fifo
        from "{cfg.rpt_schema}"."eod_positions"
        where trade_day = :day
          and eod_net_qty <> 0
          and trade_day > expiry_date
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
        ac  = float(r["avg_cost_fifo"] or 0.0)

        if qty > 0:
            realized_bump = (expiry_px - ac) * qty
        else:
            realized_bump = (ac - expiry_px) * abs(qty)

        run_sql(conn, f"""
            update "{cfg.rpt_schema}"."eod_positions"
               set realized_pnl_fifo_today = coalesce(realized_pnl_fifo_today,0) + :add_real,
                   eod_price               = :px,
                   mv_eod                  = 0,
                   eod_net_qty             = 0,
                   avg_cost_fifo           = null
             where trade_day = :day
               and position_key_trade = :pk
        """, add_real=realized_bump, px=expiry_px, day=day, pk=r["position_key_trade"])

        applied += 1

    return applied


def upsert_pnl_daily(conn: Connection, cfg: Config, day: date):
    run_sql(conn, sql_upsert_pnl_daily(cfg), day=day)


# -------------------------
# Driver
# -------------------------
def run_pipeline(cfg: Config):
    qa = QAStats()

    with get_conn(cfg.db_url) as conn:
        # 0) Working FIFO table
        run_sql(conn, sql_create_fifo_work(cfg))

        # 1) QA: unparsable dates
        count_unparsable_trade_dates(conn, cfg, qa)

        # 2) FIFO per ticket
        rows = run_sql(conn, sql_select_trades(cfg),
                       d_from=cfg.start_date,
                       d_to=cfg.end_date).mappings().all()

        from itertools import groupby
        keyfunc = lambda r: r["position_key_trade"]
        for _, group in groupby(rows, key=keyfunc):
            process_bucket(conn, cfg, group)

        # 3) Per-day: enqueue marks → build EOD → autoclose → pnl_daily
        cur = cfg.start_date
        while cur <= cfg.end_date:
            enqueue_eod_marks(conn, cfg, cur)

            missing = build_eod_positions_for_day(conn, cfg, cur)
            qa.missing_marks_by_day[cur] += missing
            qa.missing_marks_total += missing

            closed = apply_autoclose_after_expiry(conn, cfg, cur)
            qa.autoclose_by_day[cur] += closed
            qa.autoclose_total += closed

            upsert_pnl_daily(conn, cfg, cur)
            cur += timedelta(days=1)

    # --------- QA SUMMARY ----------
    print("\n====== QA SUMMARY ======")
    print(f"Date range                 : {cfg.start_date} .. {cfg.end_date}")
    print(f"Unparsable trade_date rows : total={qa.unparsable_trade_date_total} | in_window={qa.unparsable_trade_date_in_window}")
    print(f"Missing EOD marks          : total={qa.missing_marks_total}")
    print(f"Auto-closes applied        : total={qa.autoclose_total}")

    if qa.missing_marks_total > 0:
        top_mm = sorted(qa.missing_marks_by_day.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("\nTop days with missing EOD marks (day → count):")
        for d, c in top_mm:
            print(f"  {d} → {c}")

    if qa.autoclose_total > 0:
        top_ac = sorted(qa.autoclose_by_day.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("\nTop days with auto-closes applied (day → count):")
        for d, c in top_ac:
            print(f"  {d} → {c}")

    print("\n[OK] Pipeline completed.")


if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
