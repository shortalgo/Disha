#!/usr/bin/env python3
import argparse
from datetime import date
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# DB_URL = "postgresql+psycopg2://postgres:New@121@localhost:5432/postgres"  # test DB
#test db
DB_URL = URL.create(
    drivername="postgresql+psycopg2",
    username="postgres",
    password="New@121",      # adjust for test DB
    host="localhost",
    port=5432,
    database="postgres",
)

#main
# DB_URL = URL.create(
#     drivername="postgresql+psycopg2",
#     username="postgres",
#     password="New@1234",
#     host="192.168.18.23",
#     port=5432,
#     database="postgres",
# )


TIME_BUFFER_MIN = 4

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="dfrom", required=True)
    p.add_argument("--to", dest="dto", required=True)
    return date.fromisoformat(p.parse_args().dfrom), date.fromisoformat(p.parse_args().dto)

SLIP_SQL = f"""
-- rebuild window (simple + safe)
DELETE FROM reports.bot_slippage
WHERE entry_trade_date BETWEEN :d1 AND :d2;

WITH bot AS (
  SELECT *
  FROM core.bot_trades b
  WHERE b.entry_trade_date BETWEEN :d1 AND :d2
),
entry_match AS (
  SELECT
    b.*,
    te.unique_id  AS trader_entry_uid,
    te.trade_price::numeric AS trader_entry_price,
    te.qty::numeric         AS trader_entry_qty,
    te.trade_time           AS trader_entry_time,
    abs(extract(epoch from (te.trade_time - b.entry_trade_time))) AS entry_time_diff_seconds
  FROM bot b
  LEFT JOIN LATERAL (
    SELECT t.*
    FROM core.trades t
    WHERE upper(t.entry_exit_error::text) = 'ENTRY'
      AND t.strategy_variant_id = b.strategy_variant_id
      AND t.user_id = b.user_id
      AND t.instrument_name = b.instrument_name
      AND t.strike::numeric = b.strike::numeric
      AND upper(t.type::text) = upper(b.type)
      AND abs(t.qty::numeric) = abs(b.qty::numeric)
      AND (
        CASE
          WHEN t.trade_date::text ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$' THEN to_date(t.trade_date::text,'YYYY-MM-DD')
          WHEN t.trade_date::text ~ '^\\d{{2}}-\\d{{2}}-\\d{{4}}$' THEN to_date(t.trade_date::text,'DD-MM-YYYY')
          ELSE NULL
        END
      ) = b.entry_trade_date
      AND t.trade_time BETWEEN (b.entry_trade_time - interval '{TIME_BUFFER_MIN} minutes')
                         AND (b.entry_trade_time + interval '{TIME_BUFFER_MIN} minutes')
    ORDER BY abs(extract(epoch from (t.trade_time - b.entry_trade_time))) ASC
    LIMIT 1
  ) te ON true
),
full_match AS (
  SELECT
    e.*,
    tx.unique_id  AS trader_exit_uid,
    tx.trade_price::numeric AS trader_exit_price,
    tx.qty::numeric         AS trader_exit_qty,
    tx.trade_time           AS trader_exit_time,
    abs(extract(epoch from (tx.trade_time - e.exit_trade_time))) AS exit_time_diff_seconds
  FROM entry_match e
  LEFT JOIN LATERAL (
    SELECT t.*
    FROM core.trades t
    WHERE upper(t.entry_exit_error::text) = 'EXIT'
      AND t.strategy_variant_id = e.strategy_variant_id
      AND t.user_id = e.user_id
      AND t.instrument_name = e.instrument_name
      AND t.strike::numeric = e.strike::numeric
      AND upper(t.type::text) = upper(e.type)
      AND abs(t.qty::numeric) = abs(e.qty::numeric)
      AND (
        CASE
          WHEN t.trade_date::text ~ '^\\d{{4}}-\\d{{2}}-\\d{{2}}$' THEN to_date(t.trade_date::text,'YYYY-MM-DD')
          WHEN t.trade_date::text ~ '^\\d{{2}}-\\d{{2}}-\\d{{4}}$' THEN to_date(t.trade_date::text,'DD-MM-YYYY')
          ELSE NULL
        END
      ) = e.exit_trade_date
      AND t.trade_time BETWEEN (e.exit_trade_time - interval '{TIME_BUFFER_MIN} minutes')
                         AND (e.exit_trade_time + interval '{TIME_BUFFER_MIN} minutes')
    ORDER BY abs(extract(epoch from (t.trade_time - e.exit_trade_time))) ASC
    LIMIT 1
  ) tx ON true
)
INSERT INTO reports.bot_slippage (
  bot_id, strategy_variant_id, user_id, instrument_name, strike, type, qty,
  entry_trade_date, exit_trade_date,
  bot_entry, bot_exit,
  trader_entry_uid, trader_exit_uid,
  trader_entry_price, trader_exit_price,
  entry_time_diff_seconds, exit_time_diff_seconds,
  entry_slip_points, exit_slip_points, total_slip_points, total_slip_value
)
SELECT
  id AS bot_id,
  strategy_variant_id, user_id, instrument_name, strike, type, qty,
  entry_trade_date, exit_trade_date,
  entry_premium AS bot_entry,
  exit_ltp      AS bot_exit,
  trader_entry_uid, trader_exit_uid,
  trader_entry_price, trader_exit_price,
  entry_time_diff_seconds, exit_time_diff_seconds,

  -- ENTRY: if trader entry qty < 0 => SELL; else BUY
  CASE
    WHEN trader_entry_qty < 0 THEN (trader_entry_price - entry_premium)
    WHEN trader_entry_qty > 0 THEN (entry_premium - trader_entry_price)
    ELSE NULL
  END AS entry_slip_points,

  -- EXIT: derived from entry side
  CASE
    WHEN trader_entry_qty < 0 THEN (exit_ltp - trader_exit_price)     -- short entry => buy exit
    WHEN trader_entry_qty > 0 THEN (trader_exit_price - exit_ltp)     -- long entry  => sell exit
    ELSE NULL
  END AS exit_slip_points,

  (
    CASE
      WHEN trader_entry_qty < 0 THEN (trader_entry_price - entry_premium) + (exit_ltp - trader_exit_price)
      WHEN trader_entry_qty > 0 THEN (entry_premium - trader_entry_price) + (trader_exit_price - exit_ltp)
      ELSE NULL
    END
  ) AS total_slip_points,

  (
    abs(qty::numeric) *
    CASE
      WHEN trader_entry_qty < 0 THEN (trader_entry_price - entry_premium) + (exit_ltp - trader_exit_price)
      WHEN trader_entry_qty > 0 THEN (entry_premium - trader_entry_price) + (trader_exit_price - exit_ltp)
      ELSE 0
    END
  ) AS total_slip_value
FROM full_match;
"""

def main():
    d1, d2 = parse_args()
    eng = create_engine(DB_URL, future=True)
    with eng.begin() as conn:
        conn.execute(text(SLIP_SQL), {"d1": d1, "d2": d2})
    print(f"Slippage rebuilt for {d1} .. {d2} into reports.bot_slippage")

if __name__ == "__main__":
    main()
