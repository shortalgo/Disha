CREATE OR REPLACE VIEW reports.v_trades_detail_named
AS SELECT trade_date,
    trade_time,
    user_name,
    strategy_name,
    strategy_variant_name,
    tradetron_name,
    instrument_name,
    option_type,
    strike,
    expiry_date,
    trade_number,
    qty,
    trade_price
   FROM core.v_trades_named t
  ORDER BY trade_date, trade_time, trade_number;