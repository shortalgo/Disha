-- core.trades_with_eod source

CREATE OR REPLACE VIEW core.trades_with_eod
AS SELECT t.unique_id,
    t.upload_date,
    t.modified_date,
    t.instrument_name,
    t.trade_number,
    t.trade_date,
    t.trade_time,
    t.trade_price,
    t.strike,
    t.type,
    t.expiry_date,
    t.entry_exit_error,
    t.qty,
    t.reason_exit,
    t.account_id,
    t.strategy_id,
    t.user_id,
    t.theoretical_price,
    t.theoretical_time,
    t.eod_price_id,
    COALESCE(t.eod_price_id, ep.id) AS resolved_eod_price_id,
    ep.price AS resolved_eod_price
   FROM core.trades t
     LEFT JOIN core.eod_prices ep ON ep.date = t.trade_date AND ep.expiry_date = t.expiry_date AND ep.option_type = t.type AND ep.strike = t.strike;