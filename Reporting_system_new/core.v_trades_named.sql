-- core.v_trades_named source

CREATE OR REPLACE VIEW core.v_trades_named
AS SELECT t.trade_date,
    t.trade_time,
    u.name AS user_name,
    s.strategy_name,
    sv.variant_name AS strategy_variant_name,
    sv.tradetron_name,
    t.account_id,
    t.strategy_id,
    t.strategy_variant_id,
    t.user_id,
    t.instrument_name,
    upper(t.type::text) AS option_type,
    t.strike::numeric AS strike,
    t.expiry_date,
    t.trade_number::bigint AS trade_number,
    t.qty::numeric AS qty,
    t.trade_price::numeric AS trade_price
   FROM core.trades t
     LEFT JOIN core.users u ON u.id_no = t.user_id
     LEFT JOIN core.strategies s ON s.id = t.strategy_id
     LEFT JOIN core.strategy_variants sv ON sv.id = t.strategy_variant_id;