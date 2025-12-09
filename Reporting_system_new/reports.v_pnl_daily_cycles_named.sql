-- reports.v_pnl_daily_cycles_named source

CREATE OR REPLACE VIEW reports.v_pnl_daily_cycles_named
AS WITH base AS (
         SELECT p.trade_day,
            p.account_id,
            p.strategy_id,
            p.strategy_variant_id,
            p.user_id,
            p.realized_cash_today,
            p.unrealized_change,
            p.total_pnl_day
           FROM reports.pnl_daily p
        ), joined AS (
         SELECT b.trade_day,
            b.account_id,
            b.strategy_id,
            b.strategy_variant_id,
            b.user_id,
            b.realized_cash_today,
            b.unrealized_change,
            b.total_pnl_day,
            sv.instrument,
            c.month_label,
            c.cycle_start,
            c.cycle_end,
            s.strategy_name,
            sv.variant_name AS strategy_variant_name,
            u.name AS user_name
           FROM base b
             LEFT JOIN core.strategy_variants sv ON sv.id = b.strategy_variant_id
             LEFT JOIN core.strategies s ON s.id = b.strategy_id
             LEFT JOIN core.users u ON u.id_no = b.user_id
             LEFT JOIN reports.index_month_cycles c ON c.instrument = sv.instrument AND b.trade_day >= c.cycle_start AND b.trade_day <= c.cycle_end
        )
 SELECT trade_day,
    instrument,
    month_label,
    cycle_start,
    cycle_end,
    account_id,
    strategy_id,
    strategy_variant_id,
    user_id,
    strategy_name,
    strategy_variant_name,
    user_name,
    realized_cash_today,
    unrealized_change,
    total_pnl_day,
    sum(total_pnl_day) OVER (PARTITION BY instrument, month_label, account_id, strategy_id, strategy_variant_id, user_id ORDER BY trade_day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_monthly_mtm
   FROM joined;