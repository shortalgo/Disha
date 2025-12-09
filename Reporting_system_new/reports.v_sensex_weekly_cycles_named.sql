-- reports.v_sensex_weekly_cycles_named source

CREATE OR REPLACE VIEW reports.v_sensex_weekly_cycles_named
AS WITH daily AS (
         SELECT v_pnl_daily_named.trade_day,
            v_pnl_daily_named.account_id,
            v_pnl_daily_named.strategy_id,
            v_pnl_daily_named.strategy_name,
            v_pnl_daily_named.strategy_variant_id,
            v_pnl_daily_named.strategy_variant_name,
            v_pnl_daily_named.user_id,
            v_pnl_daily_named.user_name,
            v_pnl_daily_named.instrument,
            v_pnl_daily_named.realized_cash_today,
            v_pnl_daily_named.unrealized_change,
            v_pnl_daily_named.total_pnl_day
           FROM reports.v_pnl_daily_named
          WHERE v_pnl_daily_named.instrument = 'SENSEX'::text
        ), wk AS (
         SELECT index_week_cycles.instrument,
            index_week_cycles.month_label,
            index_week_cycles.week_start,
            index_week_cycles.week_end,
            (to_char(index_week_cycles.week_start::timestamp with time zone, 'YYYY-MM-DD'::text) || ' → '::text) || to_char(index_week_cycles.week_end::timestamp with time zone, 'YYYY-MM-DD'::text) AS week_label
           FROM reports.index_week_cycles
          WHERE index_week_cycles.instrument = 'SENSEX'::text
        ), mn AS (
         SELECT index_month_cycles.instrument,
            index_month_cycles.month_label,
            index_month_cycles.cycle_start,
            index_month_cycles.cycle_end
           FROM reports.index_month_cycles
          WHERE index_month_cycles.instrument = 'SENSEX'::text
        ), daily_in_week AS (
         SELECT d.trade_day,
            d.account_id,
            d.strategy_id,
            d.strategy_name,
            d.strategy_variant_id,
            d.strategy_variant_name,
            d.user_id,
            d.user_name,
            d.instrument,
            d.realized_cash_today,
            d.unrealized_change,
            d.total_pnl_day,
            w.month_label,
            w.week_start,
            w.week_end,
            w.week_label,
            m.cycle_start,
            m.cycle_end
           FROM daily d
             JOIN wk w ON d.instrument = w.instrument AND d.trade_day >= w.week_start AND d.trade_day <= w.week_end
             JOIN mn m ON m.instrument = w.instrument AND m.month_label = w.month_label
        ), cum_before_week AS (
         SELECT w.instrument,
            w.month_label,
            w.week_start,
            d.account_id,
            d.strategy_id,
            d.strategy_variant_id,
            d.user_id,
            COALESCE(sum(d.total_pnl_day), 0::numeric) AS cum_monthly_mtm
           FROM wk w
             JOIN mn m ON m.instrument = w.instrument AND m.month_label = w.month_label
             JOIN daily d ON d.instrument = w.instrument AND d.trade_day >= m.cycle_start AND d.trade_day <= m.cycle_end AND d.trade_day < w.week_start
          GROUP BY w.instrument, w.month_label, w.week_start, d.account_id, d.strategy_id, d.strategy_variant_id, d.user_id
        )
 SELECT j.instrument,
    j.month_label,
    j.week_start,
    j.week_end,
    j.week_label,
    j.account_id,
    j.strategy_id,
    j.strategy_name,
    j.strategy_variant_id,
    j.strategy_variant_name,
    j.user_id,
    j.user_name,
    COALESCE(c.cum_monthly_mtm, 0::numeric) AS cum_monthly_mtm,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 5::numeric THEN j.realized_cash_today
            ELSE 0::numeric
        END) AS fri_realized,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 5::numeric THEN j.unrealized_change
            ELSE 0::numeric
        END) AS fri_unrealized_change,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 5::numeric THEN j.total_pnl_day
            ELSE 0::numeric
        END) AS fri_total,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 1::numeric THEN j.realized_cash_today
            ELSE 0::numeric
        END) AS mon_realized,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 1::numeric THEN j.unrealized_change
            ELSE 0::numeric
        END) AS mon_unrealized_change,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 1::numeric THEN j.total_pnl_day
            ELSE 0::numeric
        END) AS mon_total,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 2::numeric THEN j.realized_cash_today
            ELSE 0::numeric
        END) AS tue_realized,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 2::numeric THEN j.unrealized_change
            ELSE 0::numeric
        END) AS tue_unrealized_change,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 2::numeric THEN j.total_pnl_day
            ELSE 0::numeric
        END) AS tue_total,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 3::numeric THEN j.realized_cash_today
            ELSE 0::numeric
        END) AS wed_realized,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 3::numeric THEN j.unrealized_change
            ELSE 0::numeric
        END) AS wed_unrealized_change,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 3::numeric THEN j.total_pnl_day
            ELSE 0::numeric
        END) AS wed_total,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 4::numeric THEN j.realized_cash_today
            ELSE 0::numeric
        END) AS thu_realized,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 4::numeric THEN j.unrealized_change
            ELSE 0::numeric
        END) AS thu_unrealized_change,
    sum(
        CASE
            WHEN EXTRACT(dow FROM j.trade_day) = 4::numeric THEN j.total_pnl_day
            ELSE 0::numeric
        END) AS thu_total
   FROM daily_in_week j
     LEFT JOIN cum_before_week c ON c.instrument = j.instrument AND c.month_label = j.month_label AND c.week_start = j.week_start AND c.account_id = j.account_id AND c.strategy_id = j.strategy_id AND c.strategy_variant_id = j.strategy_variant_id AND c.user_id = j.user_id
  GROUP BY j.instrument, j.month_label, j.week_start, j.week_end, j.week_label, j.account_id, j.strategy_id, j.strategy_name, j.strategy_variant_id, j.strategy_variant_name, j.user_id, j.user_name, c.cum_monthly_mtm
  ORDER BY j.month_label, j.week_start, j.strategy_name, j.strategy_variant_name, j.user_name, j.account_id;