CREATE OR REPLACE VIEW reports.v_sensex_weekly_cycles_named AS
WITH daily AS (
    SELECT
        trade_day,
        account_id,
        strategy_id,
        strategy_name,
        strategy_variant_id,
        strategy_variant_name,
        user_id,
        user_name,
        instrument,
        realized_cash_today,
        unrealized_change,
        total_pnl_day
    FROM reports.v_pnl_daily_cycles_named
    WHERE instrument = 'SENSEX'
),
wk AS (
    SELECT
        instrument,
        month_label,
        week_start,
        week_end,
        (to_char(week_start::timestamptz, 'YYYY-MM-DD') || ' → ' ||
         to_char(week_end::timestamptz, 'YYYY-MM-DD')) AS week_label
    FROM reports.index_week_cycles
    WHERE instrument = 'SENSEX'
),
mn AS (
    SELECT
        instrument,
        month_label,
        cycle_start,
        cycle_end
    FROM reports.index_month_cycles
    WHERE instrument = 'SENSEX'
),
daily_in_week AS (
    SELECT
        d.*,
        w.month_label,
        w.week_start,
        w.week_end,
        w.week_label,
        m.cycle_start,
        m.cycle_end
    FROM daily d
    JOIN wk w
      ON d.instrument = w.instrument
     AND d.trade_day BETWEEN w.week_start AND w.week_end
    JOIN mn m
      ON m.instrument = w.instrument
     AND m.month_label = w.month_label
),
cum_before_week AS (
    -- Month-to-date MTM up to the day BEFORE this SENSEX week starts
    SELECT
        w.instrument,
        w.month_label,
        w.week_start,
        d.account_id,
        d.strategy_id,
        d.strategy_variant_id,
        d.user_id,
        COALESCE(SUM(d.total_pnl_day), 0::numeric) AS cum_monthly_mtm
    FROM wk w
    JOIN mn m
      ON m.instrument = w.instrument
     AND m.month_label = w.month_label
    JOIN daily d
      ON d.instrument = w.instrument
     AND d.trade_day BETWEEN m.cycle_start AND m.cycle_end
     AND d.trade_day < w.week_start
    GROUP BY
        w.instrument, w.month_label, w.week_start,
        d.account_id, d.strategy_id, d.strategy_variant_id, d.user_id
)
SELECT
    j.instrument,
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

    -- Friday (dow=5)
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 5 THEN j.realized_cash_today ELSE 0::numeric END) AS fri_realized,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 5 THEN j.unrealized_change   ELSE 0::numeric END) AS fri_unrealized_change,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 5 THEN j.total_pnl_day       ELSE 0::numeric END) AS fri_total,

    -- Monday (dow=1)
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 1 THEN j.realized_cash_today ELSE 0::numeric END) AS mon_realized,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 1 THEN j.unrealized_change   ELSE 0::numeric END) AS mon_unrealized_change,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 1 THEN j.total_pnl_day       ELSE 0::numeric END) AS mon_total,

    -- Tuesday (dow=2)
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 2 THEN j.realized_cash_today ELSE 0::numeric END) AS tue_realized,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 2 THEN j.unrealized_change   ELSE 0::numeric END) AS tue_unrealized_change,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 2 THEN j.total_pnl_day       ELSE 0::numeric END) AS tue_total,

    -- Wednesday (dow=3)
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 3 THEN j.realized_cash_today ELSE 0::numeric END) AS wed_realized,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 3 THEN j.unrealized_change   ELSE 0::numeric END) AS wed_unrealized_change,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 3 THEN j.total_pnl_day       ELSE 0::numeric END) AS wed_total,

    -- Thursday (dow=4)
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 4 THEN j.realized_cash_today ELSE 0::numeric END) AS thu_realized,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 4 THEN j.unrealized_change   ELSE 0::numeric END) AS thu_unrealized_change,
    SUM(CASE WHEN EXTRACT(DOW FROM j.trade_day) = 4 THEN j.total_pnl_day       ELSE 0::numeric END) AS thu_total

FROM daily_in_week j
LEFT JOIN cum_before_week c
  ON c.instrument          = j.instrument
 AND c.month_label         = j.month_label
 AND c.week_start          = j.week_start
 AND c.account_id          = j.account_id
 AND c.strategy_id         = j.strategy_id
 AND c.strategy_variant_id = j.strategy_variant_id
 AND c.user_id             = j.user_id
GROUP BY
    j.instrument, j.month_label, j.week_start, j.week_end, j.week_label,
    j.account_id, j.strategy_id, j.strategy_name,
    j.strategy_variant_id, j.strategy_variant_name,
    j.user_id, j.user_name,
    c.cum_monthly_mtm
ORDER BY
    j.month_label,
    j.week_start,
    j.strategy_name,
    j.strategy_variant_name,
    j.user_name,
    j.account_id;
