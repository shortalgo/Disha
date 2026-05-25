import datetime as dt
import multiprocessing
import numpy as np
import pandas as pd
import psycopg2
#import talib as ta
import time
from tqdm import tqdm
from functools import partial
import os
from datetime import datetime, timedelta
import ast, json, sys, re
sys.path.insert(0, r"/home/newberry3/user/")
import warnings
warnings.filterwarnings("ignore")
import os
from datetime import datetime, timedelta
import ast, json, sys, re
from scipy.optimize import brentq
from scipy.stats import norm


# from Common_Functions.utils import (
#     # data i/o / prep
#     postgresql_query,
#     pull_index_data,
#     pull_options_data_d,
#     resample_data,
#     # resample_data_options,
#     # nearest_multiple,
#     # round_to_next_5_minutes,
#     # round_to_next_5_minutes_d,
#     compare_month_and_year,

#     # # indicator
#     # atr_wilder,
#     # supertrend,

#     # option math (BS / IV / Greeks / time to expiry)
#     black_scholes_price,
#     implied_volatility,
#     black_scholes_greeks,
#     time_to_expiry_years,

#     # strike selection by delta
#     choose_option_by_delta,
#     delta_for_row
# )

# (Optional, keep only if used elsewhere in your script)
# from Common_Functions.utils import get_open_range, check_crossover, get_target_stoploss


DEBUG_LOAD = True  # flip to False later

def debug_check_index_df(label, df):
    print(f"\n=== {label} INDEX DF ===")
    print("rows:", len(df), "| columns:", list(df.columns))
    print("index type:", type(df.index).__name__)
    if len(df.index):
        try:
            print("time range:", df.index.min(), "→", df.index.max())
        except Exception:
            pass
    required = ["Open","High","Low","Close"]
    missing = [c for c in required if c not in df.columns]
    print("missing O/H/L/C:", missing if missing else "None")
    if not missing:
        print("NaNs O/H/L/C:", df[required].isna().sum().to_dict())
        print("numeric dtypes O/H/L/C:", {c: str(df[c].dtype) for c in required})
    print("sorted by time:", df.index.is_monotonic_increasing)
    print("head:\n", df.head(3))
    print("tail:\n", df.tail(3))

def debug_check_option_df(option_df):
    print("\n=== OPTIONS DF ===")
    print("rows:", len(option_df), "| columns:", list(option_df.columns))
    print("index type:", type(option_df.index).__name__)
    if len(option_df.index):
        try:
            print("time range:", option_df.index.min(), "→", option_df.index.max())
        except Exception:
            pass
    needed = ["Type","StrikePrice","ExpiryDate","Open","High","Low","Close"]
    missing = [c for c in needed if c not in option_df.columns]
    print("missing key cols:", missing if missing else "None")
    if "Type" in option_df.columns:
        vals = option_df["Type"].astype(str).str.upper().unique()
        print("Type unique:", vals[:10])
    if "StrikePrice" in option_df.columns:
        print("Strike dtype:", option_df["StrikePrice"].dtype)
    if "ExpiryDate" in option_df.columns:
        if not np.issubdtype(option_df["ExpiryDate"].dtype, np.datetime64):
            option_df["ExpiryDate"] = pd.to_datetime(option_df["ExpiryDate"], errors="coerce")
        print("ExpiryDate dtype:", option_df["ExpiryDate"].dtype)
        print("Expiry range:", option_df["ExpiryDate"].min(), "→", option_df["ExpiryDate"].max())
        # show a tiny per-expiry count sample
        try:
            print("counts by expiry (sample):\n",
                  option_df.groupby(option_df["ExpiryDate"].dt.date).size().head())
        except Exception:
            pass
    # snapshot at a random/median timestamp to confirm CE/PE presence
    if len(option_df) > 0:
        mid_ts = option_df.index[len(option_df)//2]
        snap = option_df.loc[mid_ts]
        if isinstance(snap, pd.Series):
            snap = snap.to_frame().T
        ce = (snap["Type"].astype(str).str.upper() == "CE").sum()
        pe = (snap["Type"].astype(str).str.upper() == "PE").sum()
        print(f"snapshot @{mid_ts}: rows={len(snap)}, CE={ce}, PE={pe}")
        print(snap.head(3))

def debug_check_calendar(mapped_days):
    print("\n=== EXPIRY CALENDAR ===")
    md = mapped_days.copy()
    md["Date"] = pd.to_datetime(md["Date"], errors="coerce")
    md["ExpiryDate"] = pd.to_datetime(md["ExpiryDate"], errors="coerce")
    print("rows:", len(md))
    if "Date" in md and "ExpiryDate" in md:
        print("Date range:", md["Date"].min(), "→", md["Date"].max())
        print("Expiry range:", md["ExpiryDate"].min(), "→", md["ExpiryDate"].max())
    print("tail:\n", md.tail(3))


def postgresql_query(input_query, input_tuples = None):
    try:
        connection = psycopg2.connect(
            host="192.168.18.18",
            port = 5432,
            database="postgres",
            user="postgres",
            password="New@123",
        )
        
        cursor = connection.cursor()
        
        if input_tuples is not None:
            cursor.execute(input_query, input_tuples)
        else:
            cursor.execute(input_query)
        
        data = cursor.fetchall()
    
    except psycopg2.Error as e:
        print('Error connecting to the database:', e)
        return e
    
    else:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        
        return data

def black_scholes_price(S, K, T, r, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return None

def implied_volatility(option_price, S, K, T, r, option_type):
    try:
        return brentq(
            lambda sigma: black_scholes_price(S, K, T, r, sigma, option_type) - option_price,
            a=0.01, b=3.0, maxiter=1000, xtol=1e-6
        )
    except (ValueError, RuntimeError):
        return None

def black_scholes_greeks(S, K, T, r, sigma, option_type):
    if sigma <= 0 or T <= 0:
        return None
    F = S * np.exp(r * T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        delta = np.exp(-r * T) * norm.cdf(d1)
        theta = (-F * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = -np.exp(-r * T) * norm.cdf(-d1)
        theta = (-F * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = norm.pdf(d1) / (F * sigma * np.sqrt(T))
    vega = F * norm.pdf(d1) * np.sqrt(T) / 100
    return {
        'Delta': round(delta, 5),
        'Gamma': round(gamma, 5),
        'Vega': round(vega, 5),
        'Theta': round(theta, 5),
        'Rho': round(rho, 5)
    }

from datetime import time as _time

def calculate_time_to_expiry(manual_datetime_str, expiry_date_str):
    now = datetime.strptime(manual_datetime_str, "%Y-%m-%d %H:%M:%S")
    expiry_date = datetime.strptime(expiry_date_str, "%d-%m-%y").date()

    market_open = time(9, 15)
    market_close = time(15, 30)
    today = now.date()
    days_left = (expiry_date - today).days

    if days_left <= 0:
        days_left += 1

    total_trading_minutes = (market_close.hour * 60 + market_close.minute) - (market_open.hour * 60 + market_open.minute)
    current_minutes_since_open = (now.hour * 60 + now.minute) - (market_open.hour * 60 + market_open.minute)

    if current_minutes_since_open < 0:
        T = round(days_left / 365, 6)
    elif current_minutes_since_open >= total_trading_minutes:
        T = round(max(0, (days_left - 1) / 365), 6)
    else:
        fraction_of_day_passed = current_minutes_since_open / total_trading_minutes
        T = round((days_left - fraction_of_day_passed) / 365, 6)
    return T


# Expects these helpers to exist in your environment:
# - calculate_time_to_expiry(entry_time_str, expiry_str_dmy)
# - implied_volatility(option_price, S, K, T, r, option_type_bs)
# - black_scholes_greeks(S, K, T, r, sigma, option_type_bs)

def choose_option_by_delta(minute, daily_option_data, r=0.066):
    """
    Delta-based strike selector (does NOT use ATM).
      • CE: +0.4Δ (short), +0.2Δ (long)
      • PE: -0.4Δ (short), -0.2Δ (long)

    Expiry policy:
      • Mon–Wed  → FAR (2nd upcoming expiry at/after 'minute')
      • Thu/Fri  → NEAREST (1st upcoming expiry at/after 'minute')

    Returns a dict:
    {
      'expiry_used': Timestamp|None,
      'CE': {'delta_0p4_strike','delta_0p4_price','delta_0p2_strike','delta_0p2_price'},
      'PE': {'delta_0p4_strike','delta_0p4_price','delta_0p2_strike','delta_0p2_price'}
    }
    All fields may be None if data/spot is unavailable.
    """

    # -------------------------
    # Guard & normalization
    # -------------------------
    df_day = daily_option_data.copy()
    if not isinstance(df_day.index, pd.DatetimeIndex):
        if 'DateTime' in df_day.columns:
            df_day['DateTime'] = pd.to_datetime(df_day['DateTime'])
            df_day = df_day.set_index('DateTime')
        else:
            return {
                'expiry_used': None,
                'CE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                       'delta_0p2_strike': None, 'delta_0p2_price': None},
                'PE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                       'delta_0p2_strike': None, 'delta_0p2_price': None},
            }

    # Required cols
    req = {'StrikePrice','Type','Open','ExpiryDate'}
    if len(req - set(df_day.columns)) > 0:
        return {
            'expiry_used': None,
            'CE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                   'delta_0p2_strike': None, 'delta_0p2_price': None},
            'PE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                   'delta_0p2_strike': None, 'delta_0p2_price': None},
        }

    df_day['StrikePrice'] = pd.to_numeric(df_day['StrikePrice'], errors='coerce')
    df_day['ExpiryDate']  = pd.to_datetime(df_day['ExpiryDate'])
    df_day['Type']        = df_day['Type'].astype(str).str.upper()

    minute = pd.to_datetime(minute)

    # -------------------------
    # Time slice (exact or +5m fallback)
    # -------------------------
    exact = df_day[df_day.index.floor('min') == minute.floor('min')]
    if exact.empty:
        tmax = minute + pd.Timedelta(minutes=5)
        window = df_day[(df_day.index >= minute) & (df_day.index <= tmax)]
        if window.empty:
            return {
                'expiry_used': None,
                'CE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                       'delta_0p2_strike': None, 'delta_0p2_price': None},
                'PE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                       'delta_0p2_strike': None, 'delta_0p2_price': None},
            }
        first_ts = window.index.min()
        tick_df = window[window.index == first_ts].copy()
    else:
        tick_df = exact.copy()

    # -------------------------
    # Expiry choice: NEAREST vs FAR
    # -------------------------
    future_exp = (df_day[df_day.index >= minute]['ExpiryDate']
                  .dropna().drop_duplicates().sort_values().tolist())
    if not future_exp:
        return {
            'expiry_used': None,
            'CE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                   'delta_0p2_strike': None, 'delta_0p2_price': None},
            'PE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                   'delta_0p2_strike': None, 'delta_0p2_price': None},
        }

    nearest_exp = future_exp[0]
    far_exp     = future_exp[1] if len(future_exp) > 1 else future_exp[0]
    wd = minute.weekday()  # Mon=0 ... Fri=5
    chosen_expiry = far_exp if wd in (0, 1, 2) else nearest_exp

    # Restrict to chosen expiry rows at the selected timestamp
    tick_df = tick_df[tick_df['ExpiryDate'] == chosen_expiry].copy()
    if tick_df.empty:
        return {
            'expiry_used': pd.to_datetime(chosen_expiry),
            'CE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                   'delta_0p2_strike': None, 'delta_0p2_price': None},
            'PE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                   'delta_0p2_strike': None, 'delta_0p2_price': None},
        }

    # -------------------------
    # Spot detection (no ATM fallback)
    # -------------------------
    spot_S = None
    for col in ['Spot','Underlying','UnderlyingPrice','IndexSpot','SpotPrice','LTPUnderlying']:
        if col in tick_df.columns:
            val = tick_df[col].dropna()
            if not val.empty:
                try:
                    spot_S = float(val.iloc[0])
                    break
                except Exception:
                    pass

    if spot_S is None:
        # No reliable spot → do not guess
        return {
            'expiry_used': pd.to_datetime(chosen_expiry),
            'CE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                   'delta_0p2_strike': None, 'delta_0p2_price': None},
            'PE': {'delta_0p4_strike': None, 'delta_0p4_price': None,
                   'delta_0p2_strike': None, 'delta_0p2_price': None},
        }

    # -------------------------
    # Delta-based picker
    # -------------------------
    def find_strike_by_delta(option_type, target_delta):
        """
        Among rows for chosen expiry and selected timestamp, compute IV/Greeks and
        pick strike minimizing |Delta - target_delta|.
        """
        df = tick_df[tick_df['Type'] == option_type].copy()
        if df.empty:
            return None, None

        option_type_bs = 'call' if option_type == 'CE' else 'put'
        T = calculate_time_to_expiry(minute.strftime("%Y-%m-%d %H:%M:%S"),
                                     pd.to_datetime(chosen_expiry).strftime("%d-%m-%y"))

        best = (None, None)
        best_diff = np.inf

        for _, row in df.iterrows():
            K = float(row['StrikePrice'])
            price = float(row['Open'])
            if not np.isfinite(K) or not np.isfinite(price) or price <= 0:
                continue

            iv = implied_volatility(price, spot_S, K, T, r, option_type_bs)
            if iv is None or iv <= 0:
                continue

            greeks = black_scholes_greeks(spot_S, K, T, r, iv, option_type_bs)
            if not greeks or ('Delta' not in greeks):
                continue

            delta = greeks['Delta']
            diff = abs(delta - target_delta)
            if diff < best_diff:
                best_diff = diff
                best = (K, price)

        return best  # (strike, price) or (None, None)

    # Compute targets
    CE_04_strike, CE_04_price = find_strike_by_delta('CE',  0.4)
    CE_02_strike, CE_02_price = find_strike_by_delta('CE',  0.2)
    PE_04_strike, PE_04_price = find_strike_by_delta('PE', -0.4)
    PE_02_strike, PE_02_price = find_strike_by_delta('PE', -0.2)

    # -------------------------
    # Output
    # -------------------------
    return {
        'expiry_used': pd.to_datetime(chosen_expiry),
        'CE': {
            'delta_0p4_strike': CE_04_strike, 'delta_0p4_price': CE_04_price,  # SHORT if bearish
            'delta_0p2_strike': CE_02_strike, 'delta_0p2_price': CE_02_price,  # LONG
        },
        'PE': {
            'delta_0p4_strike': PE_04_strike, 'delta_0p4_price': PE_04_price,  # SHORT if bullish
            'delta_0p2_strike': PE_02_strike, 'delta_0p2_price': PE_02_price,  # LONG
        },
    }

def resample_data(data: pd.DataFrame, TIME_FRAME: str) -> pd.DataFrame:
    """
    Resample to a given timeframe with financial-style OHLC aggregation.
    Special case: for 1H bars, align to 09:15–10:15, 10:15–11:15, ...

    Parameters
    ----------
    data : DataFrame
        Must have a DatetimeIndex and columns ['Open','High','Low','Close', ...].
        'ExpiryDate' is optional (will be carried as 'last' if present).
    TIME_FRAME : str
        Pandas-compatible frequency string (e.g., '5T', '15T', '1H').

    Returns
    -------
    DataFrame
        Resampled with columns: Open, High, Low, Close, (optional) ExpiryDate, Date, Time.
        Index is the bar END timestamp.
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("resample_data: DataFrame index must be a DatetimeIndex")

    data = data.sort_index()

    # Build aggregation dict dynamically (carry ExpiryDate if present)
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}
    if 'ExpiryDate' in data.columns:
        agg['ExpiryDate'] = 'last'

    # Hourly: anchor to :15 so bars are 09:15–10:15, 10:15–11:15, ...
    hourly_aliases = {'1H', '60min', '60T'}
    if TIME_FRAME in hourly_aliases:
        res = data.resample(
            '60min',
            offset='15min',   # anchor windows to start at :15
            label='right',    # bar timestamp at the END (e.g., 10:15)
            closed='right'    # (prev, current] → e.g., 09:15–10:15
        ).agg(agg)
    else:
        # Generic resample for other frames
        res = data.resample(
            TIME_FRAME,
            label='right',
            closed='right'
        ).agg(agg)

    res = res.dropna(subset=['Open','High','Low','Close'])  # ensure valid OHLC bars
    res['Date'] = res.index.date
    res['Time'] = res.index.time
    return res


def nearest_multiple(x, n):
    """
    Returns the nearest multiple of `n` to the given number `x`.
    If `x` is exactly halfway between two multiples of `n`, rounds up to the higher multiple.

    Args:
        x (int or float): The number to round.
        n (int): The multiple to which to round.

    Returns:
        int: The nearest multiple of `n` to `x`.
    """
    remainder = x%n
    if remainder < n/2:
        nearest = x - remainder
    else:
        nearest = x - remainder + n
    return int(nearest)



# Function to get premium
def get_final_premium(premium_data, option_data, RATIO):
    """
    Calculates the final premium values for a set of option trades by merging premium data with option price data,
    applying lot size and brokerage adjustments, and computing the net premium for each trade.
    Args:
        premium_data (pd.DataFrame): DataFrame containing trade information such as Date, Time, ExpiryDate, ATM, CE_OTM, PE_OTM, and Position.
        option_data (pd.DataFrame): DataFrame containing option price data with columns including Ticker, Type ('CE' or 'PE'), StrikePrice, and Open.
        RATIO (tuple): A tuple (long_lots, short_lots) specifying the number of lots for long and short positions.
    Returns:
        pd.DataFrame: The input premium_data DataFrame augmented with calculated premium columns for each leg (CE_ATM, PE_ATM, CE_OTM, PE_OTM),
                      the total Premium, and DaysToExpiry.
    """
    option_data_ce = option_data[option_data['Type'] == 'CE'].drop(columns=['Ticker', 'Type'])
    option_data_pe = option_data[option_data['Type'] == 'PE'].drop(columns=['Ticker', 'Type'])
    
    del option_data

    premium_data['Time'] = pd.to_datetime(premium_data['Time'], format='%H:%M:%S')
    premium_data['Time'] = premium_data['Time'].dt.strftime('%H:%M')
    premium_data['ATM'] = premium_data['ATM'].astype('int32')

    premium_data = premium_data.merge(option_data_ce, left_on=['Date', 'Time', 'ExpiryDate', 'ATM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    
    premium_data = premium_data.rename(columns={'Open' : 'CE_ATM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_pe, left_on=['Date', 'Time', 'ExpiryDate', 'ATM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'PE_ATM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_ce, left_on=['Date', 'Time', 'ExpiryDate', 'CE_OTM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'CE_OTM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    premium_data = premium_data.merge(option_data_pe, left_on=['Date', 'Time', 'ExpiryDate', 'PE_OTM'],
                                                   right_on=['Date', 'Time', 'ExpiryDate', 'StrikePrice'], how = 'left')
    premium_data = premium_data.rename(columns={'Open' : 'PE_OTM_Premium'})
    premium_data = premium_data.drop(['StrikePrice'], axis=1)

    long_lots = RATIO[0]
    short_lots = RATIO[1]

    premium_data['CE_ATM_Price'] = np.where(premium_data['Position'] == 1, long_lots * (-LOT_SIZE * premium_data['CE_ATM_Premium'] * 1.01 - brokerage), long_lots * (LOT_SIZE * premium_data['CE_ATM_Premium'] * 0.99 - brokerage))
    premium_data['PE_ATM_Price'] = np.where(premium_data['Position'] == 1, long_lots * (-LOT_SIZE * premium_data['PE_ATM_Premium'] * 1.01 - brokerage), long_lots * (LOT_SIZE * premium_data['PE_ATM_Premium'] * 0.99 - brokerage))
    premium_data['CE_OTM_Price'] = np.where(premium_data['Position'] == 1, short_lots * (LOT_SIZE * premium_data['CE_OTM_Premium'] * 0.99 - brokerage), short_lots * (-LOT_SIZE * premium_data['CE_OTM_Premium'] * 1.01 - brokerage))
    premium_data['PE_OTM_Price'] = np.where(premium_data['Position'] == 1, short_lots * (LOT_SIZE * premium_data['PE_OTM_Premium'] * 0.99 - brokerage), short_lots * (-LOT_SIZE * premium_data['PE_OTM_Premium'] * 1.01 - brokerage))

    premium_data['Premium'] = premium_data['CE_ATM_Price'] + premium_data['PE_ATM_Price'] + premium_data['CE_OTM_Price']  + premium_data['PE_OTM_Price']

    premium_data['Date'] = pd.to_datetime(premium_data['Date'], format='%Y-%m-%d')
    premium_data['ExpiryDate'] = pd.to_datetime(premium_data['ExpiryDate'], format='%Y-%m-%d')

    premium_data['DaysToExpiry'] = (premium_data['ExpiryDate'] - premium_data['Date']).dt.days
    premium_data['DaysToExpiry'] = np.where(premium_data['DaysToExpiry']==6, 4, np.where(premium_data['DaysToExpiry']==5, 3, premium_data['DaysToExpiry']))


    return premium_data

# Function to pull options data for specified date range 
def pull_options_data_d(start_date, end_date, option_data_path, stock):
            """
            Loads and concatenates options data from pickled files within a specified date range and for a specific stock.
            Args:
                start_date (str or datetime): The start date for filtering option data files.
                end_date (str or datetime): The end date for filtering option data files.
                option_data_path (str): The directory path where option data files are stored.
                stock (str): The stock ticker symbol to filter relevant option data files.
            Returns:
                pd.DataFrame: A DataFrame containing concatenated options data filtered by date and stock, 
                              with columns ['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open', 'High', 'Low', 'Close', 'Ticker'].
                              The DataFrame index is set to the parsed datetime from the 'Ticker' column and named 'DateTime'.
            Notes:
                - Only files matching the month and year criteria (as determined by compare_month_and_year) are loaded.
                - The function prints the columns of the resulting DataFrame and the time taken to load the data.
                - The 'StrikePrice' column is cast to int32 and 'Type' to category for memory efficiency.
            """
            
            start_time = time.time()
            option_data_files = next(os.walk(option_data_path))[2]
            option_data = pd.DataFrame()

            for file in option_data_files:

                file1 = compare_month_and_year(start_date, end_date, file, stock)
                    
                if not file1:
                    continue

                temp_data = pd.read_pickle(option_data_path + file)[['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open','High' , 'Low' ,'Close', 'Ticker']]
                temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
                temp_data = temp_data.rename_axis('DateTime')
                option_data = pd.concat([option_data, temp_data])

            print('Option data columns :', option_data.columns)
            option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
            option_data['Type'] = option_data['Type'].astype('category')
            
            end_time = time.time()
            print('Time taken to pull Options data :', (end_time-start_time))

            return option_data

# Function to pull index data for specified date range
def pull_index_data(start_date_idx, end_date_idx, stock):
    """
    Retrieves index data for a given stock between specified start and end date indices.
    This function queries a PostgreSQL database for OHLC (Open, High, Low, Close) data and ticker information
    for the specified stock within the provided date range and during trading hours (09:15 to 15:29).
    The resulting data is merged with a mapped_days DataFrame, indexed by datetime, sorted, and duplicates are removed.
    Args:
        start_date_idx (str): The start date index in 'YYYYMMDD' format.
        end_date_idx (str): The end date index in 'YYYYMMDD' format.
        stock (str): The stock symbol for which to retrieve index data.
    Returns:
        pandas.DataFrame: A DataFrame containing the merged and processed index data, indexed by datetime.
    """

    start_time = time.time()
    print(start_date_idx, end_date_idx)
    table_name = stock + '_IDX'
    data = postgresql_query(f'''
                            SELECT "Open", "High", "Low", "Close", "Ticker"
                            FROM "{table_name}"
                            WHERE "Date" >= '{start_date_idx}'
                            AND "Date" <= '{end_date_idx}'
                            AND "Time" BETWEEN '09:15' AND '15:29'
                            ''')

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Time taken to get Index Data:', elapsed_time)

    column_names = ['Open', 'High', 'Low', 'Close', 'Ticker']
    index_data = pd.DataFrame(data, columns = column_names)
    # make index_data['Date'] datetime64 (no .astype(str))
    index_data['Date'] = pd.to_datetime(index_data['Ticker'].str[0:8], format='%Y%m%d').dt.normalize()

    # make a safe copy of mapped_days and coerce its Date/ExpiryDate
    md = mapped_days.copy()
    md['Date'] = pd.to_datetime(md['Date'], errors='coerce').dt.normalize()
    if 'ExpiryDate' in md.columns:
        md['ExpiryDate'] = pd.to_datetime(md['ExpiryDate'], errors='coerce')

    # merge on matching dtypes
    df = index_data.merge(md, on='Date', how='left')

    df.index = pd.to_datetime(df['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
    df = df.rename_axis('DateTime')
    df = df.sort_index()
    df = df.drop_duplicates()
    
    return df


def trade_sheet_creator_st(mapped_days, option_data, idx_1h_st, start_date, end_date,
                           output_folder_path, lot_size=None, fee_per_leg=0.0, r=0.066):
    if lot_size is None:
        # use your global LOT_SIZE if you want, or compute from stock here
        lot_size = LOT_SIZE
    """Enter on ST flips; bullish => short 0.4Δ PE (current) + long 0.2Δ PE (far).
       Bearish => short 0.4Δ CE (current) + long 0.2Δ CE (far).
       Exit on opposite flip, short-leg expiry, or TP."""
    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date).date()
    sub = idx_1h_st[(idx_1h_st.index.date >= start_d) & (idx_1h_st.index.date <= end_d)].copy()

    results = {}
    for tp in TP_LIST:
        trades = []
        position = None
        for t, bar in sub.iterrows():
            spot = float(bar['Close'])
            trend = int(bar['ST_dir'])
            flip  = bool(bar['ST_signal'])
            cur_exp, far_exp = get_next_expiries(t, mapped_days)

            # --- exit ---
            if position is not None:
                reason = None
                # signal flip
                if flip and ((trend == 1 and position['side']=='bearish') or (trend == -1 and position['side']=='bullish')):
                    reason = 'signal'
                # short expiry
                elif pd.Timestamp(t).date() >= position['short_expiry']:
                    reason = 'expiry'
                # profit target
                elif tp is not None and position['entry_value'] is not None:
                    mtm = current_net_value(option_data, t, position['short_leg'], position['long_leg'], lot_size, fee_per_leg)
                    if mtm is not None:
                        pnl = position['entry_value'] - mtm
                        if pnl >= tp * abs(position['entry_value']):
                            reason = 'tp'

                if reason is not None:
                    exit_val = current_net_value(option_data, t, position['short_leg'], position['long_leg'], lot_size, fee_per_leg)
                    if exit_val is not None:
                        trades.append({
                            'entry_t': position['entry_t'],
                            'side': position['side'],
                            'short_type': position['short_leg']['type'],
                            'short_strike': position['short_leg']['strike'],
                            'short_expiry': position['short_leg']['expiry'],
                            'long_type': position['long_leg']['type'],
                            'long_strike': position['long_leg']['strike'],
                            'long_expiry': position['long_leg']['expiry'],
                            'tp': tp,
                            'exit_t': t,
                            'entry_value': position['entry_value'],
                            'exit_value': exit_val,
                            'pnl': position['entry_value'] - exit_val,
                            'exit_reason': reason,
                        })
                        position = None  # flat

            # --- entry on flip ---
            if position is None and flip:
                if trend == 1:
                    # bullish: PUTs
                    s = choose_option_by_delta(option_data, t, cur_exp, 'PE', -0.4, spot, r)
                    l = choose_option_by_delta(option_data, t, far_exp, 'PE', -0.2, spot, r)
                    side = 'bullish'
                elif trend == -1:
                    # bearish: CALLs
                    s = choose_option_by_delta(option_data, t, cur_exp, 'CE', +0.4, spot, r)
                    l = choose_option_by_delta(option_data, t, far_exp, 'CE', +0.2, spot, r)
                    side = 'bearish'
                else:
                    s = l = None

                if s is not None and l is not None:
                    ev = entry_net_value(float(s['Close']), float(l['Close']), lot_size, fee_per_leg)
                    position = {
                        'side': side,
                        'entry_t': t,
                        'short_leg': {'type': s['Type'], 'strike': float(s['StrikePrice']), 'expiry': cur_exp},
                        'long_leg':  {'type': l['Type'], 'strike': float(l['StrikePrice']), 'expiry': far_exp},
                        'short_expiry': cur_exp,
                        'entry_value': ev,
                    }

        results[tp] = pd.DataFrame(trades)

    # write CSVs
    base = f"{stock}_ST_1H_{start_date}_{end_date}"
    for tp, df_tr in results.items():
        tag = "none" if tp is None else str(tp).replace('.','p')
        out = os.path.join(output_folder_path, f"{base}_tp_{tag}.csv")
        if not df_tr.empty:
            df_tr.to_csv(out, index=False)
    return results




TP_LIST = [0.8, 0.85, 0.9, 0.95, None]   # profit targets

def get_next_expiries(t, mapped_days):
    md = mapped_days.copy()
    md['Date'] = pd.to_datetime(md['Date']).dt.date
    md['ExpiryDate'] = pd.to_datetime(md['ExpiryDate']).dt.date
    d = pd.Timestamp(t).date()
    cands = md.loc[md['Date'] >= d, 'ExpiryDate'].dropna().sort_values().unique()
    if len(cands) == 0:
        return None, None
    if len(cands) == 1:
        return cands[0], None
    return cands[0], cands[1]

def get_leg_price(option_data, t, expiry_date, strike, typ_code):
    """Return Close price for locked leg at/after t (fallback before t)."""
    t = pd.Timestamp(t)
    idx = option_data.index.searchsorted(t, side='left')
    for i in (idx, idx-1):
        if i < 0 or i >= len(option_data.index):
            continue
        ts = option_data.index[i]
        snap = option_data.loc[ts]
        if isinstance(snap, pd.Series):
            snap = snap.to_frame().T
        mask = (
            (snap['Type'] == typ_code) &
            (snap['StrikePrice'].astype(float) == float(strike)) &
            (pd.to_datetime(snap['ExpiryDate']).dt.date == pd.Timestamp(expiry_date).date())
        )
        rows = snap.loc[mask]
        if not rows.empty:
            return float(rows['Close'].iloc[0])
    return None

def entry_net_value(short_px, long_px, lot_size=25, fee_per_leg=0.0):
    gross = (short_px - long_px) * lot_size
    fees = 2 * fee_per_leg
    return gross - fees

def current_net_value(option_data, t, short_leg, long_leg, lot_size=25, fee_per_leg=0.0):
    sp = get_leg_price(option_data, t, short_leg['expiry'], short_leg['strike'], short_leg['type'])
    lp = get_leg_price(option_data, t, long_leg['expiry'], long_leg['strike'], long_leg['type'])
    if sp is None or lp is None:
        return None
    gross = (sp - lp) * lot_size
    fees = 2 * fee_per_leg
    return gross - fees

########################################### INPUTS #####################################################

# Core identifiers
superset     = 'SupertrendDeltaSpread'   # folder group for this strategy
stock        = 'NIFTY'                   # NIFTY / BANKNIFTY / SENSEX ...
option_type  = 'ND'                      # keep your existing categorization

# Market-specific settings
roundoff  = 50 if stock == 'NIFTY' else (100 if stock in ('BANKNIFTY', 'SENSEX') else None)
brokerage = 4.5 if stock == 'NIFTY' else (3.0 if stock in ('BANKNIFTY', 'SENSEX') else 0.0)
LOT_SIZE  = 25 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else 10)

# Risk-free rate (annualized) used in BS/Greeks
RISK_FREE_RATE = 0.066

# Supertrend parameters (1H timeframe)
CANDLE_TIMEFRAME = '1H'
ST_PERIOD = 10
ST_MULT   = 3.0

# Delta targets (absolute)
DELTA_SHORT_TARGET = 0.40   # short leg target delta
DELTA_LONG_TARGET  = 0.20   # long leg target delta

# Profit targets to sweep (fraction of entry credit); None = no TP
TP_LIST = [0.8, 0.85, 0.9, 0.95, None]

# --- Paths (use os.path.join to avoid missing slashes) ---
root_path = os.path.join("/home/newberry3/user", superset, stock, option_type)
output_folder_path = os.path.join(root_path, "Trade_Sheets")
# (Optional; not used by this strategy flow but harmless to keep)
filter_df_path = os.path.join(root_path, "Filter_Sheets")
txt_file_path  = os.path.join(root_path, "new_done.txt")

# Expiry calendar file
expiry_file_path = "/home/newberry3/disha/Supertrend/Common_Functions/NIFTY Market Dates.xlsx"

# Option data location
if stock == 'NIFTY':
    option_data_path = "/home/newberry3/main/Data/NIFTY/"
elif stock == 'BANKNIFTY':
    option_data_path = "/home/newberry3/main/Data/BANKNIFTY"
elif stock == 'SENSEX':
    option_data_path = "/home/newberry3/main/Data/SENSEX"
else:
    # fallback: set your custom path for other symbols
    option_data_path = "/home/newberry3/main/Data/OTHER"

# Ensure folders exist
os.makedirs(root_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)  # optional
if not os.path.exists(txt_file_path):       # optional
    open(txt_file_path, 'a').close()

# Define the backtesting date ranges
date_ranges = [ 
    ('2024-06-01', '2024-10-30')
    # ('2024-02-01', '2024-05-31'),
    # ('2023-10-01', '2024-01-31'),
    # ('2023-06-01', '2023-09-30'),
    # ('2023-02-01', '2023-05-31'),
    # ('2022-10-01', '2023-01-31'),
    # ('2022-06-01', '2022-09-30'),
    # ('2022-01-01', '2022-05-31'), 
    # ('2021-06-01', '2021-12-31')
]


# Function that handles trade logic for each parameter set (Supertrend + delta legs, 1H)
def parameter_process(parameter, mapped_days, option_data, df, start_date, end_date, counter, output_folder_path):
    """
    Backward-compatible unpack:
      - old: (TIME_FRAME, STRIKE, ENTRY, EXIT)
      - new (optional): (TIME_FRAME, STRIKE, ENTRY, EXIT, ST_PERIOD, ST_MULT)

    Uses fixed 1H timeframe, computes Supertrend, and runs the ST-delta engine.
    Returns a key string for logging (similar to the old behavior).
    """
    # --- unpack with compatibility ---
    ST_PERIOD, ST_MULT = 10, 3.0  # defaults
    try:
        # try (TIME_FRAME, STRIKE, ENTRY, EXIT, ST_PERIOD, ST_MULT)
        TIME_FRAME, STRIKE, ENTRY, EXIT, ST_PERIOD, ST_MULT = parameter
    except ValueError:
        # fall back to (TIME_FRAME, STRIKE, ENTRY, EXIT)
        TIME_FRAME, STRIKE, ENTRY, EXIT = parameter

    # --- force 1H timeframe for this strategy ---
    TIME_FRAME = '1H'

    # --- build 1H candles & Supertrend ---
    idx_1h = resample_data(df, TIME_FRAME)[['Open', 'High', 'Low', 'Close', 'ExpiryDate']].dropna()
    idx_1h = supertrend(idx_1h, period=ST_PERIOD, multiplier=ST_MULT)

    # --- ensure option expiries are datetime for downstream selection ---
    if 'ExpiryDate' in option_data.columns:
        option_data['ExpiryDate'] = pd.to_datetime(option_data['ExpiryDate'], errors='coerce')

    # --- run the Supertrend + delta 0.4/0.2 engine (writes one CSV per TP internally) ---
    # NOTE: trade_sheet_creator_st should NOT have lot_size=LOT_SIZE as a default in its signature.
    #       It's safe to pass LOT_SIZE here (evaluated at call-time, after it's defined).
    _results = trade_sheet_creator_st(
        mapped_days=mapped_days,
        option_data=option_data,
        idx_1h_st=idx_1h,
        start_date=start_date,
        end_date=end_date,
        output_folder_path=output_folder_path,
        lot_size=LOT_SIZE,
        fee_per_leg=0.0,
        r=0.066,
    )

    # return a simple key string for "done" logging
    return f"ST_{TIME_FRAME}_P{ST_PERIOD}_M{ST_MULT}_{start_date}_{end_date}"

# Adjust strike step for BANKNIFTY or SENSEX (legacy; safe to keep even if unused)
if stock in ('BANKNIFTY', 'SENSEX'):
    strikes = [x * 2 for x in strikes]

# === MAIN ===
if __name__ == "__main__":
    counter = 0

    # ---- Fix the global start/end (use earliest start & latest end across ranges)
    start_date_idx = min(pd.to_datetime(r[0]) for r in date_ranges).strftime("%Y-%m-%d")
    end_date_idx   = max(pd.to_datetime(r[1]) for r in date_ranges).strftime("%Y-%m-%d")

    # ---- Load expiry map & coerce types
    mapped_days = pd.read_excel(expiry_file_path)
    # rename if the old column exists
    if 'WeeklyDaysToExpiry' in mapped_days.columns and 'DaysToExpiry' not in mapped_days.columns:
        mapped_days = mapped_days.rename(columns={'WeeklyDaysToExpiry': 'DaysToExpiry'})
    mapped_days['Date'] = pd.to_datetime(mapped_days['Date'], errors='coerce')
    mapped_days['ExpiryDate'] = pd.to_datetime(mapped_days['ExpiryDate'], errors='coerce')
    mapped_days = mapped_days[(mapped_days['Date'] >= pd.to_datetime(start_date_idx)) &
                              (mapped_days['Date'] <= pd.to_datetime(end_date_idx))].sort_values('Date')
    # DEBUG: calendar sanity
    debug_check_calendar(mapped_days)

    # ---- Pull index minute data once for the full span
    df = pull_index_data(start_date_idx, end_date_idx, stock)

    # DEBUG: raw index sanity
    debug_check_index_df("RAW", df)
    # ---- 1H resample + Supertrend (pandas-ta; OHLC only to avoid drops)
ST_PERIOD = 10
ST_MULT   = 3.0

ohlc_1h = resample_data(df, '1H')[['Open','High','Low','Close']].dropna(how='any')
debug_check_index_df("1H (OHLC-only)", ohlc_1h)

# --- import pandas-ta safely (NumPy 2.x alias; no TA-Lib needed)
import numpy as np
if not hasattr(np, "NaN"):  # pandas_ta expects np.NaN
    np.NaN = np.nan
import pandas_ta as pta
print("[debug] pandas-ta OK:", pta.__version__)

def pta_supertrend_df(ohlc: pd.DataFrame, length: int, mult: float) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: ST, ST_dir, ST_signal
    computed via pandas-ta's supertrend.
    """
    ref = pta.supertrend(
        ohlc['High'], ohlc['Low'], ohlc['Close'],
        length=length, multiplier=mult
    )
    # Identify returned columns from pandas-ta
    sup_line_col = next(c for c in ref.columns if c.startswith('SUPERT_') and c.count('_') >= 2)
    dir_col      = next(c for c in ref.columns if c.startswith('SUPERTd_'))
    out = pd.DataFrame({
        'ST': ref[sup_line_col],
        'ST_dir': ref[dir_col],
    }, index=ref.index)
    out['ST_signal'] = out['ST_dir'].ne(out['ST_dir'].shift())
    return out

# Compute ST (and auto-retry for sensitivity if needed)
idx_1h = pta_supertrend_df(ohlc_1h, ST_PERIOD, ST_MULT)
flip_count = int(idx_1h['ST_signal'].sum())
bull = int((idx_1h['ST_dir'] == 1).sum())
bear = int((idx_1h['ST_dir'] == -1).sum())
print(f"[pta] 1H bars={len(idx_1h)}, flips={flip_count}, bull_bars={bull}, bear_bars={bear}")

# show first few flips with Close for context
if flip_count > 0:
    preview = idx_1h[idx_1h['ST_signal']].join(ohlc_1h[['Close']]).head(5)[['Close','ST_dir']]
    print(preview)
else:
    for (p, m) in [(10, 2.0), (7, 3.0), (7, 2.0), (10, 1.5)]:
        tmp = pta_supertrend_df(ohlc_1h, p, m)
        fc  = int(tmp['ST_signal'].sum())
        print(f"[pta] retry ST with period={p}, mult={m} -> flips={fc}")
        if fc > 0:
            ST_PERIOD, ST_MULT = p, m
            idx_1h = tmp
            flip_count = fc
            break
idx_1h = idx_1h.join(ohlc_1h[['Close']], how='left')
# ---- Iterate each date sub-range for options + backtest
for start_date, end_date in date_ranges:
    counter += 1
    print(start_date, end_date, counter)

    start_date_object = pd.to_datetime(start_date)
    end_date_object   = pd.to_datetime(end_date)
    new_end_date      = (end_date_object + timedelta(days=30)).strftime('%Y-%m-%d')  # capture far expiry

    # Pull options data covering [start_date, end_date+30d]
    option_data = pull_options_data_d(start_date, new_end_date, option_data_path, stock)
    # Ensure types required by the engine
    if 'ExpiryDate' in option_data.columns:
        option_data['ExpiryDate'] = pd.to_datetime(option_data['ExpiryDate'], errors='coerce')

    # DEBUG: options sanity (first range only to keep logs readable)
    if counter == 1:
        debug_check_option_df(option_data)

    # Run the Supertrend + 0.4/0.2-delta strategy (writes one CSV per TP)
    _ = trade_sheet_creator_st(
            mapped_days=mapped_days,
            option_data=option_data,
            idx_1h_st=idx_1h,                 # full ST series; function filters by range internally
            start_date=start_date,
            end_date=end_date,
            output_folder_path=output_folder_path,
            lot_size=LOT_SIZE,                 # ensure LOT_SIZE is defined earlier
            fee_per_leg=0.0,
            r=0.066,
        )

print('Finished at :', time.time())
