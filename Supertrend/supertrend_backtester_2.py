# import datetime as dt
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

def resample_data(data, TIME_FRAME):
    """
    Resamples financial time series data to a specified time frame and aggregates OHLC values.
    Parameters:
        data (pd.DataFrame): Input DataFrame with a DateTimeIndex and columns ['Open', 'High', 'Low', 'Close', 'ExpiryDate'].
        TIME_FRAME (str): Pandas-compatible resampling frequency string (e.g., '5T' for 5 minutes, '1H' for 1 hour).
    Returns:
        pd.DataFrame: Resampled DataFrame with aggregated OHLC values, 'ExpiryDate', and additional 'Date' and 'Time' columns.
    """
    
    resampled_data = data.resample(TIME_FRAME).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first'}).dropna()
    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    
    return resampled_data

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

def get_strike(minute, daily_option_data, r=0.066):
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
    index_data['Date'] = pd.to_datetime(index_data['Ticker'].str[0:8], format = '%Y%m%d').astype(str)

    df = index_data.merge(mapped_days, on = 'Date')
    df.index = pd.to_datetime(df['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
    df = df.rename_axis('DateTime')
    df = df.sort_index()
    df = df.drop_duplicates()
    
    return df

def trade_sheet_creator(mapped_days, option_data, df, start_date, end_date, counter, output_folder_path, resampled, TIME_FRAME, STRIKE, ENTRY, EXIT):
        """
        Generates a trade sheet for a given options backtesting strategy over a specified date range.
        This function processes mapped trading days, filters them based on date constraints, and simulates option trades
        according to the provided strategy parameters. It records entry and exit trades, calculates premiums, and saves
        the results to CSV files. Additionally, it maintains a filter DataFrame to track the profitability of different
        strategy parameter combinations.
        Parameters:
            mapped_days (pd.DataFrame): DataFrame containing trading days and expiry information.
            option_data (pd.DataFrame): DataFrame containing option price data indexed by datetime.
            df (pd.DataFrame): DataFrame containing index price data (e.g., NIFTY, BANKNIFTY) indexed by datetime.
            start_date (str): Start date for backtesting in 'YYYY-MM-DD' format.
            end_date (str): End date for backtesting in 'YYYY-MM-DD' format.
            counter (int): Counter used for naming output filter files.
            output_folder_path (str): Path to the folder where trade sheets will be saved.
            resampled (bool): Indicates if the data is resampled (unused in this function).
            TIME_FRAME (int): Candle time frame in minutes for the strategy.
            STRIKE (int): Strike difference for selecting option contracts.
            ENTRY (str): Entry time for trades in 'HH:MM' format.
            EXIT (str): Exit time for trades in 'HH:MM' format.
        Returns:
            str: A string containing the sanitized strategy name along with the start and end dates, 
                 used as a unique identifier for the generated trade sheet.
        Notes:
            - The function expects certain global variables and helper functions (e.g., `nearest_multiple`, `TSL_SS`, `stock`, `filter_df_path`) to be defined elsewhere.
            - The function writes trade sheets and filter DataFrames to CSV files in the specified output directory.
            - Some features, such as re-entry logic and DTE-based profitability checks, are present but commented out.
        """
    
        column_names = ['Date', 'Position', 'Action', 'CE_Time', 'PE_Time', 'Index Value', 'CE_OTM', 'PE_OTM', 'Exit', 'ExpiryDate', 'DaysToExpiry', 'CE_OTM_Premium', 'PE_OTM_Premium', 'EXIT_TYPE']
        trade_sheet = []

        # Parameters for filtering combinations
        target_list = [TIME_FRAME, STRIKE, ENTRY, EXIT]

        # Filter mapped days
        mapped_days_temp = mapped_days[
        (mapped_days['Date'] >= start_date) & 
        (mapped_days['Date'] <= end_date) &
        (mapped_days['Date'] > '2021-06-03') & 
        (mapped_days['Date'] != '2024-05-18') & 
        (mapped_days['Date'] != '2024-05-20') &
        ~((mapped_days['Date'] >= '2024-05-31') & (mapped_days['Date'] <= '2024-06-06'))
        ]

        for _, row in mapped_days_temp.iterrows():
            date = row['Date']
            print(date)
            # expiry_date = row['MonthlyExpiry']
            expiry_date = row['ExpiryDate']
            days_to_expiry = row['DaysToExpiry']

            start_time = pd.to_datetime(f'{date} {ENTRY}:00')
            end_time = pd.to_datetime(f'{expiry_date} {EXIT}:00')

            daily_data_start_time = start_time - pd.Timedelta('5T')
            daily_data = df[(df.index >= daily_data_start_time) & (df.index <= end_time)]

            filter_start_date = pd.to_datetime(date)
            filter_end_date = pd.to_datetime(expiry_date)
            daily_option_data = option_data[(option_data.index.date >= filter_start_date.date()) & 
                                            (option_data.index.date <= filter_end_date.date())]

            position = 0
            entry_time = start_time
            reentry_count = 0  # Initialize re-entry counter

            for minute, dd_row in daily_data.iloc[1:-1].iterrows():
                current_index_open = dd_row['Open']
                if position == 1:
                    break

                # Entry
                if (position == 0) & (minute == entry_time):
                    position = 1

                    if ((date >= '2022-10-19') & (stock == 'FINNIFTY')) or (stock == 'NIFTY'):
                        ATM = nearest_multiple(current_index_open, 50)
                        ATM = nearest_multiple(ATM, 50)
                        CE_OTM = ATM * (1 + 0.03)
                        CE_OTM = nearest_multiple(CE_OTM, 50)
                        PE_OTM = ATM * (1 - 0.03)
                        PE_OTM = nearest_multiple(PE_OTM, 50)
                    else:
                        ATM = nearest_multiple(current_index_open, 100)
                        CE_OTM = ATM + (STRIKE * 2)
                        PE_OTM = ATM - (STRIKE * 2)

                    CE_OTM_entry_price, CE_OTM_exit_price, ce_exit_time_period, PE_OTM_entry_price, PE_OTM_exit_price, pe_exit_time_period, exit_type = TSL_SS(daily_option_data, minute, CE_OTM, PE_OTM, EXIT,days_to_expiry)

                    minute_time = minute.time()

                    if isinstance(ce_exit_time_period, int) or ce_exit_time_period == 0:
                        ce_exit_time_period = end_time
                    if isinstance(pe_exit_time_period, int) or pe_exit_time_period == 0:
                        pe_exit_time_period = end_time

                    ce_exit_time = ce_exit_time_period.time()
                    pe_exit_time = pe_exit_time_period.time()

                    trade_sheet.append(pd.Series([date, 1, 'Short', minute_time, minute_time, current_index_open, CE_OTM, PE_OTM, '', expiry_date, days_to_expiry, CE_OTM_entry_price, PE_OTM_entry_price, ''], index=column_names))
                    trade_sheet.append(pd.Series([date, 0, 'Long', ce_exit_time, pe_exit_time, current_index_open, CE_OTM, PE_OTM, '', expiry_date, days_to_expiry, CE_OTM_exit_price, PE_OTM_exit_price,exit_type], index=column_names))


        strategy_name = f'{stock}_candle_{TIME_FRAME}_strike_{STRIKE}_entry_{ENTRY}_exit_{EXIT}'
        sanitized_strategy_name = strategy_name.replace('.', ',').replace(':', ',')

        
        try:
            trade_sheet = pd.concat(trade_sheet, axis = 1).T
        except Exception as e:
            print(f"An error occurred: {e}")
            return sanitized_strategy_name + '_' + start_date + '_' + end_date

        # trade_sheet = get_final_premium(trade_sheet, option_data, RATIO)
        # trade_sheet['Time'] = pd.to_datetime(trade_sheet['Time'], format='%H:%M')
        # trade_sheet['Time'] = trade_sheet['Time'].dt.strftime('%H:%M:%S')

        trade_sheet['Premium'] = np.where(trade_sheet['Action']=='Short', trade_sheet['CE_OTM_Premium'] + trade_sheet['PE_OTM_Premium'], - (trade_sheet['CE_OTM_Premium'] + trade_sheet['PE_OTM_Premium']))

        # create filter_df to store profitable combo and dte
        filter_df1 = pd.DataFrame(columns=['Strategy', 'Parameters', 'DTE0', 'DTE1', 'DTE2', 'DTE3', 'DTE4', 'Status'])
        filter_df1.loc[len(filter_df1), 'Strategy'] = sanitized_strategy_name
        row_index = filter_df1.index[filter_df1['Strategy'] == sanitized_strategy_name].tolist()[0]
        filter_df1.loc[row_index, 'Parameters'] = target_list
        filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 0
        filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Start_Date'] = start_date
        filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'End_Date'] = end_date
        
        trade_sheet = trade_sheet[trade_sheet['Date'] > '2021-06-03']

        # Go through each dte for the current combo to check if it's profitable
        # for dte in dte_list:
            
        #     trade_sheet_temp = trade_sheet[trade_sheet['DaysToExpiry'] == dte]
        if not trade_sheet.empty:
            trade_sheet.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
                # if trade_sheet_temp['Premium'].sum() > 0:
                #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 1
                #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'] = 1
                    
                    # trade_sheet_temp.to_csv(f'{output_folder_path}{sanitized_strategy_name}.csv', mode='a', header=(not os.path.exists(f'{output_folder_path}{sanitized_strategy_name}.csv')), index = False)
                # else:
                #     filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, f'DTE{dte}'] = 0

        # Store the combo and it's dte which is profitable in filter_df file
        # if filter_df1.loc[filter_df1['Strategy'] == sanitized_strategy_name, 'Status'].iloc[0] == 1:
            
        existing_csv_file = rf"{filter_df_path}/filter_df{counter}.csv"
        if os.path.isfile(existing_csv_file):
            filter_df1.to_csv(existing_csv_file, index=False, mode='a', header=False)
        else:
            filter_df1.to_csv(existing_csv_file, index=False)
            
        return sanitized_strategy_name + '_' + str(start_date) + '_' + str(end_date)



def resample_data(data, TIME_FRAME):
    """
    Resamples financial time series data to a specified time frame and aggregates OHLC values.
    Parameters:
        data (pd.DataFrame): Input DataFrame with a DateTimeIndex and columns ['Open', 'High', 'Low', 'Close', 'ExpiryDate'].
        TIME_FRAME (str): Pandas-compatible resampling frequency string (e.g., '5T' for 5 minutes, '1H' for 1 hour).
    Returns:
        pd.DataFrame: Resampled DataFrame with aggregated OHLC values, 'ExpiryDate', and additional 'Date' and 'Time' columns.
    """
    
    resampled_data = data.resample(TIME_FRAME).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first'}).dropna()
    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    
    return resampled_data

########################################### INPUTS #####################################################

# Basic configuration for the strategy
superset = 'plain_vanilla'              # Strategy category/folder name
stock = 'NIFTY'                         # Stock/index being tested
option_type = 'ND'                      # Option category (e.g., 'ND' could mean non-directional)

# Set roundoff step, brokerage, and lot size based on selected stock
roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' or stock == 'SENSEX' else None)
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'BANKNIFTY' or stock == 'SENSEX' else None)
LOT_SIZE = 25 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else 10)

# Define folder structure based on parameters
root_path = rf"/home/newberry3/user/STARTEGY_NAME/{superset}/{stock}/{option_type}/"
filter_df_path = rf"{root_path}/Filter_Sheets/"                   # Folder for filtered parameter files
expiry_file_path = rf"/home/newberry3/user/Common_Files/{stock} market dates.xlsx"   # Excel containing expiry mapping
txt_file_path = rf'{root_path}/new_done.txt'                     # File to track completed parameters
output_folder_path = rf'{root_path}/Trade_Sheets/'               # Folder for final trade sheets

# Set option data path depending on stock type
if stock == 'NIFTY':
    option_data_path = rf"/home/newberry3/user/Data/NIFTY/folder/folder/"
elif stock == 'BANKNIFTY':
    option_data_path = rf"/home/newberry3/user/Data/BANKNIFTY/folder/"
elif stock == 'FINNIFTY':
    option_data_path = rf"/home/newberry3/user/Data/FINNIFTY/folder/"
elif stock == 'SENSEX':
    option_data_path = rf"/home/newberry3/user/Data/SENSEX/folder/"

# Ensure necessary folders and tracking file exist
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None

# Define the backtesting date ranges
date_ranges = [ 
    ('2024-06-01', '2024-10-30'),
    # ('2024-02-01', '2024-05-31'),
    # ('2023-10-01', '2024-01-31'),
    # ('2023-06-01', '2023-09-30'),
    # ('2023-02-01', '2023-05-31'),
    # ('2022-10-01', '2023-01-31'),
    # ('2022-06-01', '2022-09-30'),
    # ('2022-01-01', '2022-05-31'), 
    # ('2021-06-01', '2021-12-31')
]

# Strategy parameters
candle_time_frame = ['5T']          # 5-minute candle frequency

parameters = []                     # To store all parameter combinations

# Function that handles trade logic for each parameter set
def parameter_process(parameter, mapped_days, option_data, df, start_date, end_date, counter, output_folder_path):
    TIME_FRAME, STRIKE, ENTRY, EXIT = parameter
    resampled_df = resample_data(df, TIME_FRAME)         # Resample index data to desired timeframe
    resampled = resampled_df.dropna()                    # Drop missing values
    return trade_sheet_creator(mapped_days, option_data, df, start_date, end_date,
                               counter, output_folder_path, resampled, TIME_FRAME, STRIKE, ENTRY, EXIT)

# Adjust strike step for BANKNIFTY or SENSEX (e.g., step size is 2x)
if stock == 'BANKNIFTY' or stock == 'SENSEX':
    strikes = [x * 2 for x in strikes]

# Main execution block
if __name__ == "__main__":
    
    counter = 0
    start_date_idx = date_ranges[-1][0]     # Start from earliest range
    end_date_idx = date_ranges[0][-1]       # End at latest range

    # Read expiry date file and filter between the start/end date
    mapped_days = pd.read_excel(expiry_file_path)
    mapped_days = mapped_days[(mapped_days['Date'] >= start_date_idx) & (mapped_days['Date'] <= end_date_idx)]
    mapped_days = mapped_days.rename(columns={'WeeklyDaysToExpiry' : 'DaysToExpiry'})  # Rename for uniformity

    # Pull index price data for the backtesting window
    df = pull_index_data(start_date_idx, end_date_idx, stock)
    resampled_df_main = resample_data(df, '5T')          # Resample entire index data once

    # Iterate over each date range for backtest
    for start_date, end_date in date_ranges: 
        counter += 1
        print(start_date, end_date, counter)

        start_date_object = pd.to_datetime(start_date)
        end_date_object = pd.to_datetime(end_date)
        new_end_date_object = end_date_object + timedelta(days=30)     # To capture expiry after range
        new_end_date = new_end_date_object.strftime('%Y-%m-%d')        # Convert to string

        # Pull relevant options data for the extended window
        option_data = pull_options_data_d(start_date, new_end_date, option_data_path, stock)

        parameters = []        # Reset parameters list

        if counter == 1:
            filter_df = pd.DataFrame()      # For the first run, no filters
        elif counter > 1:
            # For subsequent runs, load prior filter data
            filter_file = f"{filter_df_path}/filter_df{counter-1}.csv"
            if not os.path.exists(filter_file):
                print(f"File filter_df{counter-1}.csv does not exist. Stopping the code.")
                sys.exit()
            else:
                filter_df = pd.read_csv(filter_file)
                filter_df = filter_df.drop_duplicates()

        # Read previous run parameters
        if counter != 1:
            parameters = filter_df['Parameters'].to_list()
            parameters = [ast.literal_eval(item.replace("'", "\"")) for item in parameters]

        # For first run, generate all parameter combinations
        elif counter == 1:
            for TIME_FRAME in candle_time_frame:
                for STRIKE in strikes:
                    for ENTRY in entries:
                        for EXIT in exits:
                            if ENTRY < EXIT:
                                parameters.append([TIME_FRAME, STRIKE, ENTRY, EXIT])

        # Filter out parameters that are already processed (tracked in txt file)
        print('Total parameters :', len(parameters))
        file_path = txt_file_path
        with open(file_path, 'r') as file:
            existing_values = [line.strip() for line in file]

        print('Existing files :', len(existing_values))
        parameters = [value for value in parameters if
                      (stock + '_candle_' + str(value[0]) + '_strike_' + str(value[2]) +
                       '_entry_' + str(value[3]).replace(':', ',') +
                       '_exit_' + '_' + start_date + '_' + end_date) not in existing_values]

        print('Parameters to run :', len(parameters))

        # Log parameters to run
        for value in parameters:
            string = (stock + '_candle_' + str(value[0]) + '_strike_' + str(value[2]) +
                      '_entry_' + str(value[3]).replace(':', ',') +
                      '_exit_' + '_' + start_date + '_' + end_date)
            print(string)
        
        # Begin multiprocessing-based execution
        start_time = time.time()
        num_processes = 12
        print('No. of processes :', num_processes)

        # Create a partial function for multiprocessing
        partial_process = partial(parameter_process, mapped_days=mapped_days, option_data=option_data,
                                  df=df, start_date=start_date, end_date=end_date,
                                  counter=counter, output_folder_path=output_folder_path)

        # Start multiprocessing pool
        with multiprocessing.Pool(processes=num_processes) as pool:
            with tqdm(total=len(parameters), desc='Processing', unit='Iteration') as pbar:
                def update_progress(combinations):
                    with open(txt_file_path, 'a') as fp:
                        fp.write(str(combinations) + '\n')   # Log completed parameter
                    pbar.update()

                arg_tuples = [tuple(parameter) for parameter in parameters]  # Convert list to tuples for multiprocessing
                
                for result in pool.imap_unordered(partial_process, arg_tuples):
                    update_progress(result)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Time taken to get Initial Tradesheets:', elapsed_time)

# Final print once all processing is complete
print('Finished at :', time.time())