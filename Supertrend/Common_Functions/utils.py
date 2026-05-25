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
from scipy.optimize import brentq
from scipy.stats import norm

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

def time_to_expiry_years(now_ts, expiry_date):
    """
    now_ts: pd.Timestamp (bar time)
    expiry_date: pd.Timestamp or date (expiry day)
    Market hours: 09:15–15:30 IST; returns fractional years T.
    """
    now_ts = pd.Timestamp(now_ts)
    expiry_date = pd.Timestamp(expiry_date).normalize()

    market_open  = _time(9, 15)
    market_close = _time(15, 30)

    today = now_ts.normalize()
    days_left = (expiry_date - today).days
    if days_left <= 0:
        days_left += 1

    total = (market_close.hour*60 + market_close.minute) - (market_open.hour*60 + market_open.minute)
    mins_since_open = (now_ts.hour*60 + now_ts.minute) - (market_open.hour*60 + market_open.minute)

    if mins_since_open < 0:
        frac_days = days_left
    elif mins_since_open >= total:
        frac_days = max(0, days_left - 1)
    else:
        frac_days = days_left - mins_since_open/total

    return round(frac_days/365.0, 6)



def postgresql_query(input_query, input_tuples = None):
    try:
        connection = psycopg2.connect(
            host = "00.000.00.00",
            port = 25060,
            database = "db",
            user = "user",
            password = "password",
            sslmode = "require"
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
    Resamples time series data to a specified time frame and aggregates OHLCV and expiry-related columns.
    Parameters:
        data (pandas.DataFrame): Input DataFrame with a DateTimeIndex and columns including 'Open', 'High', 'Low', 'Close', and 'ExpiryDate'.
        TIME_FRAME (str): Pandas-compatible resampling frequency string (e.g., '5T' for 5 minutes).
    Returns:
        pandas.DataFrame: Resampled DataFrame with aggregated columns and additional 'Date' and 'Time' columns extracted from the index.
    """
    
    # resampled_data = data.resample(TIME_FRAME, offset='09:15:00').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first', 'MonthlyDaysToExpiry' : 'first', 'NextMonthlyExpiry' : 'first'}).dropna()
    resampled_data = data.resample(TIME_FRAME, offset='09:15:00').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'ExpiryDate': 'first'}).dropna()

    resampled_data['Date'] = resampled_data.index.date
    resampled_data['Time'] = resampled_data.index.time
    
    return resampled_data

def resample_data_options(df, TIME_FRAME):
    """
    Resamples a DataFrame of options data by a specified time frame, grouping by 'StrikePrice', 'Type', and 'ExpiryDate'.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing options data. Must have columns 'StrikePrice', 'Type', 'ExpiryDate', and a DateTime index.
        TIME_FRAME (str): The resampling frequency string (e.g., '5T' for 5 minutes, '1H' for 1 hour).

    Returns:
        pd.DataFrame: A DataFrame with the resampled data, where for each group the first entry in each resampled interval is retained.

    Notes:
        - The input DataFrame must have a DateTimeIndex for resampling to work correctly.
        - The function concatenates all resampled groups into a single DataFrame.
    """
    resampled_dfs = []
    for strike, group in df.groupby(['StrikePrice', 'Type', 'ExpiryDate']):
        resampled_group = group.resample(TIME_FRAME).first()  # Resample and take the first entry for each interval
        resampled_dfs.append(resampled_group)

    # Concatenate all the resampled groups back into a single DataFrame
    resampled_df = pd.concat(resampled_dfs)
    return resampled_df    

def nearest_multiple(x, n):
    """
    Returns the nearest multiple of `n` to the given number `x`.

    If `x` is exactly halfway between two multiples of `n`, the function rounds up to the higher multiple.

    Args:
        x (int or float): The number to find the nearest multiple for.
        n (int or float): The multiple to use.

    Returns:
        int: The nearest multiple of `n` to `x`.
    """
    remainder = x%n
    if remainder < n/2:
        nearest = x - remainder
    else:
        nearest = x - remainder + n
    return int(nearest)



def round_to_next_5_minutes(time_str, TIME_FRAME):
    """
    Rounds a given datetime object up to the next multiple of a specified minute interval.

    Args:
        time_str (datetime): The datetime object to be rounded.
        TIME_FRAME (str): The minute interval as a string ending with a character (e.g., '5m', '15m').
                          Only the numeric part is used to determine the interval in minutes.

    Returns:
        datetime: The rounded datetime object, adjusted to the next interval.

    Example:
        >>> round_to_next_5_minutes(datetime(2023, 1, 1, 12, 3, 0), '5m')
        datetime.datetime(2023, 1, 1, 12, 10, 0)
    """

    TIME_FRAME = int(TIME_FRAME[:-1])

    # Parse the input time string to a datetime object
    time_str = time_str.strftime('%Y-%m-%d %H:%M:%S')

    time_obj = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

    # Calculate the minutes to the next 5-minute multiple
    minutes_to_next_5 = (TIME_FRAME - time_obj.minute % TIME_FRAME) % TIME_FRAME

    # Round the time to the next 5-minute multiple
    rounded_time = time_obj + timedelta(minutes=minutes_to_next_5) + timedelta(minutes=TIME_FRAME)

    # Format the rounded time as a string
    #rounded_time_str = rounded_time.strftime('%Y-%m-%d %H:%M:%S')

    return rounded_time


def get_open_range(data, OPEN_RANGE):
    """
    Calculates the open range (high and low) for a given time interval from market data.
    Parameters:
        data (pd.DataFrame): A DataFrame containing market data with a DateTimeIndex and columns 'Open', 'High', 'Low', and 'Close'.
        OPEN_RANGE (str or pd.Timedelta): The time interval (e.g., '15min') to calculate the open range from the start of the data.
    Returns:
        tuple: A tuple (Lower_TH, Upper_TH) where:
            Lower_TH (float): The lowest 'Low' price within the open range interval.
            Upper_TH (float): The highest 'High' price within the open range interval.
    """
    
    open_range_close_time = data.index[0] + pd.to_timedelta(OPEN_RANGE)
    
    candle_open = data['Open'].iloc[0]
    candle_high = data[data.index < open_range_close_time]['High'].max()
    candle_low = data[data.index < open_range_close_time]['Low'].min()
    candle_close = data[data.index < open_range_close_time]['Close'].iloc[-1]
    
    Upper_TH = candle_high
    Lower_TH = candle_low
    
    return Lower_TH, Upper_TH

def check_crossover(row, Range, Lower_TH, Upper_TH, WICK_THRESHOLD):
    """
    Determines if a candlestick row crosses a specified threshold (upper or lower) with optional wick filtering.
    Args:
        row (dict or pandas.Series): A dictionary or Series containing candlestick data with keys 'Open', 'High', 'Low', 'Close'.
        Range (str): Specifies which threshold to check for crossover. Must be either 'Upper' (bullish) or 'Lower' (bearish).
        Lower_TH (float): The lower threshold value for bearish crossover.
        Upper_TH (float): The upper threshold value for bullish crossover.
        WICK_THRESHOLD (float or str): Multiplier to limit the wick size relative to the candle body. If 'NA', wick filtering is ignored.
    Returns:
        bool: True if a crossover occurs according to the specified conditions, False otherwise.
    """
    
    Open = row['Open']
    High = row['High']
    Low = row['Low']
    Close = row['Close']
    
    # Bullish
    if (Range == 'Upper') and (Open < Close) and (Open <= Upper_TH) and (Close > Upper_TH)\
    and ((WICK_THRESHOLD == 'NA') or ((High - Close) <= (Close - Open)*WICK_THRESHOLD)):
        return True
    
    # Bearish
    elif (Range == 'Lower') and (Open > Close) and (Open >= Lower_TH) and (Close < Lower_TH)\
    and ((WICK_THRESHOLD == 'NA') or ((Close - Low) <= (Open - Close)*WICK_THRESHOLD)):
        return True
    
    return False

def get_target_stoploss(row, Type, TARGET_TH, STOPLOSS_TH, Band=None):
    """
    Calculate the target and stoploss values for a given row of OHLC data based on the trade type and threshold parameters.
    Parameters:
        row (pd.Series or dict): A row containing at least 'Open', 'High', 'Low', and 'Close' prices.
        Type (str): The trade type, either 'CE' (Call/Long) or 'PE' (Put/Short).
        TARGET_TH (tuple): Target threshold, where the first element is the method ('Bips', 'Band', 'candleOC', 'candleHL') and the second is the value.
        STOPLOSS_TH (tuple): Stoploss threshold, where the first element is the method ('Bips', 'Band', 'Points', 'CandleLow', 'MotherLow', 'InsideLow', 'InsideBips', 'CandleHigh', 'MotherHigh', 'InsideHigh', 'InsideBips') and the second is the value.
        Band (float, optional): The band value used in 'Band' calculations. Default is None.
    Returns:
        tuple: (target, stoploss) values calculated based on the provided thresholds and trade type.
    Notes:
        - The function supports different calculation methods for both target and stoploss, depending on the provided threshold method.
        - For 'Band' methods, the Band parameter must be provided.
        - The function assumes the input row contains the required OHLC keys.
    """
    
    Open = row['Open']
    High = row['High']
    Low = row['Low']
    Close = row['Close']
    
    # Stoploss Calculation
    # CE
    if Type == 'CE':

        if STOPLOSS_TH[0] == 'Bips':
            stoploss = Low*(1 - 0.0001*STOPLOSS_TH[1])
        elif STOPLOSS_TH[0] == 'Band':
            stoploss = Low - Band*STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'Points':
            stoploss = Low - STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'CandleLow':
            stoploss = Low
        elif STOPLOSS_TH[0] == 'MotherLow':
            stoploss = Low
        elif STOPLOSS_TH[0] == 'InsideLow':
            stoploss = Low
        elif STOPLOSS_TH[0] == 'InsideBips':
            stoploss = Low*(1 - 0.0001*STOPLOSS_TH[1])

    # PE
    elif Type == 'PE':

        if STOPLOSS_TH[0] == 'Bips':
            stoploss = High*(1 + 0.0001*STOPLOSS_TH[1])
        elif STOPLOSS_TH[0] == 'Band':
            stoploss = High + Band*STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'Points':
            stoploss = High + STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'CandleHigh':
            stoploss = High 
        elif STOPLOSS_TH[0] == 'MotherHigh':
            stoploss = High
        elif STOPLOSS_TH[0] == 'InsideHigh':
            stoploss = High
        elif STOPLOSS_TH[0] == 'InsideBips':
            stoploss = High*(1 + 0.0001*STOPLOSS_TH[1])   
    
    # Target Calculation
    # CE
    if Type == 'CE':
        
        if TARGET_TH[0] == 'Bips':
            target = Close*(1 + 0.0001*TARGET_TH[1])
        elif TARGET_TH[0] == 'Band':
            target = Close + Band*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleOC':
            target = Close + (Close - Open)*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleHL':
            target = Close + (High - Low)*TARGET_TH[1]
        
    
    #PE
    elif Type == 'PE':
        
        if TARGET_TH[0] == 'Bips':
            target = Close*(1 - 0.0001*TARGET_TH[1])
        elif TARGET_TH[0] == 'Band':
            target = Close - Band*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleOC':
            target = Close - (Open - Close)*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleHL':
            target = Close - (High - Low)*TARGET_TH[1]
    
    return target, stoploss

def get_target(row, Type, TARGET_TH, Band=None):
    """
    Calculates the target price based on the provided row data, option type, target threshold, and optional band value.
    Parameters:
        row (dict or pandas.Series): A dictionary or Series containing at least the keys 'Open', 'High', 'Low', and 'Close'.
        Type (str): The option type, either 'CE' (Call European) or 'PE' (Put European).
        TARGET_TH (tuple): A tuple specifying the target calculation method and its parameter.
            - If TARGET_TH[0] == 'Bips': TARGET_TH[1] is the number of basis points.
            - If TARGET_TH[0] == 'Band': TARGET_TH[1] is a multiplier for the band.
            - If TARGET_TH[0] == 'candleOC': TARGET_TH[1] is a multiplier for the open-close difference.
            - If TARGET_TH[0] == 'candleHL': TARGET_TH[1] is a multiplier for the high-low difference.
        Band (float, optional): The band value used when TARGET_TH[0] == 'Band'. Default is None.
    Returns:
        float: The calculated target price based on the specified method and parameters.
    Raises:
        KeyError: If required keys are missing from the row.
        ValueError: If Type or TARGET_TH[0] is not recognized.
    """
    
    Open = row['Open']
    High = row['High']
    Low = row['Low']
    Close = row['Close']
    
    # Target Calculation
    # CE
    if Type == 'CE':
        
        if TARGET_TH[0] == 'Bips':
            target = Close*(1 + 0.0001*TARGET_TH[1])
        elif TARGET_TH[0] == 'Band':
            target = Close + Band*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleOC':
            target = Close + (Close - Open)*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleHL':
            target = Close + (High - Low)*TARGET_TH[1]
        
    
    #PE
    elif Type == 'PE':
        
        if TARGET_TH[0] == 'Bips':
            target = Close*(1 - 0.0001*TARGET_TH[1])
        elif TARGET_TH[0] == 'Band':
            target = Close - Band*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleOC':
            target = Close - (Open - Close)*TARGET_TH[1]
        elif TARGET_TH[0] == 'candleHL':
            target = Close - (High - Low)*TARGET_TH[1]
    
    return target

def get_stoploss(row, Type, STOPLOSS_TH, Band=None):
    """
    Calculate the stoploss value for a given row of OHLC data based on the specified type and stoploss threshold.
    Parameters:
        row (dict or pandas.Series): A dictionary or Series containing at least the keys 'Open', 'High', 'Low', 'Close'.
        Type (str): The option type, either 'CE' (Call European) or 'PE' (Put European).
        STOPLOSS_TH (tuple): A tuple where the first element is a string indicating the stoploss method 
            ('Bips', 'Band', 'Points', 'CandleLow', 'MotherLow', 'InsideLow', 'RangeLow', 'InsideBips', 
            'CandleHigh', 'MotherHigh', 'InsideHigh', 'RangeHigh'), and the second element is a numeric threshold.
        Band (float, optional): The band value used when the stoploss method is 'Band'. Default is None.
    Returns:
        float: The calculated stoploss value based on the provided parameters.
    Raises:
        KeyError: If required keys are missing from the row.
        ValueError: If Type or STOPLOSS_TH[0] is not recognized.
    """
    
    Open = row['Open']
    High = row['High']
    Low = row['Low']
    Close = row['Close']
    
    # Stoploss Calculation
    # CE
    if Type == 'CE':

        if STOPLOSS_TH[0] == 'Bips':
            stoploss = Low*(1 - 0.0001*STOPLOSS_TH[1])
        elif STOPLOSS_TH[0] == 'Band':
            stoploss = Low - Band*STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'Points':
            stoploss = Low - STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'CandleLow':
            stoploss = Low
        elif STOPLOSS_TH[0] == 'MotherLow':
            stoploss = Low
        elif STOPLOSS_TH[0] == 'InsideLow':
            stoploss = Low
        elif STOPLOSS_TH[0] == 'RangeLow':
            stoploss = Low
        elif STOPLOSS_TH[0] == 'InsideBips':
            stoploss = Low*(1 - 0.0001*STOPLOSS_TH[1])

    # PE
    elif Type == 'PE':

        if STOPLOSS_TH[0] == 'Bips':
            stoploss = High*(1 + 0.0001*STOPLOSS_TH[1])
        elif STOPLOSS_TH[0] == 'Band':
            stoploss = High + Band*STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'Points':
            stoploss = High + STOPLOSS_TH[1]
        elif STOPLOSS_TH[0] == 'CandleHigh':
            stoploss = High 
        elif STOPLOSS_TH[0] == 'MotherHigh':
            stoploss = High
        elif STOPLOSS_TH[0] == 'InsideHigh':
            stoploss = High
        elif STOPLOSS_TH[0] == 'RangeHigh':
            stoploss = High
        elif STOPLOSS_TH[0] == 'InsideBips':
            stoploss = High*(1 + 0.0001*STOPLOSS_TH[1])   
    
    return stoploss
    
# Function to pull options data for specified date range 
def pull_options_data_d(start_date, end_date, option_data_path):
            """
            Pulls and concatenates options data from pickled files within a specified date range.
            Args:
                start_date (str): The start date in 'YYYY-MM-DD' format for filtering option data files.
                end_date (str): The end date in 'YYYY-MM-DD' format for filtering option data files.
                option_data_path (str): The directory path containing the pickled option data files.
            Returns:
                pd.DataFrame: A DataFrame containing concatenated options data filtered by the specified date range.
                              The DataFrame includes columns: 'ExpiryDate', 'StrikePrice', 'Type', 'Open', 'Ticker',
                              with the index set as 'DateTime' parsed from the 'Ticker' column.
            Side Effects:
                Prints the time taken to pull and process the options data.
            Raises:
                FileNotFoundError: If the specified option_data_path does not exist.
                KeyError: If expected columns are missing in the pickled data files.
                ValueError: If date parsing fails due to unexpected filename or ticker formats.
            """
            
            start_time = time.time()
            option_data_files = next(os.walk(option_data_path))[2]
            option_data = pd.DataFrame()

            for file in option_data_files:

                match = re.search(r'(\d+)', file).group(1)
                date1 = datetime.strptime(match[0:4] + match[4:6] + '01', "%Y%m%d").date().strftime("%Y-%m-%d")
                
                if (date1>=start_date) & (date1<=end_date):

                    temp_data = pd.read_pickle(option_data_path + file)[['ExpiryDate', 'StrikePrice', 'Type', 'Open', 'Ticker']]
                    temp_data.index = pd.to_datetime(temp_data['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
                    temp_data = temp_data.rename_axis('DateTime')
                    option_data = pd.concat([option_data, temp_data])

            option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
            option_data['Type'] = option_data['Type'].astype('category')
            
            end_time = time.time()
            print('Time taken to pull Options data :', (end_time-start_time))

            return option_data

# Function to pull index data for specified date range
def pull_index_data(start_date_idx, end_date_idx, mapped_days, stock):
    """
    Fetches and processes index data for a given stock between specified date indices.
    Parameters:
        start_date_idx (str): The start date index in 'YYYYMMDD' format.
        end_date_idx (str): The end date index in 'YYYYMMDD' format.
        mapped_days (pd.DataFrame): DataFrame containing mapped trading days with a 'Date' column.
        stock (str): The stock symbol for which to fetch index data.
    Returns:
        pd.DataFrame: A DataFrame containing the merged and processed index data, indexed by datetime.
    Notes:
        - The function queries a PostgreSQL database for OHLC data within the specified date range and trading hours (09:15 to 15:29).
        - The resulting DataFrame is merged with the provided mapped_days DataFrame on the 'Date' column.
        - The index of the returned DataFrame is set to the combined date and time extracted from the 'Ticker' column.
        - Duplicate rows are dropped and the DataFrame is sorted by the datetime index.
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

def compare_month_and_year(date1, date2, file, stock):
    """
    Compares whether the month and year extracted from a filename (representing a date) 
    falls within the range defined by two given dates (inclusive).
    Args:
        date1 (str): The start date in the format "%Y-%m-%d".
        date2 (str): The end date in the format "%Y-%m-%d".
        file (str): The filename containing the date information, expected to start with the stock name followed by a date in "YYYYMM" format.
        stock (str): The stock name prefix to be removed from the filename to extract the date.
    Returns:
        str or None: Returns the filename if the extracted date is within the range [date1, date2] (by month and year), otherwise returns None.
    """

    # Create a new datetime object with day set to 01
    date1 = datetime.strptime(date1, "%Y-%m-%d").replace(day=1)
    #print(date2)
    date2 = datetime.strptime(date2, "%Y-%m-%d").replace(day=1)
    #print(date2)

    date3 = datetime.strptime(file.replace(stock + '_', '')[0:4] + file.replace(stock + '_', '')[4:6] + '01', "%Y%m%d").date().strftime("%Y-%m-%d")
    date3 = datetime.strptime(date3, '%Y-%m-%d')
    
    if (date1 <= date3) & (date3 <= date2):
        return file
    else:
        return None
        
# Function to calculate trailing stoploss
def TSL(df, option_data, next_time_period, ATM, Action, Type, position, target, stoploss, LAST_ENTRY, I_T, L_P, INC_T, INC_S):
    """
    Trailing Stop Loss (TSL) logic for intraday options trading.
    This function simulates a trailing stop loss strategy for an options position, updating targets and stop losses as the price moves in favor of the trade. It checks for various exit conditions such as hitting the initial stop loss, reaching the target, or hitting a trailing stop loss, and returns the entry and exit details.
    Parameters:
        df (pd.DataFrame): Index data with datetime index and 'Close' prices.
        option_data (pd.DataFrame): Option data with datetime index, 'StrikePrice', 'Type', and 'Open' columns.
        next_time_period (pd.Timestamp): The datetime of the entry.
        ATM (float or int): At-the-money strike price for the option.
        Action (str): 'long' for long position.
        Type (str): Option type, 'CE' (Call) or 'PE' (Put).
        position (int): Current position status (1 for open, 0 for closed).
        target (float): Initial target price for the underlying index.
        stoploss (float): Initial stop loss price for the underlying index.
        LAST_ENTRY (str): Last allowed entry time in 'HH:MM' format.
        I_T (float): Initial target increment for the option premium.
        L_P (float): Loss protection value for trailing stop loss.
        INC_T (float): Increment to increase the target after each hit.
        INC_S (float): Increment to increase the stop loss after each target hit.
    Returns:
        list: [entry_price, exit_price, exit_time_period, exit_type, position]
            entry_price (float): The price at which the position was entered.
            exit_price (float): The price at which the position was exited.
            exit_time_period (pd.Timestamp): The datetime of the exit.
            exit_type (str): The reason for exit ('Universal', 'Initial Stoploss Exit', 'Initial Target Exit', 'TSL').
            position (int): Final position status (0 for closed).
    """

    start_time = next_time_period
    end_time = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')

    # Filter option data for the date, time, strike and type of the entry position    
    intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]

    # print(start_time, ATM, Type)
    # print(intraday_data)
    # print(option_data[option_data['ExpiryDate'] == '2024-01-04'])

    intraday_data = intraday_data[(intraday_data['StrikePrice']==ATM) & (intraday_data['Type']==Type)]    
    intraday_data = intraday_data.sort_index()
    
    entry_price = intraday_data.iloc[0]['Open']
    intraday_data = intraday_data[intraday_data.index > next_time_period]

    # Filter index data for the date, time, strike and type of the entry position
    temp_df  = df[(df.index > next_time_period) & (df.index < end_time)]
    
    if Action=='long':    
        target_pt = entry_price + I_T
        initial_stoploss_pt =  stoploss
        initial_target_pt = target
    target_hit = 0

    # crosses_threshold1 - when index goes below the initial stoploss pt (we take exit here)
    # crosses_threshold2 - when option premium goes above the profit pt
    # crosses_threshold3 - when option premium goes below the stoploss pt (we take exit here)
    # crooses_threshold4 - when index goes above the initial target pt (we take exit here)
    while True:
        
        # If the target pt is not hit once yet check for intitial stoploss pt hit
        if target_hit==0:
            if Type=='CE':
                crosses_threshold1 = (temp_df['Close'] < initial_stoploss_pt)
            elif Type=='PE':
                crosses_threshold1 = (temp_df['Close'] > initial_stoploss_pt)
        else:
            crosses_threshold1 = pd.Series()
         
        crosses_threshold2 = (intraday_data['Open'] > target_pt)

        if Type=='CE':
            crosses_threshold4 = (temp_df['Close'] > initial_target_pt)
        elif Type=='PE':
            crosses_threshold4 = (temp_df['Close'] < initial_target_pt)

        # print(next_time_period)
        if (crosses_threshold4.empty) & (temp_df.empty):
            crosses_threshold4 = pd.Series([False], index=[end_time])
            crosses_threshold4.index.name = 'DateTime'
        
        # If the target pt is hit already check for stoplos pt hit
        if target_hit:
            crosses_threshold3 = (intraday_data['Open'] < stoploss_pt)
        else:
            crosses_threshold3 = pd.Series()
        
        # When target pt is not hit for the first time yet and initial stoploss pt is also not hit throughout day
        if (target_hit==0) & (not crosses_threshold1.any()) & (not crosses_threshold2.any()) & (not crosses_threshold4.any()):
            exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
            exit_price = intraday_data.at[exit_time_period, 'Open']
            exit_type = 'Universal'
            position = 0
            break

        # When the target pt is not hit for the first time yet
        if (target_hit==0): 

            # When the initial stoploss pt got hit before target pt
            if crosses_threshold1.any() & crosses_threshold2.any() &  crosses_threshold4.any() & ((crosses_threshold1.idxmax()) < crosses_threshold2.idxmax()) \
                                                                                               & ((crosses_threshold1.idxmax()) < crosses_threshold4.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & crosses_threshold2.any() &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) < crosses_threshold1.idxmax()) \
                                                                                               & ((crosses_threshold4.idxmax()) <= crosses_threshold2.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break
            
            # When the target pt got hit before the initial stoploss pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold1.idxmax()) \
                                                                                                & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax())    :
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = target_pt - L_P + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
            
            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & (not crosses_threshold4.any()) & ((crosses_threshold1.idxmax()) < crosses_threshold2.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break


            # When the target pt got hit before the initial stoploss pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & (not crosses_threshold4.any()) & (crosses_threshold1.idxmax() > crosses_threshold2.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = target_pt - L_P + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 3 : ', temp_df)

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & (not crosses_threshold2.any()) &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) < crosses_threshold1.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & (not crosses_threshold2.any()) &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) > crosses_threshold1.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif (not crosses_threshold1.any()) & crosses_threshold2.any() & crosses_threshold4.any() & ((crosses_threshold2.idxmax()) >= crosses_threshold4.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # WHen the target pt got hit before the initial stoploss pt
            elif (not crosses_threshold1.any()) & crosses_threshold2.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = target_pt - L_P + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 4 : ', temp_df)


            # When the initial stoploss pt got hit but target pt is not hit
            elif crosses_threshold1.any():
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break
            
            # When the initial stoploss pt got hit but target pt is not hit
            elif crosses_threshold4.any():
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break
            
            # When only target pt got hit
            elif crosses_threshold2.any():
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = target_pt - L_P + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 5 : ', temp_df)
                # print('Crosses_Threshold2 :', crosses_threshold2.idxmax())
                # print('Intraday_Data :', intraday_data)


        # When the target pt is already hit once
        elif target_hit: 
            
            #print('Crosses_Threshold4 : ', crosses_threshold4.any())
            # When target pt is already hit once and if target pt and stoploss pt none of them got hit throughout day take an exit
            # try:
            if (not crosses_threshold2.any()) & (not crosses_threshold3.any()) & (not crosses_threshold4.any()):
                
                exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                exit_price = intraday_data.at[exit_time_period, 'Open']
                exit_type = 'Universal'
                position = 0
                break 
            
            # When target pt is already hit once and stoploss pt got hit before target pt 
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & \
                (crosses_threshold3.idxmax() < crosses_threshold2.idxmax()) &  (crosses_threshold3.idxmax() < crosses_threshold4.idxmax()) :
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break


            # When target pt is already hit once and stoploss pt got hit before target pt 
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & (crosses_threshold4.idxmax() <= crosses_threshold2.idxmax())\
                                                                                                &  (crosses_threshold4.idxmax() < crosses_threshold3.idxmax())   :
                #print(ATM, exit_time_period)
                #print(intraday_data)
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            
            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold3.idxmax()) \
                                                                                                &  (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 6 : ', temp_df)


            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif (not crosses_threshold2.any()) & crosses_threshold3.any() & (crosses_threshold4.any()) & (crosses_threshold3.idxmax() < crosses_threshold4.idxmax()):
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break

            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif (not crosses_threshold2.any()) & crosses_threshold3.any() & (crosses_threshold4.any()) & (crosses_threshold4.idxmax() < crosses_threshold3.idxmax()):
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold2.any() & crosses_threshold3.any() & (not crosses_threshold4.any()) & (crosses_threshold3.idxmax() < crosses_threshold2.idxmax()):
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break

            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & crosses_threshold3.any() & (not crosses_threshold4.any()) & (crosses_threshold2.idxmax() < crosses_threshold3.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 7 : ', temp_df)



            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold2.any() & (not crosses_threshold3.any()) & crosses_threshold4.any() & (crosses_threshold4.idxmax() <= crosses_threshold2.idxmax()):
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break  

            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & (not crosses_threshold3.any()) & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 8 : ', temp_df)


            # When target pt is already hit once and only target pt got hit 
            elif crosses_threshold2.any():
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 9 : ', temp_df)

            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold3.any():
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break
            
            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold4.any():
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # except:
            #     print('Crosses_Threshold4 ', crosses_threshold4)
            #     print('Temp df ', temp_df)
            #     print('Target List : ', target_list)
            #     exit()

        if position == 0:
            break

    return [entry_price, exit_price, exit_time_period, exit_type, position]


# Function to calculate trailing stoploss, initial target and initial stoploss all based on premium prices for directional strategies
def TSL_PREMIUM(df, option_data, next_time_period, ATM, Action, Type, position, target, stoploss, LAST_ENTRY, I_T, L_P, INC_T, INC_S):
    """
    TSL_PREMIUM simulates a trailing stop loss (TSL) and target-based exit strategy for option premium trading.
    Parameters:
        df (pd.DataFrame): DataFrame containing index data with datetime index.
        option_data (pd.DataFrame): DataFrame containing option data with datetime index.
        next_time_period (datetime): The entry time for the trade.
        ATM (float or int): At-the-money strike price for the option.
        Action (str): Trade direction, e.g., 'long'.
        Type (str): Option type, e.g., 'CE' (Call) or 'PE' (Put).
        position (int): Current position status (1 for open, 0 for closed).
        target (float): Initial target in premium points.
        stoploss (float): Initial stoploss in premium points.
        LAST_ENTRY (str): Last allowed entry time in 'HH:MM' format.
        I_T (float): Initial profit threshold for trailing stop logic.
        L_P (float): Not used in this function.
        INC_T (float): Increment in target for each trailing step.
        INC_S (float): Increment in stoploss for each trailing step.
    Returns:
        list: [
            entry_price (float): Option premium at entry,
            exit_price (float): Option premium at exit,
            exit_time_period (datetime): Time of exit,
            exit_type (str): Reason/type of exit (e.g., 'Universal', 'TSL', 'Initial Stoploss Exit', 'Initial Target Exit'),
            position (int): Final position status (0 for closed),
            initial_target_pt (float): Initial target premium,
            initial_stoploss_pt (float): Initial stoploss premium
        ]
    Notes:
        - The function iteratively checks for target and stoploss hits, updating trailing stop and target levels as needed.
        - Handles multiple exit scenarios including universal exit at end of day, initial stoploss/target, and trailing stop loss.
        - Assumes input DataFrames are indexed by datetime and contain required columns ('Open', 'StrikePrice', 'Type').
    """

    start_time = next_time_period
    print("start_time",start_time)
    # end_time = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
    end_time = LAST_ENTRY
    
    # Filter option data for the date, time, strike and type of the entry position    
    intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]



    print(f"Processing next_time_period: {next_time_period}")
    print(f"Intraday data length: {len(intraday_data)}")
    

    intraday_data = intraday_data[(intraday_data['StrikePrice']==ATM) & (intraday_data['Type']==Type)]   
    
    intraday_data = intraday_data.sort_index()
    # print(intraday_data) 
    
    entry_price = intraday_data.iloc[0]['Open']
    intraday_data = intraday_data[intraday_data.index > next_time_period]

    # Filter index data for the date, time, strike and type of the entry position
    temp_df  = df[(df.index > next_time_period) & (df.index < end_time)]
    
    if Action=='long':    
        target_pt = entry_price + I_T
        initial_stoploss_pt =  entry_price - stoploss
        initial_target_pt = entry_price + target
    target_hit = 0

    # crosses_threshold1 - when index goes below the initial stoploss pt (we take exit here)
    # crosses_threshold2 - when option premium goes above the profit pt
    # crosses_threshold3 - when option premium goes below the stoploss pt (we take exit here)
    # crooses_threshold4 - when index goes above the initial target pt (we take exit here)
    while True:
        
        # If the target pt is not hit once yet check for intitial stoploss pt hit
        if target_hit==0:
            # if Type=='CE':
            #     crosses_threshold1 = (temp_df['Close'] < initial_stoploss_pt)
            # elif Type=='PE':
            #     crosses_threshold1 = (temp_df['Close'] > initial_stoploss_pt)
            crosses_threshold1 = (intraday_data['Open'] < initial_stoploss_pt)

        else:
            crosses_threshold1 = pd.Series()
         
        crosses_threshold2 = (intraday_data['Open'] > target_pt)

        # if Type=='CE':
        #     crosses_threshold4 = (temp_df['Close'] > initial_target_pt)
        # elif Type=='PE':
        #     crosses_threshold4 = (temp_df['Close'] < initial_target_pt)
        crosses_threshold4 = (intraday_data['Open'] > initial_target_pt)

        # print(next_time_period)
        if (crosses_threshold4.empty) & (temp_df.empty):
            crosses_threshold4 = pd.Series([False], index=[end_time])
            crosses_threshold4.index.name = 'DateTime'
        
        # If the target pt is hit already check for stoplos pt hit
        if target_hit:
            crosses_threshold3 = (intraday_data['Open'] < stoploss_pt)
        else:
            crosses_threshold3 = pd.Series()
        
        # When target pt is not hit for the first time yet and initial stoploss pt is also not hit throughout day
        if (target_hit==0) & (not crosses_threshold1.any()) & (not crosses_threshold2.any()) & (not crosses_threshold4.any()):
            
            exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
            
            try:
                exit_price = intraday_data.at[exit_time_period, 'Open']
            except:
                next_index = intraday_data.index[intraday_data.index < exit_time_period].max()
                exit_price = intraday_data.at[next_index, 'Open']   
            exit_type = 'Universal'
            position = 0
            break

        # When the target pt is not hit for the first time yet
        if (target_hit==0): 

            # When the initial stoploss pt got hit before target pt
            if crosses_threshold1.any() & crosses_threshold2.any() &  crosses_threshold4.any() & ((crosses_threshold1.idxmax()) < crosses_threshold2.idxmax()) \
                                                                                               & ((crosses_threshold1.idxmax()) < crosses_threshold4.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & crosses_threshold2.any() &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) < crosses_threshold1.idxmax()) \
                                                                                               & ((crosses_threshold4.idxmax()) <= crosses_threshold2.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break
            
            # Wenh the target pt got hit before the initial stoploss pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold1.idxmax()) \
                                                                                                & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax())    :
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = entry_price + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                # temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
            
            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & (not crosses_threshold4.any()) & ((crosses_threshold1.idxmax()) < crosses_threshold2.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break

            # When the target pt got hit before the initial stoploss pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & (not crosses_threshold4.any()) & (crosses_threshold1.idxmax() > crosses_threshold2.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = entry_price + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                # temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 3 : ', temp_df)

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & (not crosses_threshold2.any()) &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) < crosses_threshold1.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif crosses_threshold1.any() & (not crosses_threshold2.any()) &  crosses_threshold4.any() & ((crosses_threshold4.idxmax()) > crosses_threshold1.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break

            # When the initial stoploss pt got hit before target pt
            elif (not crosses_threshold1.any()) & crosses_threshold2.any() & crosses_threshold4.any() & ((crosses_threshold2.idxmax()) >= crosses_threshold4.idxmax()):
                crosses_threshold4_index = crosses_threshold4.idxmax()

                if intraday_data[intraday_data.index > crosses_threshold4.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[exit_time_period, 'Open']
                    exit_type = 'Universal'
                    position = 0
                else:
                    try:
                        exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                    except:
                        next_index = intraday_data.index[intraday_data.index < (crosses_threshold4.idxmax() + pd.to_timedelta('1T'))].max()
                        exit_price = intraday_data.at[next_index, 'Open']   
                    exit_time_period = (crosses_threshold4.idxmax() + pd.to_timedelta('1T'))      
                    exit_type = 'Initial Target Exit'
                    position = 0
                break


            # WHen the target pt got hit before the initial stoploss pt
            elif (not crosses_threshold1.any()) & crosses_threshold2.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = entry_price + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                # temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 4 : ', temp_df)


            # When the initial stoploss pt got hit but target pt is not hit
            elif crosses_threshold1.any():
                crosses_threshold1_index = crosses_threshold1.idxmax()
                try:
                    exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                except:
                    next_index = intraday_data.index[intraday_data.index < (crosses_threshold1_index + pd.to_timedelta('1T'))].max()
                    exit_price = intraday_data.at[next_index, 'Open'] 

                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break
            
            # When the initial stoploss pt got hit but target pt is not hit
            elif crosses_threshold4.any():
                crosses_threshold4_index = crosses_threshold4.idxmax()
                exit_price = intraday_data.at[crosses_threshold4_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4_index + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break
            
            # When only target pt got hit
            elif crosses_threshold2.any():
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = entry_price + (target_mul-1) * INC_S
                target_pt = target_pt + target_mul * INC_T
                target_hit = 1

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                # temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 5 : ', temp_df)
                # print('Crosses_Threshold2 :', crosses_threshold2.idxmax())
                # print('Intraday_Data :', intraday_data)


        # When the target pt is already hit once
        elif target_hit: 
            
            #print('Crosses_Threshold4 : ', crosses_threshold4.any())
            # When target pt is already hit once and if target pt and stoploss pt none of them got hit throughout day take an exit
            # try:
            if (not crosses_threshold2.any()) & (not crosses_threshold3.any()) & (not crosses_threshold4.any()):
                
                exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                exit_price = intraday_data.at[exit_time_period, 'Open']
                exit_type = 'Universal'
                position = 0
                break 
            
            # When target pt is already hit once and stoploss pt got hit before target pt 
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & \
                (crosses_threshold3.idxmax() < crosses_threshold2.idxmax()) &  (crosses_threshold3.idxmax() < crosses_threshold4.idxmax()) :
                exit_price = intraday_data.at[crosses_threshold3.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold3.idxmax() + pd.to_timedelta('1T')
                exit_type = 'TSL'
                position = 0
                break


            # When target pt is already hit once and initial target pt got hit before target pt 
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & (crosses_threshold4.idxmax() <= crosses_threshold2.idxmax())\
                                                                                                &  (crosses_threshold4.idxmax() < crosses_threshold3.idxmax())   :
                #print(ATM, exit_time_period)
                #print(intraday_data)
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            
            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & crosses_threshold3.any() & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold3.idxmax()) \
                                                                                                &  (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                # temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 6 : ', temp_df)


            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif (not crosses_threshold2.any()) & crosses_threshold3.any() & (crosses_threshold4.any()) & (crosses_threshold3.idxmax() < crosses_threshold4.idxmax()):
                exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                exit_time_period = crosses_threshold3.idxmax()
                exit_type = 'TSL'
                position = 0
                break

            # When target pt is already hit once and initial target pt got hit but target pt is not hit
            elif (not crosses_threshold2.any()) & crosses_threshold3.any() & (crosses_threshold4.any()) & (crosses_threshold4.idxmax() < crosses_threshold3.idxmax()):
                exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                exit_type = 'Initial Target Exit'
                position = 0
                break

            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold2.any() & crosses_threshold3.any() & (not crosses_threshold4.any()) & (crosses_threshold3.idxmax() < crosses_threshold2.idxmax()):
                exit_price = intraday_data.at[crosses_threshold3.idxmax() + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold3.idxmax() + pd.to_timedelta('1T')
                exit_type = 'TSL'
                position = 0
                break

            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & crosses_threshold3.any() & (not crosses_threshold4.any()) & (crosses_threshold2.idxmax() < crosses_threshold3.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                # temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 7 : ', temp_df)



            # When target pt is already hit once and initial target pt got hit but target pt is not hit
            elif crosses_threshold2.any() & (not crosses_threshold3.any()) & crosses_threshold4.any() & (crosses_threshold4.idxmax() <= crosses_threshold2.idxmax()):
                # exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                # exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')


                if intraday_data[intraday_data.index > crosses_threshold4.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[exit_time_period, 'Open']
                    exit_type = 'Universal'
                    position = 0
                else:
                    try:
                        exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                    except:
                        next_index = intraday_data.index[intraday_data.index < (crosses_threshold4.idxmax() + pd.to_timedelta('1T'))].max()
                        exit_price = intraday_data.at[next_index, 'Open']   
                    exit_time_period = (crosses_threshold4.idxmax() + pd.to_timedelta('1T'))      
                    exit_type = 'Initial Target Exit'
                    position = 0
                break  

            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & (not crosses_threshold3.any()) & crosses_threshold4.any() & (crosses_threshold2.idxmax() < crosses_threshold4.idxmax()):
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                # temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 8 : ', temp_df)


            # When target pt is already hit once and only target pt got hit 
            elif crosses_threshold2.any():
                target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                stoploss_pt = stoploss_pt + target_mul * INC_S
                target_pt = target_pt + target_mul * INC_T
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]
                # temp_df = temp_df[temp_df.index > crosses_threshold2.idxmax()]
                # print('Temp df 9 : ', temp_df)

            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold3.any():
                
                if intraday_data[intraday_data.index > crosses_threshold3.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[exit_time_period, 'Open']
                    exit_type = 'Universal'
                else:
                    try:
                        exit_price = intraday_data.at[crosses_threshold3.idxmax() + pd.to_timedelta('1T'), 'Open']
                    except:
                        next_index = intraday_data.index[intraday_data.index < (crosses_threshold3.idxmax() + pd.to_timedelta('1T'))].max()
                        exit_price = intraday_data.at[next_index, 'Open']                
                    exit_time_period = crosses_threshold3.idxmax() + pd.to_timedelta('1T')
                    exit_type = 'TSL'
                position = 0
                break
            
            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold4.any():

                if intraday_data[intraday_data.index > crosses_threshold4.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[exit_time_period, 'Open']
                    exit_type = 'Universal'
                else:
                    try:
                        exit_price = intraday_data.at[crosses_threshold4.idxmax() + pd.to_timedelta('1T'), 'Open']
                    except:
                        next_index = intraday_data.index[intraday_data.index < (crosses_threshold4.idxmax() + pd.to_timedelta('1T'))].max()
                        exit_price = intraday_data.at[next_index, 'Open']                
                    exit_time_period = crosses_threshold4.idxmax() + pd.to_timedelta('1T')
                    exit_type = 'Initial Target Exit'
                position = 0
                break

        if position == 0:
            break

    return [entry_price, exit_price, exit_time_period, exit_type, position, initial_target_pt, initial_stoploss_pt]

# Function to calculate square off points for CE and PE legs in short straddle
def Square_Off_Func(option_data, next_time_period, CE_OTM, PE_OTM, EXIT, STOPLOSS_PT, PREMIUM_TP):
    """
    Executes the square-off logic for option trades based on intraday price movements, stoploss, and exit conditions.
    Parameters:
        option_data (pd.DataFrame): DataFrame containing option price data with a DateTime index and columns including 'StrikePrice', 'Type', and 'Open'.
        next_time_period (datetime): The starting time for the trade evaluation.
        CE_OTM (float or int): Strike price for the Call Option (CE) Out-Of-The-Money position.
        PE_OTM (float or int): Strike price for the Put Option (PE) Out-Of-The-Money position.
        EXIT (str): Time (in 'HH:MM' format) to force exit the position if stoploss/target is not hit.
        STOPLOSS_PT (float): Stoploss percentage (e.g., 0.25 for 25%).
        PREMIUM_TP (str): Determines which premium to use for target calculation ('MIN' or other).
    Returns:
        list: [
            CE_OTM_entry_price (float): Entry price for CE OTM,
            CE_OTM_exit_price (float): Exit price for CE OTM,
            ce_exit_time_period (datetime): Exit time for CE OTM,
            PE_OTM_entry_price (float): Entry price for PE OTM,
            PE_OTM_exit_price (float): Exit price for PE OTM,
            pe_exit_time_period (datetime): Exit time for PE OTM,
            ce_stoploss (int): 1 if CE OTM exited by stoploss/target, 0 if exited by time,
            pe_stoploss (int): 1 if PE OTM exited by stoploss/target, 0 if exited by time,
            CE_OTM_new_entry_price (float or str): New entry price for CE OTM if re-entered, else empty string,
            PE_OTM_new_entry_price (float or str): New entry price for PE OTM if re-entered, else empty string
        ]
    Notes:
        - The function handles both CE and PE legs, checks for stoploss/target breaches, and manages re-entry logic if one leg is squared off before the other.
        - If neither stoploss nor target is hit, positions are exited at the specified EXIT time.
        - The function expects the option_data DataFrame to be indexed by datetime and to contain the necessary columns.
    """


    start_time = next_time_period
    end_time = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')

    # Filter option data for the date, time, strike and type of the entry position    
    intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]

    intraday_data_ce = intraday_data[(intraday_data['StrikePrice']==CE_OTM) & (intraday_data['Type']=='CE')]    
    intraday_data_ce = intraday_data_ce.sort_index()

    intraday_data_pe = intraday_data[(intraday_data['StrikePrice']==PE_OTM) & (intraday_data['Type']=='PE')]    
    intraday_data_pe = intraday_data_pe.sort_index()

    try:
        CE_OTM_entry_price = intraday_data_ce.iloc[0]['Open']
    except:
        print(next_time_period, CE_OTM, PE_OTM)
        print(option_data)
        CE_OTM_entry_price = intraday_data_ce.iloc[0]['Open']

    intraday_data_ce = intraday_data_ce[intraday_data_ce.index > next_time_period]
    
    PE_OTM_entry_price = intraday_data_pe.iloc[0]['Open']
    intraday_data_pe = intraday_data_pe[intraday_data_pe.index > next_time_period]
    
    PE_OTM_new_entry_price = CE_OTM_new_entry_price = ''

    if PREMIUM_TP=='MIN':
        
        if CE_OTM_entry_price < PE_OTM_entry_price:
            ce_target_pt = CE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
            pe_target_pt = PE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
        
        else:
            ce_target_pt = CE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT
            pe_target_pt = PE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT
    else:
            ce_target_pt = CE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
            pe_target_pt = PE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT

    crosses_threshold_ce = (intraday_data_ce['Open'] > ce_target_pt)
    crosses_threshold_pe = (intraday_data_pe['Open'] > pe_target_pt)

    if intraday_data_ce.empty:
            index = pd.to_datetime([start_time])
            data = [False]
            crosses_threshold_ce = pd.Series(data, index=index, name='Value')
    if intraday_data_pe.empty:
            index = pd.to_datetime([start_time])
            data = [False]
            crosses_threshold_pe = pd.Series(data, index=index, name='Value')

    
    if crosses_threshold_ce.any() & crosses_threshold_pe.any() & (crosses_threshold_ce.idxmax() < crosses_threshold_pe.idxmax()):

        if intraday_data_ce[intraday_data_ce.index > crosses_threshold_ce.idxmax()].empty:
            ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
            try:
                CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
            except:    
                print(ce_exit_time_period)
                next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
                CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']
            ce_stoploss = 0
        else:
            crosses_threshold_ce_index = crosses_threshold_ce.idxmax()
            try:
                CE_OTM_exit_price = intraday_data_ce.at[crosses_threshold_ce_index + pd.to_timedelta('1T'), 'Open']
            except:
                print(crosses_threshold_ce_index + pd.to_timedelta('1T'))
                next_index = intraday_data_ce.index[intraday_data_ce.index < (crosses_threshold_ce_index + pd.to_timedelta('1T'))].max()
                CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']

            ce_exit_time_period = crosses_threshold_ce_index + pd.to_timedelta('1T')
            exit_type = 'Initial Target Exit'
            ce_stoploss = 1

            if crosses_threshold_ce.idxmax() < crosses_threshold_pe.idxmax():

                intraday_data_pe = intraday_data_pe.sort_index()
                intraday_data_pe = intraday_data_pe[intraday_data_pe.index >= ce_exit_time_period]

                PE_OTM_new_entry_price = intraday_data_pe.iloc[0]['Open'] 
                pe_target_pt = PE_OTM_new_entry_price * (1 + STOPLOSS_PT)

                crosses_threshold_pe = (intraday_data_pe['Open'] > pe_target_pt)

                if crosses_threshold_pe.any():
                    if intraday_data_pe[intraday_data_pe.index > crosses_threshold_pe.idxmax()].empty:
                        pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                        try:
                            PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
                        except:
                            next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
                            PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
                        
                        pe_stoploss = 0
                    else:
                        crosses_threshold_pe_index = crosses_threshold_pe.idxmax()
                        try:
                            PE_OTM_exit_price = intraday_data_pe.at[crosses_threshold_pe_index + pd.to_timedelta('1T'), 'Open']
                        except:
                            next_index = intraday_data_pe.index[intraday_data_pe.index < (crosses_threshold_pe_index + pd.to_timedelta('1T'))].max()
                            PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
                        
                        pe_exit_time_period = crosses_threshold_pe_index + pd.to_timedelta('1T')
                        exit_type = 'Initial Target Exit'
                        pe_stoploss = 1                
                else:
                    pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                    try:    
                        PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
                    except:
                        next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
                        PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open'] 
                    
                    pe_stoploss = 0

    elif crosses_threshold_pe.any() & crosses_threshold_ce.any() & (crosses_threshold_pe.idxmax() < crosses_threshold_ce.idxmax()):

        if intraday_data_pe[intraday_data_pe.index > crosses_threshold_pe.idxmax()].empty:
            pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
            try:
                PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
            except:
                next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
                PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
            
            pe_stoploss = 0
        else:
            crosses_threshold_pe_index = crosses_threshold_pe.idxmax()
            try:
                PE_OTM_exit_price = intraday_data_pe.at[crosses_threshold_pe_index + pd.to_timedelta('1T'), 'Open']
            except:
                next_index = intraday_data_pe.index[intraday_data_pe.index < (crosses_threshold_pe_index + pd.to_timedelta('1T'))].max()
                PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
            
            pe_exit_time_period = crosses_threshold_pe_index + pd.to_timedelta('1T')
            exit_type = 'Initial Target Exit'
            pe_stoploss = 1

            if crosses_threshold_pe.idxmax() < crosses_threshold_ce.idxmax():

                intraday_data_ce = intraday_data_ce.sort_index()
                intraday_data_ce = intraday_data_ce[intraday_data_ce.index >= pe_exit_time_period]

                CE_OTM_new_entry_price = intraday_data_ce.iloc[0]['Open'] 
                ce_target_pt = CE_OTM_new_entry_price * (1 + STOPLOSS_PT)

                crosses_threshold_ce = (intraday_data_ce['Open'] > ce_target_pt)
                update_sl = 1

                if crosses_threshold_ce.any():
                    if intraday_data_ce[intraday_data_ce.index > crosses_threshold_ce.idxmax()].empty:
                        ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
                        try:
                            CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
                        except:    
                            print(ce_exit_time_period)
                            next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
                            CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']
                        ce_stoploss = 0
                    else:
                        crosses_threshold_ce_index = crosses_threshold_ce.idxmax()
                        try:
                            CE_OTM_exit_price = intraday_data_ce.at[crosses_threshold_ce_index + pd.to_timedelta('1T'), 'Open']
                        except:
                            print(crosses_threshold_ce_index + pd.to_timedelta('1T'))
                            next_index = intraday_data_ce.index[intraday_data_ce.index < (crosses_threshold_ce_index + pd.to_timedelta('1T'))].max()
                            CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']

                        ce_exit_time_period = crosses_threshold_ce_index + pd.to_timedelta('1T')
                        exit_type = 'Initial Target Exit'
                        ce_stoploss = 1                   
                else:
                    ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10] + ' ' + EXIT + ':00')
                    try:
                        CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
                    except KeyError:
                        # print(start_time, end_time, CE_OTM, ce_exit_time_period)
                        next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
                        CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open'] 

                    ce_stoploss = 0

    elif crosses_threshold_ce.any() & (not crosses_threshold_pe.any()):
        if intraday_data_ce[intraday_data_ce.index > crosses_threshold_ce.idxmax()].empty:
            ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
            try:
                CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
            except:    
                print(ce_exit_time_period)
                next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
                CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']
            ce_stoploss = 0
        else:
            crosses_threshold_ce_index = crosses_threshold_ce.idxmax()
            try:
                CE_OTM_exit_price = intraday_data_ce.at[crosses_threshold_ce_index + pd.to_timedelta('1T'), 'Open']
            except:
                print(crosses_threshold_ce_index + pd.to_timedelta('1T'))
                next_index = intraday_data_ce.index[intraday_data_ce.index < (crosses_threshold_ce_index + pd.to_timedelta('1T'))].max()
                CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open']

            ce_exit_time_period = crosses_threshold_ce_index + pd.to_timedelta('1T')
            exit_type = 'Initial Target Exit'
            ce_stoploss = 1   

        pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        try:    
            PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
        except:
            next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
            PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open'] 
        
        pe_stoploss = 0
    
    elif crosses_threshold_pe.any() & (not crosses_threshold_ce.any()):
        if intraday_data_pe[intraday_data_pe.index > crosses_threshold_pe.idxmax()].empty:
            pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
            try:
                PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
            except:
                next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
                PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
            
            pe_stoploss = 0

        else:
            crosses_threshold_pe_index = crosses_threshold_pe.idxmax()
            try:
                PE_OTM_exit_price = intraday_data_pe.at[crosses_threshold_pe_index + pd.to_timedelta('1T'), 'Open']
            except:
                next_index = intraday_data_pe.index[intraday_data_pe.index < (crosses_threshold_pe_index + pd.to_timedelta('1T'))].max()
                PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open']
            
            pe_exit_time_period = crosses_threshold_pe_index + pd.to_timedelta('1T')
            exit_type = 'Initial Target Exit'
            pe_stoploss = 1

        ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        try:
            CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
        except KeyError:
            # print(start_time, end_time, CE_OTM, ce_exit_time_period)
            next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
            CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open'] 

        ce_stoploss = 0
    else:
        ce_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        try:
            CE_OTM_exit_price = intraday_data_ce.at[ce_exit_time_period, 'Open']
        except KeyError:
            next_index = intraday_data_ce.index[intraday_data_ce.index < ce_exit_time_period].max()
            CE_OTM_exit_price = intraday_data_ce.at[next_index, 'Open'] 
        ce_stoploss = 0

        pe_exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + EXIT + ':00')
        try:    
            PE_OTM_exit_price = intraday_data_pe.at[pe_exit_time_period, 'Open']
        except:
            next_index = intraday_data_pe.index[intraday_data_pe.index < pe_exit_time_period].max()
            PE_OTM_exit_price = intraday_data_pe.at[next_index, 'Open'] 
        pe_stoploss = 0


    del intraday_data_ce
    del intraday_data_pe
    return [CE_OTM_entry_price, CE_OTM_exit_price, ce_exit_time_period, PE_OTM_entry_price, PE_OTM_exit_price, pe_exit_time_period, ce_stoploss, pe_stoploss, CE_OTM_new_entry_price, PE_OTM_new_entry_price]

# Function to calculate trailing stoploss, 
def TSL_SS(option_data, next_time_period, CE_OTM, PE_OTM, LAST_ENTRY, STOPLOSS_PT, PREMIUM_TP, option_type, position, I_T, L_P, INC_T, INC_S):
    """
    Implements a trailing stop loss (TSL) and step stop (SS) strategy for options trading based on intraday option data.
    Parameters:
        option_data (pd.DataFrame): DataFrame containing intraday option data with a DateTime index and columns including 'StrikePrice', 'Type', and 'Open'.
        next_time_period (datetime): The starting datetime for the trade entry.
        CE_OTM (float or int): Strike price for the Call Option (CE) Out-of-the-Money.
        PE_OTM (float or int): Strike price for the Put Option (PE) Out-of-the-Money.
        LAST_ENTRY (str): The last allowed entry time in 'HH:MM' format.
        STOPLOSS_PT (float): Stop loss percentage (e.g., 0.2 for 20%).
        PREMIUM_TP (str): Premium type, either 'MIN' for minimum premium logic or other for standard logic.
        option_type (str): Option type, either 'CE' (Call) or 'PE' (Put).
        position (int): Current position status (1 for open, 0 for closed).
        I_T (float): Initial target percentage (used to calculate the target price).
        L_P (float): Not used in the function (legacy parameter).
        INC_T (float): Increment percentage for target adjustment after each target hit.
        INC_S (float): Increment percentage for stop loss adjustment after each target hit.
    Returns:
        list: [entry_price, exit_price, exit_time_period, exit_type, position]
            entry_price (float): The price at which the position was entered.
            exit_price (float): The price at which the position was exited.
            exit_time_period (datetime): The time at which the position was exited.
            exit_type (str): The reason/type of exit ('Universal', 'Initial Stoploss Exit', 'TSL', etc.).
            position (int): Final position status (0 for closed, 1 for open).
    Notes:
        - The function implements a dynamic trailing stop loss and target adjustment strategy.
        - Handles both CE and PE options and supports both minimum premium and standard entry logic.
        - Returns empty values if entry or exit prices cannot be determined due to missing data.
    """

    start_time = next_time_period
    end_time = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')

    # Filter option data for the date, time, strike and type of the entry position    
    intraday_data = option_data[(option_data.index >= start_time) & (option_data.index <= end_time)]

    # Minimum Premium
    if PREMIUM_TP=='MIN':
        intraday_data_ce = intraday_data[(intraday_data['StrikePrice']==CE_OTM) & (intraday_data['Type']=='CE')]    
        intraday_data_ce = intraday_data_ce.sort_index()

        intraday_data_pe = intraday_data[(intraday_data['StrikePrice']==PE_OTM) & (intraday_data['Type']=='PE')]    
        intraday_data_pe = intraday_data_pe.sort_index()

        try:
            CE_OTM_entry_price = intraday_data_ce.iloc[0]['Open']        
            PE_OTM_entry_price = intraday_data_pe.iloc[0]['Open']   
        except:
            print(start_time)
            return ['', '', '', '', '']

        if CE_OTM_entry_price < PE_OTM_entry_price:

            if option_type=='CE':
                initial_stoploss_pt = CE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT
            elif option_type=='PE':
                initial_stoploss_pt = PE_OTM_entry_price + CE_OTM_entry_price * STOPLOSS_PT

        else:
            if option_type=='CE':
                initial_stoploss_pt = CE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT
            elif option_type=='PE':
                initial_stoploss_pt = PE_OTM_entry_price + PE_OTM_entry_price * STOPLOSS_PT

    if option_type == 'CE':
        intraday_data = intraday_data[(intraday_data['StrikePrice']==CE_OTM) & (intraday_data['Type']==option_type)]    
    elif option_type == 'PE':
        intraday_data = intraday_data[(intraday_data['StrikePrice']==PE_OTM) & (intraday_data['Type']==option_type)]

    intraday_data = intraday_data.sort_index()
    
    try:
        entry_price = intraday_data.iloc[0]['Open']
    except:    
        return ['', '', '', '', '']

    intraday_data = intraday_data[intraday_data.index > next_time_period]

    target_pt = entry_price - entry_price * I_T / 100
    if PREMIUM_TP != 'MIN':
        initial_stoploss_pt = entry_price + entry_price * STOPLOSS_PT

    INC_T = entry_price * INC_T / 100
    INC_S = entry_price * INC_S / 100

    # initial_target_pt = target
    target_hit = 0

    # crosses_threshold1 - when index goes below the initial stoploss pt (we take exit here)
    # crosses_threshold2 - when option premium goes above the profit pt
    # crosses_threshold3 - when option premium goes below the stoploss pt (we take exit here)
    # crooses_threshold4 - when index goes above the initial target pt (we take exit here)

    while True:

        # If the target pt is not hit once yet check for intitial stoploss pt hit
        if target_hit==0:
            crosses_threshold1 = (intraday_data['Open'] > initial_stoploss_pt)

        else:
            crosses_threshold1 = pd.Series()
         
        crosses_threshold2 = (intraday_data['Open'] < target_pt)
        
        # If the target pt is hit already check for stoplos pt hit
        if target_hit:
            crosses_threshold3 = (intraday_data['Open'] > stoploss_pt)

        else:
            crosses_threshold3 = pd.Series()
        
        # When target pt is not hit for the first time yet and initial stoploss pt is also not hit throughout day
        if (target_hit==0) & (not crosses_threshold1.any()) & (not crosses_threshold2.any()):

            exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
            # exit_price = intraday_data.at[exit_time_period, 'Open']
            
            try:
                exit_price = intraday_data.at[exit_time_period, 'Open']
            except:

                next_index = intraday_data.index[intraday_data.index < exit_time_period].max()
                try:
                    exit_price = intraday_data.at[next_index, 'Open']
                except:
                    return ['', '', '', '', '']
            exit_type = 'Universal'
            position = 0
            break

        # When the target pt is not hit for the first time yet
        if (target_hit==0): 

            # When the initial stoploss pt got hit before target pt
            if crosses_threshold1.any() & crosses_threshold2.any() & ((crosses_threshold1.idxmax()) < crosses_threshold2.idxmax()):
                crosses_threshold1_index = crosses_threshold1.idxmax()
                exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break
            
            # When the target pt got hit before the initial stoploss pt
            elif crosses_threshold1.any() & crosses_threshold2.any() & (crosses_threshold2.idxmax() < crosses_threshold1.idxmax()):
                # target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                target_mul = ((target_pt - intraday_data.at[crosses_threshold2.idxmax(), 'Open']) // INC_T) + 1
                stoploss_pt = entry_price - (target_mul-1) * INC_S
                target_pt = target_pt - target_mul * INC_T
                target_hit = 1
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]

            # When the initial stoploss pt got hit but target pt is not hit
            elif crosses_threshold1.any():
                crosses_threshold1_index = crosses_threshold1.idxmax()
                try:
                    exit_price = intraday_data.at[crosses_threshold1_index + pd.to_timedelta('1T'), 'Open']
                except:
                    next_index = intraday_data.index[intraday_data.index < (crosses_threshold1_index + pd.to_timedelta('1T'))].max()
                    exit_price = intraday_data.at[next_index, 'Open']

                exit_time_period = crosses_threshold1_index + pd.to_timedelta('1T')
                exit_type = 'Initial Stoploss Exit'
                position = 0
                break
            
            # When only target pt got hit
            elif crosses_threshold2.any():
                # target_mul = ((intraday_data.at[crosses_threshold2.idxmax(), 'Open'] - target_pt) // INC_T) + 1
                target_mul = ((target_pt - intraday_data.at[crosses_threshold2.idxmax(), 'Open']) // INC_T) + 1
                stoploss_pt = entry_price - (target_mul-1) * INC_S
                target_pt = target_pt - target_mul * INC_T
                target_hit = 1

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]

        # When the target pt is already hit once
        elif target_hit: 
            
            # When target pt is already hit once and if target pt and stoploss pt none of them got hit throughout day take an exit
            if (not crosses_threshold2.any()) & (not crosses_threshold3.any()):
                
                exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                try:
                    exit_price = intraday_data.at[exit_time_period, 'Open']
                except:
                    # print(option_type, CE_OTM, PE_OTM, exit_time_period)
                    # print(intraday_data)
                    exit_price = intraday_data['Open'].iloc[-1]

                exit_type = 'Universal'
                position = 0
                break 
            
            # When target pt is already hit once and stoploss pt got hit before target pt 
            elif crosses_threshold2.any() & crosses_threshold3.any() & (crosses_threshold3.idxmax() < crosses_threshold2.idxmax()) :

                try:
                    exit_price = intraday_data.at[crosses_threshold3.idxmax() + pd.to_timedelta('1T'), 'Open']
                except:
                    next_index = intraday_data.index[intraday_data.index < (crosses_threshold3.idxmax() + pd.to_timedelta('1T'))].max()
                    exit_price = intraday_data.at[next_index, 'Open']

                exit_time_period = crosses_threshold3.idxmax() + pd.to_timedelta('1T')
                exit_type = 'TSL'
                position = 0
                break
     
            # When target pt is already hit once and target pt got hit before the initial stoploss pt
            elif crosses_threshold2.any() & crosses_threshold3.any() & (crosses_threshold2.idxmax() < crosses_threshold3.idxmax()):
                target_mul = ((target_pt - intraday_data.at[crosses_threshold2.idxmax(), 'Open']) // INC_T) + 1
                
                stoploss_pt = stoploss_pt - target_mul * INC_S
                target_pt = target_pt - target_mul * INC_T

                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:

                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    try:
                        exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    except:
                        # print(crosses_threshold2.idxmax())
                        # print(intraday_data)
                        exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']

                    exit_type = 'Universal'
                    position = 0
                    break
                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]

            # When target pt is already hit once and only target pt got hit 
            elif crosses_threshold2.any():
                target_mul = ((target_pt - intraday_data.at[crosses_threshold2.idxmax(), 'Open']) // INC_T) + 1
                
                stoploss_pt = stoploss_pt - target_mul * INC_S
                target_pt = target_pt - target_mul * INC_T
                
                # If target pt got hit on the last timestamp take an exit
                if intraday_data[intraday_data.index > crosses_threshold2.idxmax()].empty:
                    exit_time_period = pd.to_datetime(next_time_period.strftime('%Y-%m-%d %H:%M:%S')[0:10]+ ' ' + LAST_ENTRY + ':00')
                    exit_price = intraday_data.at[crosses_threshold2.idxmax(), 'Open']
                    exit_type = 'Universal'
                    position = 0
                    break

                intraday_data = intraday_data[intraday_data.index > crosses_threshold2.idxmax()]

            # When target pt is already hit once and stoploss pt got hit but target pt is not hit
            elif crosses_threshold3.any():
                if intraday_data[intraday_data.index > crosses_threshold3.idxmax()].empty:
                    exit_price = intraday_data.at[crosses_threshold3.idxmax(), 'Open']
                    exit_time_period = crosses_threshold3.idxmax()
                else:

                    try:
                        exit_price = intraday_data.at[crosses_threshold3.idxmax() + pd.to_timedelta('1T'), 'Open']
                    except:
                        next_index = intraday_data.index[intraday_data.index < (crosses_threshold3.idxmax() + pd.to_timedelta('1T'))].max()
                        exit_price = intraday_data.at[next_index, 'Open']

                    exit_time_period = crosses_threshold3.idxmax() + pd.to_timedelta('1T')
                
                exit_type = 'TSL'
                position = 0
                break

        if position == 0:
            break

    return [entry_price, exit_price, exit_time_period, exit_type, position]











































































