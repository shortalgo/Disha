# import datetime as dt
import multiprocessing
import numpy as np
import pandas as pd
import psycopg2
import talib as ta
import time
from tqdm import tqdm
from functools import partial
import os
from datetime import datetime, timedelta
import ast, json, sys, re
sys.path.insert(0, r"/home/newberry3/user/")
from Common_Functions.utils import TSL, postgresql_query, resample_data, nearest_multiple, round_to_next_5_minutes
from Common_Functions.utils import get_target_stoploss, get_open_range, check_crossover, compare_month_and_year
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



            option_data['StrikePrice'] = option_data['StrikePrice'].astype('int32')
            option_data['Type'] = option_data['Type'].astype('category')
            
            end_time = time.time()
            print('Time taken to pull Options data :', (end_time-start_time))

            return option_data

# Function to pull index data for specified date range
def pull_index_data(start_date_idx, end_date_idx, stock, mapped_days):
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
    index_data['Date'] = pd.to_datetime(index_data['Date'])
    df = index_data.merge(mapped_days, on = 'Date')

    df.index = pd.to_datetime(df['Ticker'].str[0:13], format = '%Y%m%d%H:%M')
    df = df.rename_axis('DateTime')
    df = df.sort_index()
    df = df.drop_duplicates()
    
    return df

def trade_sheet_creator(mapped_days, option_data, df, start_date, end_date, counter,
                        output_folder_path, resampled, TIME_FRAME, STRIKE, option_side, ENTRY, EXIT, MONEYNESS):
    column_names = ['Date', 'Position', 'Action', 'Time', 'Index Value', 'Strike',
                    'Exit Time', 'ExpiryDate', 'DaysToExpiry', 'Option_Premium', 'EXIT_TYPE', 'Strike_Label']
    trade_sheet = []

    target_list = [TIME_FRAME, STRIKE, option_side, ENTRY, EXIT]

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
        expiry_date = pd.to_datetime(row['ExpiryDate'])
        days_to_expiry = row['DaysToExpiry']

        start_time = pd.to_datetime(f'{date} 09:15:00')
        end_time = pd.to_datetime(f'{expiry_date} {EXIT}:00')

        daily_data = df[(df.index >= start_time) & (df.index <= end_time)].copy()
        daily_option_data = option_data[
            (option_data.index.date >= date.date()) &
            (option_data.index.date <= expiry_date.date())
        ]

        signal_rows = resampled[
            (resampled['Entry_Signal']) & (resampled.index.date == date.date())
        ]

        if signal_rows.empty:
            continue

        signal_time = signal_rows.index[0]
        entry_time = signal_time + pd.Timedelta(minutes=5)

        if entry_time not in daily_data.index:
            continue

        current_index_open = daily_data.loc[entry_time, 'Open']
        round_step = 50 if stock == 'NIFTY' else 100
        ATM = nearest_multiple(current_index_open, round_step)
        strike_price = int(ATM + STRIKE * round_step)

        strike_label = MONEYNESS
        try:
            entry_price, exit_price, exit_time, *_ , exit_type = TSL_SS(
                daily_option_data, entry_time, strike_price, strike_price, EXIT, days_to_expiry, option_side
            )
        except Exception as e:
            print(f"[{date}] TSL_SS failed: {e}")
            continue

        if isinstance(exit_time, int):
            exit_time = pd.to_datetime(f'{date} {EXIT}:00').time()

        trade_sheet.append(pd.Series([
            date, 1, 'Buy', entry_time.time(), current_index_open,
            strike_price, '', expiry_date, days_to_expiry,
            entry_price, '', strike_label
        ], index=column_names))

        trade_sheet.append(pd.Series([
            date, 0, 'Sell', exit_time, current_index_open,
            strike_price, '', expiry_date, days_to_expiry,
            exit_price, exit_type, strike_label
        ], index=column_names))

    try:
        trade_sheet = pd.concat(trade_sheet, axis=1).T
    except Exception as e:
        print(f"Concat error: {e}")
        return f"{stock}_candle_{TIME_FRAME}_strike_{STRIKE}_side_{option_side}_entry_{ENTRY}_exit_{EXIT}_{start_date}_{end_date}"

    trade_sheet['Premium'] = np.where(
        trade_sheet['Action'] == 'Buy',
        -trade_sheet['Option_Premium'],
        trade_sheet['Option_Premium']
    )

    trade_sheet['P&L'] = 0.0
    for date in trade_sheet['Date'].unique():
        trades = trade_sheet[trade_sheet['Date'] == date].reset_index(drop=True)
        if len(trades) == 2:
            pnl = trades.loc[1, 'Option_Premium'] - trades.loc[0, 'Option_Premium']
            trade_sheet.loc[trade_sheet['Date'] == date, 'P&L'] = pnl

    # Summary Metrics
    gross_pnl = trade_sheet['P&L'].sum()
    num_trades = len(trade_sheet['Date'].unique())
    total_brokerage = num_trades * 2 * brokerage
    net_pnl = gross_pnl - total_brokerage

    daywise = trade_sheet.groupby('Date')['P&L'].sum()
    winning_trades = daywise[daywise > 0]
    losing_trades = daywise[daywise <= 0]

    win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
    avg_gain = winning_trades.mean() if not winning_trades.empty else 0
    avg_loss = losing_trades.mean() if not losing_trades.empty else 0
    max_gain = winning_trades.max() if not winning_trades.empty else 0
    max_loss = losing_trades.min() if not losing_trades.empty else 0
    expectancy = (win_rate * avg_gain) + ((1 - win_rate) * avg_loss)

    # Sharpe Ratio & Drawdown Stats
    daily_returns = daywise / abs(daywise.shift(1).fillna(1))
    sharpe_ratio = (
        daily_returns.mean() / daily_returns.std(ddof=1) * np.sqrt(252)
        if daily_returns.std(ddof=1) != 0 else 0
    )

    equity_curve = daywise.cumsum()
    roll_max = equity_curve.cummax()
    drawdown = equity_curve - roll_max
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean() if not drawdown[drawdown < 0].empty else 0

    drawdown_flags = daywise < 0
    consec_loss_streaks = (drawdown_flags != drawdown_flags.shift()).cumsum()
    consec_loss_lengths = drawdown_flags.groupby(consec_loss_streaks).sum()
    max_consec_losses = consec_loss_lengths.max()

    # Save to Excel
    entry_safe = ENTRY.replace(":", "-")
    exit_safe = EXIT.replace(":", "-")
    strategy_name = f'{stock}_candle_{TIME_FRAME}_strike_{MONEYNESS}_side_{option_side}_entry_{ENTRY}_exit_{EXIT}'
    sanitized_name = f'{strategy_name}_{start_date}_{end_date}'
    file_path = os.path.join(output_folder_path, sanitized_name + '.xlsx')

    if not trade_sheet.empty:
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            trade_sheet.to_excel(writer, sheet_name='TradeSheet', index=False)

            summary = pd.DataFrame({
                'Metric': [
                    'Gross P&L', 'Total Brokerage', 'Net P&L',
                    'Number of Trades', 'Win Rate (%)', 'Avg Gain',
                    'Avg Loss', 'Max Gain', 'Max Loss', 'Expectancy',
                    'Sharpe Ratio', 'Max Drawdown', 'Avg Drawdown',
                    'Max Consecutive Loss Days'
                ],
                'Value': [
                    round(gross_pnl, 2), round(total_brokerage, 2), round(net_pnl, 2),
                    num_trades, round(win_rate * 100, 2), round(avg_gain, 2),
                    round(avg_loss, 2), round(max_gain, 2), round(max_loss, 2),
                    round(expectancy, 2), round(sharpe_ratio, 2),
                    round(max_drawdown, 2), round(avg_drawdown, 2),
                    int(max_consec_losses)
                ]
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)

    # Save metadata for filter tracking
    filter_df1 = pd.DataFrame(columns=['Strategy', 'Parameters', 'Status'])
    filter_df1.loc[len(filter_df1), 'Strategy'] = sanitized_name
    filter_df1.loc[filter_df1['Strategy'] == sanitized_name, 'Parameters'] = [target_list]
    filter_df1.loc[filter_df1['Strategy'] == sanitized_name, 'Status'] = 0

    filter_csv_file = os.path.join(filter_df_path, f'filter_df{counter}.csv')
    if os.path.isfile(filter_csv_file):
        filter_df1.to_csv(filter_csv_file, index=False, mode='a', header=False)
    else:
        filter_df1.to_csv(filter_csv_file, index=False)

    return sanitized_name





 

def TSL_SS(option_data, next_time_period, CE_OTM, PE_OTM, EXIT, days_to_expiry, option_side):
    """
    EOD exit logic for selected option type ('CE' or 'PE')
    """
    end_time = pd.to_datetime(next_time_period.strftime('%Y-%m-%d') + ' ' + EXIT + ':00')

    strike = CE_OTM if option_side == 'CE' else PE_OTM

    option_data_filtered = option_data[
        (option_data['StrikePrice'] == strike) &
        (option_data['Type'] == option_side) &
        (option_data.index >= next_time_period) &
        (option_data.index <= end_time)
    ].sort_index()

    if option_data_filtered.empty:
        return [0, 0, 0, 0, 0, 0, "No Option Data"]

    entry_price = option_data_filtered.iloc[0]['Open']
    exit_price = option_data_filtered.iloc[-1]['Open']
    final_exit_time = option_data_filtered.index[-1]

    # Filler zeros for compatibility
    return [entry_price, exit_price, final_exit_time, 0, 0, final_exit_time, "Time Exit"]





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
superset = 'Simple_EMA5-20_Crossover'              # Strategy category/folder name
stock = 'NIFTY'                         # Stock/index being tested

# Strategy parameters
candle_time_frame = ['5T']          # 5-minute candle frequency
entries = ['09:25']                 # Entry time
exits = ['15:20']                   # Exit time
option_side = ['CE']               # CE or PE 
parameters = []                     # To store all parameter combinations
moneyness = ['ATM']                  #ATM, OTM, ITM




#OPTION_SIDE= option_side
# Set roundoff step, brokerage, and lot size based on selected stock
roundoff = 50 if stock == 'NIFTY' else (100 if stock == 'BANKNIFTY' or stock == 'SENSEX' else None)
brokerage = 4.5 if stock == 'NIFTY' else (3 if stock == 'BANKNIFTY' or stock == 'SENSEX' else None)
LOT_SIZE = 25 if stock == 'NIFTY' else (15 if stock == 'BANKNIFTY' else 10)

# Define folder structure based on parameters
root_path = rf"/home/newberry3/main/BASE-NB{superset}/{stock}/{option_side}/"
filter_df_path = rf"{root_path}/Filter_Sheets/"                   # Folder for filtered parameter files
expiry_file_path = rf"/home/newberry3/main/BASE-NB/Creating Pickle Files/NIFTY Market Dates.xlsx"   # Excel containing expiry mapping
txt_file_path = rf'{root_path}/new_done.txt'                     # File to track completed parameters
output_folder_path = rf'{root_path}/Trade_Sheets/'               # Folder for final trade sheets



# Set option data path depending on stock type
if stock == 'NIFTY':
    option_data_path = rf"/home/newberry3/Data/NIFTY/"
# elif stock == 'BANKNIFTY':
#     option_data_path = rf"/home/newberry3/user/Data/BANKNIFTY/folder/"
# elif stock == 'FINNIFTY':
#     option_data_path = rf"/home/newberry3/user/Data/FINNIFTY/folder/"
# elif stock == 'SENSEX':
#     option_data_path = rf"/home/newberry3/user/Data/SENSEX/folder/"

# Ensure necessary folders and tracking file exist
os.makedirs(root_path, exist_ok=True)
os.makedirs(filter_df_path, exist_ok=True)
os.makedirs(output_folder_path, exist_ok=True)
open(txt_file_path, 'a').close() if not os.path.exists(txt_file_path) else None

# Define the backtesting date ranges (only 2024)
date_ranges = [
     ('2024-01-01', '2024-12-31')
    # ('2024-02-01', '2024-02-29'),  # 2024 is a leap year
    # ('2024-03-01', '2024-03-31'),
    # ('2024-04-01', '2024-04-30'),
    # ('2024-05-01', '2024-05-31'),
    # ('2024-06-01', '2024-06-30'),
    # ('2024-07-01', '2024-07-31'),
    # ('2024-08-01', '2024-08-31'),
    # ('2024-09-01', '2024-09-30'),
    # ('2024-10-01', '2024-10-31'),
    # ('2024-11-01', '2024-11-30'),
    # ('2024-12-01', '2024-12-31'),
]



# Function that handles trade logic for each parameter set
def parameter_process(parameter, mapped_days, option_data, df, start_date, end_date, counter, output_folder_path):
    TIME_FRAME, STRIKE, OPTION_SIDE, ENTRY, EXIT, MONEYNESS 

    resampled_df = resample_data(df, TIME_FRAME)

    # Compute EMAs
    resampled_df['EMA_5'] = resampled_df['Close'].ewm(span=5, adjust=False).mean()
    resampled_df['EMA_20'] = resampled_df['Close'].ewm(span=20, adjust=False).mean()

    # ENTRY LOGIC BASED ON OPTION SIDE
    if OPTION_SIDE.upper() == 'CE':  # Call logic (bullish)
        resampled_df['EMA_Cross_Up'] = (resampled_df['EMA_5'] > resampled_df['EMA_20']) & \
                                       (resampled_df['EMA_5'].shift(1) <= resampled_df['EMA_20'].shift(1))
        resampled_df['Above_Both_EMAs'] = (resampled_df['Close'] > resampled_df['EMA_5']) & \
                                          (resampled_df['Close'] > resampled_df['EMA_20'])
        resampled_df['Entry_Signal'] = resampled_df['EMA_Cross_Up'] & resampled_df['Above_Both_EMAs']

    elif OPTION_SIDE.upper() == 'PE':  # Put logic (bearish)
        resampled_df['EMA_Cross_Down'] = (resampled_df['EMA_5'] < resampled_df['EMA_20']) & \
                                         (resampled_df['EMA_5'].shift(1) >= resampled_df['EMA_20'].shift(1))
        resampled_df['Below_Both_EMAs'] = (resampled_df['Close'] < resampled_df['EMA_5']) & \
                                          (resampled_df['Close'] < resampled_df['EMA_20'])
        resampled_df['Entry_Signal'] = resampled_df['EMA_Cross_Down'] & resampled_df['Below_Both_EMAs']

    else:
        print(f"Unknown OPTION_SIDE: {OPTION_SIDE}")
        return None

    resampled = resampled_df.dropna()

    return trade_sheet_creator(
        mapped_days, option_data, df, start_date, end_date,
        counter, output_folder_path, resampled,
        TIME_FRAME, STRIKE, OPTION_SIDE, ENTRY, EXIT, MONEYNESS
    )



# Adjust strike step for BANKNIFTY or SENSEX
if stock == 'BANKNIFTY' or stock == 'SENSEX':
    strikes = [x * 2 for x in strikes]

# Main execution block
if __name__ == "__main__":
    superset = 'Simple_EMA5-20_Crossover'
    stock = 'NIFTY'
    roundoff = 50 if stock == 'NIFTY' else 100
    LOT_SIZE = 25 if stock == 'NIFTY' else 15
    brokerage = 4.5 if stock == 'NIFTY' else 3

    counter = 0
    start_date_idx = date_ranges[0][0]
    end_date_idx = date_ranges[-1][1]

    mapped_days = pd.read_excel(expiry_file_path)
    mapped_days['Date'] = pd.to_datetime(mapped_days['Date'])
    mapped_days = mapped_days.rename(columns={'WeeklyDaysToExpiry': 'DaysToExpiry'})
    mapped_days = mapped_days[(mapped_days['Date'] >= start_date_idx) & (mapped_days['Date'] <= end_date_idx)]

    df = pull_index_data(start_date_idx, end_date_idx, stock, mapped_days)
    resampled_df_main = resample_data(df, '5T')

for start_date, end_date in date_ranges:
    counter += 1
    print(start_date, end_date, counter)

    start_date_object = pd.to_datetime(start_date)
    end_date_object = pd.to_datetime(end_date)
    new_end_date_object = end_date_object + timedelta(days=30)
    new_end_date = new_end_date_object.strftime('%Y-%m-%d')

    option_data = pull_options_data_d(start_date, new_end_date, option_data_path, stock)

    # Build parameters
    parameters = []
    if counter == 1:
        for TIME_FRAME in candle_time_frame:
            for MONEYNESS in moneyness:
                for OPTION_SIDE in option_side:
                    for ENTRY in entries:
                        for EXIT in exits:
                            if ENTRY < EXIT:
                                if MONEYNESS == 'ATM':
                                    STRIKE = 0
                                elif MONEYNESS == 'OTM':
                                    STRIKE = 1 if OPTION_SIDE == 'CE' else -1
                                elif MONEYNESS == 'ITM':
                                    STRIKE = -1 if OPTION_SIDE == 'CE' else 1

                                parameters.append([TIME_FRAME, STRIKE, OPTION_SIDE, ENTRY, EXIT, MONEYNESS])
    else:
        filter_file = f"/home/newberry3/main/BASE-NB{superset}/{stock}/{option_side}/Filter_Sheets/filter_df{counter - 1}.csv"
        if not os.path.exists(filter_file):
            print(f"File {filter_file} does not exist. Stopping the code.")
            sys.exit()
        filter_df = pd.read_csv(filter_file).drop_duplicates()
        parameters = [ast.literal_eval(item.replace("'", "\"")) for item in filter_df['Parameters'].to_list()]

    print('Total parameters :', len(parameters))

    txt_file_path = rf"/home/newberry3/main/BASE-NB/{superset}/{stock}/{OPTION_SIDE}/new_done.txt"
    with open(txt_file_path, 'r') as file:
        existing_values = [line.strip() for line in file]

    parameters = [value for value in parameters if
                  (stock + '_candle_' + str(value[0]) +
                   '_strike_' + str(value[1]) +
                   '_side_' + str(value[2]) +
                   '_entry_' + str(value[3]).replace(':', ',') +
                   '_exit_' + str(value[4]).replace(':', ',') +
                   '_' + start_date + '_' + end_date) not in existing_values]

    print('Parameters to run :', len(parameters))
    for value in parameters:
        string = (stock + '_candle_' + str(value[0]) +
                  '_strike_' + str(value[1]) +
                  '_side_' + str(value[2]) +
                  '_entry_' + str(value[3]).replace(':', ',') +
                  '_exit_' + str(value[4]).replace(':', ',') +
                  '_' + start_date + '_' + end_date)
        print(string)

    start_time = time.time()
    num_processes = 12
    print('No. of processes :', num_processes)

    partial_process = partial(parameter_process,
                              mapped_days=mapped_days,
                              option_data=option_data,
                              df=df,
                              start_date=start_date,
                              end_date=end_date,
                              counter=counter,
                              output_folder_path=output_folder_path)

    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(parameters), desc='Processing', unit='Iteration') as pbar:
            def update_progress(combinations):
                with open(txt_file_path, 'a') as fp:
                    fp.write(str(combinations) + '\n')
                pbar.update()

            arg_tuples = []
            for parameter in parameters:
                TIME_FRAME, STRIKE, OPTION_SIDE, ENTRY, EXIT, MONEYNESS = parameter
                arg_tuples.append((
                    mapped_days, option_data, df, start_date, end_date,
                    counter, output_folder_path,
                    TIME_FRAME, STRIKE, OPTION_SIDE, ENTRY, EXIT, MONEYNESS
                ))

            for result in pool.imap_unordered(partial_process, arg_tuples):
                update_progress(result)

    print('Time taken to get Initial Tradesheets:', time.time() - start_time)

print('Finished at :', time.time())
