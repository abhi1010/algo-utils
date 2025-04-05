from trading.common import utils

logger = utils.get_logger('screener-momo', should_add_ts=True, use_rich=True)

from enum import StrEnum
import io
import sys
import argparse
from datetime import datetime, date
import os
import pickle

import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
import yfinance as yf
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
import time
import calendar

from trading.stocks import dhan
from trading.common.utils import HLOC
from trading.data import download_bhav

from trading.screener import screener_common


# Function to calculate the Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.01):
  excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
  return excess_returns.mean() / excess_returns.std()


'''

Usage:

DATA_DL=0 scripts/screener_nifty_momo.sh
  DATA_DL=1 ETF=micro scripts/screener_nifty_momo.sh
python trading/screener/momo.py --all-tickers
python trading/screener/momo.py --all-tickers --max-workers 8
ETF=micro scripts/screener_nifty_momo.sh
ETF=small scripts/screener_nifty_momo.sh
ETF=mid   scripts/screener_nifty_momo.sh

Parameters:

- Witin 25% of all time high
- Median Daily Volume > 1 Cr
- Ignore top beta stocks (10% of all stocks)

'''

ALL_TICKERS = [
    'NSE:TRENT', 'NSE:STAR', 'NSE:POCL', 'NSE:MCX', 'NSE:DIXON',
    'NSE:CHOICEIN', 'NSE:NSIL', 'NSE:SUVENPHAR', 'NSE:GANECOS',
    'NSE:JUBLPHARMA', 'NSE:KICL', 'NSE:DPABHUSHAN', 'NSE:PGEL', 'NSE:GLENMARK',
    'NSE:PILANIINVS', 'NSE:NEULANDLAB', 'NSE:SUMMITSEC', 'NSE:WABAG',
    'NSE:VMART', 'NSE:LTFOODS', 'NSE:BASF', 'NSE:SUNPHARMA', 'NSE:WINDLAS',
    'NSE:SBGLP', 'NSE:SARDAEN', 'NSE:POWERINDIA', 'NSE:HERITGFOOD', 'NSE:ANUP',
    'NSE:GOKULAGRO', 'NSE:KALYANKJIL', 'NSE:RAMRAT', 'NSE:CHEMFAB',
    'NSE:DIVISLAB', 'NSE:PPLPHARMA', 'NSE:GOLDIAM', 'NSE:TVSHLTD', 'NSE:HSCL',
    'NSE:BHARTIARTL', 'NSE:GRWRHITECH', 'NSE:TECHNOE', 'NSE:PRIMESECU',
    'NSE:TIMETECHNO', 'NSE:POLYMED', 'NSE:VOLTAS', 'NSE:TORNTPOWER',
    'NSE:LUPIN', 'NSE:OFSS', 'NSE:UNICHEMLAB', 'NSE:TBZ', 'NSE:CHOLAHLDNG',
    'NSE:MOTHERSON', 'NSE:ZOMATO', 'NSE:ALKEM', 'NSE:FORTIS', 'NSE:TORNTPHARM',
    'NSE:SHILPAMED', 'NSE:MARKSANS', 'NSE:ABREL', 'NSE:TIPSMUSIC', 'NSE:BBOX',
    'NSE:INDRAMEDCO', 'NSE:MANORAMA', 'NSE:SHAILY', 'NSE:RPGLIFE', 'NSE:JSWHL',
    'NSE:POKARNA', 'NSE:PITTIENG', 'NSE:TARC', 'NSE:MUNJALAU', 'NSE:PRUDENT',
    'NSE:SIEMENS', 'NSE:NAUKRI', 'NSE:BOSCHLTD', 'NSE:AKZOINDIA',
    'NSE:BLUESTARCO', 'NSE:VEDL', 'NSE:TVSMOTOR', 'NSE:ADSL', 'NSE:GRAVITA',
    'NSE:LLOYDSME', 'NSE:RADICO', 'NSE:CDSL', 'NSE:ERIS', 'NSE:IFBIND',
    'NSE:BAJAJ-AUTO', 'NSE:NEWGEN', 'NSE:SANSERA', 'NSE:PERSISTENT',
    'NSE:SUNDARMHLD', 'NSE:POLICYBZR', 'NSE:AJANTPHARM', 'NSE:GREENPLY',
    'NSE:VLSFINANCE', 'NSE:INDIGO', 'NSE:BIKAJI', 'NSE:FSL', 'NSE:BALRAMCHIN',
    'NSE:M&M', 'NSE:SENCO', 'NSE:TECHM'
]


class IndexCap(StrEnum):
  ALL = 'all'
  MICRO_CAP = 'micro'
  SMALL_CAP = 'small'
  MID_CAP = 'mid'
  US_MICRO = 'us_micro'
  US_SMALL = 'us_small'
  US_MID = 'us_mid'


def get_index_tickers(index):
  filename = ''
  tickers_list = []
  if index == IndexCap.ALL:
    tickers_list = dhan.get_dhan_scrips_as_list_with_info(
        exchange_to_use='NSE')
    return tickers_list

  if index == IndexCap.MICRO_CAP:
    filename = download_bhav.download_csv_file_from_url(
        download_bhav.NIFTY_MICRO_250)

  elif index == IndexCap.SMALL_CAP:
    filename = download_bhav.download_csv_file_from_url(
        download_bhav.NIFTY_SMALL_250)

  elif index == IndexCap.MID_CAP:
    filename = download_bhav.download_csv_file_from_url(
        download_bhav.NIFTY_MID_250)

  elif index == IndexCap.US_MICRO:
    # https://www.ishares.com/us/products/239716/ishares-microcap-etf
    filename = 'data/momo-us/iShares-Micro-Cap-ETF_fund.xlsx'

  elif index == IndexCap.US_SMALL:
    # https://www.ishares.com/us/products/239774/ishares-core-sp-smallcap-etf
    filename = 'data/momo-us/iShares-Core-SP-Small-Cap-ETF_fund.xlsx'

  elif index == IndexCap.US_MID:
    # https://www.ishares.com/us/products/239763/ishares-core-sp-midcap-etf
    filename = 'data/momo-us/iShares-Core-SP-Mid-Cap-ETF_fund.xlsx'

  if 'data/momo-us' in filename:
    tickers_list = _read_momo_tickers_for_us(filename)

  else:

    logger.info(f'index= {index}; filename={filename}')
    df = pd.read_csv(filename)
    logger.info(f'df = {df}')

    # rename columns to lower case
    df.columns = df.columns.str.lower()

    # column_names = df.columns.to_list()
    # logger.info(f'column names = {column_names}')
    tickers_list = df.to_dict('records')
    logger.info(f'tickers = {tickers_list}')

  logger.info(f'filename = {filename}; tickers_list = {tickers_list}; '
              f'len = {len(tickers_list)}')

  return tickers_list


def _read_momo_tickers_for_us(filename):
  df = screener_common.read_etf_holdings(filename)
  # rename columns to lower case
  df.columns = df.columns.str.lower()
  # df = df.replace({'name': 'symbol'})

  df = df.rename(columns={'ticker': 'symbol'})
  tickers_list = df.to_dict('records')

  return tickers_list


def get_tickers(index=IndexCap.ALL, dry_run=False, num_to_pick=100):
  if dry_run:
    tickers = get_dry_run_tickers(num_to_pick)

    logger.info(f'tickers from dhan = {tickers}')
    logger.info(f'len of tickers: {len(tickers)}')
    return tickers

  tickers_list = get_index_tickers(index)
  prefix = '.NS' if is_nifty_index(index) else ''
  tickers = [f'{x["symbol"]}{prefix}' for x in tickers_list]
  if dry_run:
    tickers = tickers[:num_to_pick]
  logger.info(f'tickers from dhan = {tickers}')
  logger.info(f'len of tickers: {len(tickers)}')
  return tickers


def get_dry_run_tickers(num_to_pick):
  return [f.replace('NSE:', '') + '.NS' for f in ALL_TICKERS[:num_to_pick]]


def is_nifty_index(index):
  return index in [
      IndexCap.MICRO_CAP, IndexCap.SMALL_CAP, IndexCap.MID_CAP, IndexCap.ALL
  ]


def get_yf_ticker_info(ticker,
                       index,
                       yf_info_dir='data/yf-cache',
                       force_refresh=False):
  """
    Retrieve ticker info with local caching.

    Args:
        ticker (str): Stock ticker symbol
        dir_to_try_loading_from (str): Directory to store/load cached ticker info
        force_refresh (bool, optional): Force fetching from Yahoo Finance. Defaults to False.

    Returns:
        dict: Ticker information
    """

  if is_nifty_index(index):
    # Ensure ticker has .NS suffix
    if not ticker.endswith('.NS'):
      ticker = ticker + '.NS'

  # Create cache directory if it doesn't exist
  os.makedirs(yf_info_dir, exist_ok=True)

  # Generate cache file path
  cache_filename = os.path.join(yf_info_dir,
                                f"{ticker.replace('.', '_')}_info.pkl")

  # Check if cached file exists and not forcing refresh
  if not force_refresh and os.path.exists(cache_filename):
    try:
      # Try to load from cache
      with open(cache_filename, 'rb') as cache_file:
        cached_info = pickle.load(cache_file)
        logger.info(f'Ticker: {ticker}; Cached info: {cached_info}')
        return cached_info
    except (pickle.PickleError, EOFError):
      # If cache is corrupted, we'll fall back to fetching from Yahoo Finance
      pass

  # Fetch from Yahoo Finance if no valid cache
  try:
    time.sleep(1)  # Rate limiting courtesy
    yf_ticker = yf.Ticker(ticker)
    yf_ticker_info = yf_ticker.info

    # Validate the fetched info (optional, but recommended)
    if not yf_ticker_info or len(yf_ticker_info) == 0:
      raise ValueError(f"No info found for ticker {ticker}")

    # Cache the fetched info
    try:
      with open(cache_filename, 'wb') as cache_file:
        pickle.dump(yf_ticker_info, cache_file)
    except Exception as cache_error:
      logger.warning(
          f"Warning: Could not write cache file. Error: {cache_error}")

    return yf_ticker_info

  except Exception as e:
    logger.warning(f"Error fetching ticker info for {ticker}: {e}")
    return {}


def do_we_have_existing_data(ticker, index, dir_to_try_loading_from):
  # if it is a nifty symbol, always return false
  if ticker.endswith('.NS'):
    return False

  exp_file_path = os.path.join(dir_to_try_loading_from, f'{ticker}.csv')
  exp_file_path_exists = os.path.exists(exp_file_path)
  logger.info(f'ticker: {ticker}; exp_file_path = {exp_file_path}; '
              f'exp_file_path_exists = {exp_file_path_exists}')

  # if we have file , we should always have more than 2 rows
  if exp_file_path_exists:
    df = pd.read_csv(exp_file_path, index_col=0)
    if len(df) > 2:
      return exp_file_path

  return ''


def clean_ticker_rows(ticker, df):
  """
    Remove first and second rows if second row's first column starts with 'ticker'

    Args:
        df: pandas DataFrame to clean

    Returns:
        cleaned DataFrame with rows removed if condition is met
    """
  # if no ticker data, return the original df

  try:
    if df.shape[0] < 3:
      return df
    a = df.iloc[0].iloc[0]
    r1 = df.iloc[0]
    r2 = df.iloc[1]
    logger.info(
        f'clean_ticker_rows: <{ticker}>: shape ={df.shape[0]}; df = \n{df}; \na={a} ; '
        f' r1={r1}; r2={r2}')
    if str(df.iloc[0].iloc[0]).lower().startswith('ticker'):
      logger.info(f'clean_ticker_rows: <{ticker}>:UPDATED df = {df.iloc[2:]}')
      return df.iloc[2:]
  except Exception as e:
    logger.info(f'clean_ticker_rows: <{ticker}>:EXCEPTION df = {df}; a={a}')
    raise e
  return df


def get_ticker_data(ticker, index, dir_to_try_loading_from, reference_date):
  logger.info(f'reference_date = {reference_date}')
  if is_nifty_index(index):
    # get market cap of ticker
    if not ticker.endswith('.NS'):
      ticker = ticker + '.NS'

  yf_ticker_info = get_yf_ticker_info(ticker, index, force_refresh=False)

  logger.info(f'ticker = {ticker}; info={yf_ticker_info}')
  market_cap = yf_ticker_info[
      'marketCap'] if 'marketCap' in yf_ticker_info else 0

  exp_file_path = do_we_have_existing_data(ticker, index,
                                           dir_to_try_loading_from)
  if exp_file_path:
    logger.info(
        f'loading file: {exp_file_path}; end = {reference_date}; type={type(reference_date)}'
    )
    data = pd.read_csv(exp_file_path, index_col=0)
    data = clean_ticker_rows(ticker, data)

    # drop everything after reference_date
    if reference_date:
      data = data.loc[data.index <= str(reference_date)]
  else:
    logger.info(f'About to download data for {ticker}; index={index}')
    data = yf.download(ticker,
                       start='2022-01-01',
                       end=reference_date if reference_date else None,
                       progress=False,
                       multi_level_index=False)
    #multi_level_index=False)
    data.to_csv(os.path.join(dir_to_try_loading_from, f'{ticker}.csv'))

  if data.empty:
    logger.warning(f'Empty data for {ticker}. Returning')
    return yf_ticker_info, data
  # Ensure the index is a DatetimeIndex
  logger.info(f'ticker={ticker}; data = \n{data} ; '
              f'index={data.index}')
  data.index = pd.to_datetime(data.index)
  logger.info(f'ticker={ticker}; ticker={ticker}; data = \n{data}')

  return yf_ticker_info, data


def create_sharpe_ratio_table(indexcap):
  """Create and configure a Rich table for displaying Sharpe ratios."""
  console = Console(file=io.StringIO(), force_terminal=False)
  table = Table(title=f"Average Sharpe Ratios - {indexcap}")

  columns = [("Ticker", "center", "cyan"), ("Name", "left", "cyan"),
             ("12M_Sharpe", "center", "magenta"),
             ("6M_Sharpe", "center", "green"),
             ("3M_Sharpe", "center", "yellow"), ("Average", "center", "blue"),
             ("3M_Beta", "center", "red"), ("AboveSma200", "center", "cyan"),
             ("DailyAvgValue", "center", "magenta"),
             ("MCap", "center", "green"), ("DistFromATH", "center", "yellow"),
             ("Returns1Y", "center", "red"), ("Momo", "center", "cyan")]

  for name, justify, style in columns:
    table.add_column(name, justify=justify, style=style)

  return console, table


def calculate_average_ratios(sharpe_results):
  """
    Calculate weighted average Sharpe ratios handling NaN values.

    The function dynamically adjusts weights based on data availability:
    - If all periods available: 12M (25%), 6M (25%), 3M (50%)
    - If only 6M and 3M available: 6M (35%), 3M (70%)
    - If only 3M available: 3M (100%)
    - If all periods are NaN: Result is NaN

    Args:
        sharpe_results (dict): Dictionary of DataFrames with Sharpe ratios

    Returns:
        dict: Updated sharpe_results with new 'Avg Ratio' column
    """
  # Define the base weight configurations based on availability
  weight_configs = {
      'all': {
          '12M_Sharpe': 0.25,
          '6M_Sharpe': 0.25,
          '3M_Sharpe': 0.50
      },
      'six_three': {
          '6M_Sharpe': 0.35,
          '3M_Sharpe': 0.70
      },
      'three_only': {
          '3M_Sharpe': 1.0
      }
  }

  logger.info(f'iloc: 1: calculate average ratios')
  '''
    TODO: I need to fix all iloc function calls.
    Currently I pick up the latest row automatically to decide everything else.
    But I should ideally use a reference date, so that I can use it to help decide
    what values are good/ranked on that exact date.
    Then I can do a backtest against it.
    '''
  for ticker, [_, data] in sharpe_results.items():
    last_row = data.iloc[-1]

    # Check which periods are available
    has_12m = pd.notna(last_row['12M_Sharpe'])
    has_6m = pd.notna(last_row['6M_Sharpe'])
    has_3m = pd.notna(last_row['3M_Sharpe'])

    # Select appropriate weight configuration
    if has_12m and has_6m and has_3m:
      base_weights = weight_configs['all']
    elif has_6m and has_3m:
      base_weights = weight_configs['six_three']
    elif has_3m:
      base_weights = weight_configs['three_only']
    else:
      # All periods are NaN
      avg_ratio = np.nan
      logger.warning(f"Ticker {ticker}: All Sharpe ratios are NaN")
      data['Avg Ratio'] = avg_ratio
      logger.info(
          f'Ticker {ticker} avg ratio = {avg_ratio:.4f}\n{data.tail(1)}')
      continue

    # Calculate weighted average using selected weights
    available_periods = {
        period: weight
        for period, weight in base_weights.items()
        if pd.notna(last_row[period])
    }

    # Calculate weighted average
    avg_ratio = sum(last_row[period] * weight
                    for period, weight in available_periods.items())

    # Log the weights used
    weight_str = ', '.join(f"{period}: {weight:.2f}"
                           for period, weight in available_periods.items())
    logger.info(
        f"Ticker {ticker}: Using weights {weight_str} "
        f"(missing: {set(weight_configs['all'].keys()) - set(available_periods.keys())})"
    )

    data['Avg Ratio'] = avg_ratio
    logger.info(f'Ticker {ticker} avg ratio = {avg_ratio:.4f}\n{data.tail(1)}')

  return sharpe_results


def create_beta_sorted_list(sharpe_results_sorted):
  logger.info(f'iloc: 4: create beta sorted list')  # 4
  """Create a sorted list of (ticker, data, beta) tuples."""
  return [(ticker, yf_ticker_info, data, data.iloc[-1]["3M_Beta"])
          for ticker, yf_ticker_info, data in sharpe_results_sorted]


def sort_by_beta(ticker_data_beta):
  """Sort instruments by beta value, handling NaN values."""

  def sort_key(x):
    beta_value = x[3]

    return float('-inf') if pd.isna(beta_value) else float(beta_value)

  ticker_data_beta.sort(key=sort_key, reverse=True)
  return ticker_data_beta


def filter_high_beta(ticker_data_beta):
  """Filter out instruments with highest beta values based on total count."""
  total_instruments = len(ticker_data_beta)
  instruments_to_filter = 20 if total_instruments > 30 else 1 if total_instruments > 0 else 0

  filtered_ticker_data = []
  counter = 0
  logger.info(f'iloc: 6: filter high beta')  # 6

  for ticker, yf_ticker_info, data, _ in ticker_data_beta:
    beta_value = data.iloc[-1]['3M_Beta']
    logger.info(f'ABHI: ticker={ticker}; beta={beta_value}')
    if not pd.isna(beta_value) and counter < instruments_to_filter:
      counter += 1
      logger.info(
          f'Ignoring instrument {ticker} with beta value {beta_value:.2f}')
      data['MomoPass'] = False
    filtered_ticker_data.append((ticker, yf_ticker_info, data))

  return filtered_ticker_data


def prepare_table_data(filtered_ticker_data):
  """Prepare formatted data for the Rich table."""
  console_table_data = []
  columns = [
      '12M_Sharpe', '6M_Sharpe', '3M_Sharpe', 'Avg Ratio', '3M_Beta',
      'AboveSma200', 'DailyAvgValue', 'MCap', 'DistFromATH', 'Returns1Y',
      'MomoPass'
  ]
  bool_columns = ['AboveSma200', 'MomoPass']

  logger.info(f'iloc: 7: prepare table data')  # 7

  for ticker, yf_ticker_info, data in filtered_ticker_data:
    try:
      ratios = data.iloc[-1]
      name = ''
      if yf_ticker_info:
        name = (yf_ticker_info.get('longName')
                or yf_ticker_info.get('shortName')
                or yf_ticker_info.get('symbol') or ticker)
      row = [ticker, name] + [
          f"{ratios[col]}" if col in bool_columns else f"{ratios[col]:.2f}"
          for col in columns
      ]
      logger.info(f'prepare table data: row = {row}')
      console_table_data.append(tuple(row))
    except Exception as e:
      logger.error(f'Exception in ticker: {ticker}; Exception={str(e)}; '
                   f'yf_ticker_info:{yf_ticker_info}')
      raise e
  return console_table_data


def retrieve_top_n_momo_tickers(console_table_data, num_of_rows=20):
  top_momo_n_rows = []
  for row in console_table_data:
    if row[-1] == 'True' and len(top_momo_n_rows) < num_of_rows:
      top_momo_n_rows.append(row)
  return top_momo_n_rows


def save_sharpe_data_to_csv(data, output_dir, index_name, reference_date):
  """
    Save Sharpe ratio data to a CSV file with formatted filename.

    Args:
        data (list): List of tuples containing Sharpe ratio data
        output_dir (str): Directory path to save the file
        index_name (str): Name of the index for filename

    Returns:
        str: Path to the saved file
    """
  # Create output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)

  # Generate filename with current date
  date_to_use = datetime.now().strftime(
      '%y%m%d') if not reference_date else reference_date
  filename = f"{date_to_use}-{index_name}.csv"
  filepath = os.path.join(output_dir, filename)

  # Convert data to DataFrame
  columns = [
      'Ticker', 'Name', '12M_Sharpe', '6M_Sharpe', '3M_Sharpe', 'Average',
      '3M_Beta', 'AboveSma200', 'DailyAvgValue', 'MCap', 'DistFromATH',
      'Returns1Y', 'MomoPass'
  ]
  df = pd.DataFrame(data, columns=columns)

  # Save to CSV
  df.to_csv(filepath, index=False)
  logger.info(f"Saved Sharpe ratio data to {filepath}")

  return filepath


def print_sharpe_ratios(indexcap,
                        sharpe_results,
                        output_dir="data/momo-nse",
                        reference_date='',
                        last_date_of_month=None,
                        last_day_of_next_month=None):
  """
    Main function to process, display, and save Sharpe ratios.

    Args:
        indexcap (str): Index name/caption
        sharpe_results (dict): Dictionary containing Sharpe ratio results
        output_dir (str): Directory to save the CSV file (default: "sharpe_ratios")

    Returns:
        dict: Filtered results dictionary
    """
  # Create table
  console, table = create_sharpe_ratio_table(indexcap)

  # Calculate average ratios
  sharpe_results = calculate_average_ratios(sharpe_results)
  # format is [yf_ticker_info, data]

  # Sort by average ratio
  sharpe_results_sorted = sort_dataframes_by_avg_ratio(sharpe_results)
  logger.info(f'sharpe_results_sorted_by_avg_ratio = {sharpe_results_sorted}')

  # Create and sort beta list
  ticker_data_beta = create_beta_sorted_list(sharpe_results_sorted)
  ticker_data_beta = sort_by_beta(ticker_data_beta)
  logger.info(f'ticker_data_beta 1 = {ticker_data_beta}')
  ticker_data_beta = filter_momo_conditions(indexcap, ticker_data_beta)
  logger.info(f'ticker_data_beta 2 = {ticker_data_beta}')
  # Filter high beta instruments
  # filtered_ticker_data = ticker_data_beta
  filtered_ticker_data = filter_high_beta(ticker_data_beta)
  logger.info(f'filtered_ticker_data = {filtered_ticker_data}')
  filtered_ticker_data = sort_dataframes_by_avg_ratio(
      dict([(ticker, [yf_ticker_info, data])
            for ticker, yf_ticker_info, data in filtered_ticker_data]))
  logger.info(f'filtered_ticker_data 3 = {filtered_ticker_data}')

  # Prepare and display table
  console_table_data = prepare_table_data(filtered_ticker_data)
  top_momo_passed_rows = retrieve_top_n_momo_tickers(console_table_data)

  for row in top_momo_passed_rows:
    table.add_row(*row)

  save_sharpe_data_to_csv(console_table_data, output_dir, indexcap,
                          reference_date)

  logger.info(f'Final Sharpe ratios for {indexcap}... ')
  console.print(table)

  results = console.file.getvalue()
  logger.info(results)
  check_monthly_returns_of_top_momo_tickers(indexcap, top_momo_passed_rows,
                                            last_date_of_month,
                                            last_day_of_next_month)

  # Return filtered results
  return {
      ticker: [yf_ticker_info, data]
      for ticker, yf_ticker_info, data in filtered_ticker_data
  }


def check_monthly_returns_of_top_momo_tickers(indexcap,
                                              top_momo_passed_rows,
                                              last_date_of_month=None,
                                              last_day_of_next_month=None):
  if not last_date_of_month:
    logger.info(f'No reference date, ignoring any returns calculation')
    return
  tickers = [x[0] for x in top_momo_passed_rows]
  df = yf.download(tickers,
                   start=last_date_of_month,
                   end=last_day_of_next_month,
                   progress=False)
  df_close = df[HLOC.Col_CLOSE]
  logger.info(f'df = \n{df_close}')
  first_row = df_close.iloc[0]
  last_row = df_close.iloc[-1]
  returns = (last_row - first_row) * 100 / first_row
  logger.info(f'returns = \n{returns}')
  # total returns should only include non-nan values
  total_returns = sum(returns.dropna())
  logger.info(f'indexcap={indexcap}; last_date_of_month={last_date_of_month}; '
              f'total_returns={total_returns}')
  return df


def sort_dataframes_by_avg_ratio(df_dict):
  """
    Sort dataframes based on the last row's 'Avg Ratio' value.

    Args:
        df_dict (dict): Dictionary of dataframes with ticker symbols as keys

    Returns:
        list: Sorted list of (ticker, dataframe) tuples
    """
  # Create a list of (ticker, df, last_avg_ratio) tuples
  logger.info(f'iloc: 3: sort dataframes by avg ratio')  # 3
  df_with_ratios = []
  for ticker, [yf_ticker_info, df] in df_dict.items():
    # Get the last row's Avg Ratio value, handle NaN with a very low number
    last_avg_ratio = df['Avg Ratio'].iloc[-1] if not pd.isna(
        df['Avg Ratio'].iloc[-1]) else float('-inf')
    df_with_ratios.append((ticker, yf_ticker_info, df, last_avg_ratio))

  # Sort the list based on Avg Ratio in descending order
  sorted_dfs = sorted(df_with_ratios, key=lambda x: x[3], reverse=True)

  # Create the final sorted list of (ticker, df) tuples
  sharpe_results_sorted_by_avg_ratio = [
      (ticker, yf_ticker_info, df)
      for ticker, yf_ticker_info, df, _ in sorted_dfs
  ]

  return sharpe_results_sorted_by_avg_ratio


def create_args():
  parser = argparse.ArgumentParser(description="Run momo screener")

  parser.add_argument('--reference-date',
                      '-rd',
                      type=str,
                      default='',
                      help='Reference date to use for the calculations')

  # add arg for "--use-pickle"
  parser.add_argument('--use-pickle',
                      '-p',
                      action='store_true',
                      help='Use pickle file instead of downloading data')

  parser.add_argument(
      "--num-of-tickers",
      default=100,
      type=int,
      help=
      "Comma-separated list of tickers or path to a file containing tickers")

  parser.add_argument('--index',
                      '-i',
                      default=IndexCap.ALL,
                      type=IndexCap,
                      help='What index to use')

  parser.add_argument("--dir",
                      type=str,
                      default='data/nse-bbwp-screener',
                      help="Which directory to use")

  parser.add_argument("--amount",
                      type=float,
                      default=100000,
                      help="Amount to begin with")

  # add an arg for after market order, default being true, which means AMO is true, and false means AMO is false
  parser.add_argument('--dry-run',
                      '-dr',
                      action='store_true',
                      help='Is this a dry run')

  return parser.parse_args()


def momo_calculate_sharpe_ratio(returns, risk_free_rate=0.01):
  """Calculate the Sharpe ratio for a series of returns"""
  excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
  if len(excess_returns) < 2:  # Check if we have enough data
    return np.nan
  return excess_returns.mean() / excess_returns.std() if excess_returns.std(
  ) != 0 else np.nan


def momo_calculate_rolling_sharpe(returns, window_days, risk_free_rate=0.01):
  """Calculate rolling Sharpe ratio for a given window size"""
  excess_returns = returns - risk_free_rate / 252
  rolling_mean = excess_returns.rolling(window=window_days).mean()
  rolling_std = excess_returns.rolling(window=window_days).std()
  return rolling_mean / rolling_std


def momo_calculate_beta(ticker, stock_returns, market_returns):
  """Calculate Beta of a stock against the market

    Args:
        stock_returns (pd.Series): Daily returns of the stock
        market_returns (pd.Series): Daily returns of the market index (NIFTY)

    Returns:
        float: Beta value
    """
  try:
    logger.info(f'ticker={ticker}; stock_returns = \n{stock_returns}')
    logger.info(f'ticker={ticker}; market_returns = \n{market_returns}')
    # Remove any NA values
    clean_data = pd.DataFrame({
        'stock': stock_returns,
        'market': market_returns
    }).dropna()

    if len(clean_data) < 2:
      return np.nan

    # Calculate covariance of stock with market
    covariance = clean_data['stock'].cov(clean_data['market'])
    # Calculate market variance
    market_variance = clean_data['market'].var()

    # Calculate beta
    beta = covariance / market_variance if market_variance != 0 else np.nan

    return beta
  except Exception as e:
    logger.info(f'Excpetion ticker={ticker}; e={e}')
    raise e


def momo_calculate_rolling_beta(stock_returns, market_returns, window_days):
  """Calculate rolling Beta for a given window size

    Args:
        stock_returns (pd.Series): Daily returns of the stock
        market_returns (pd.Series): Daily returns of the market index (NIFTY)
        window_days (int): Window size in days

    Returns:
        pd.Series: Rolling beta values
    """
  # Create a DataFrame with both return series
  returns_df = pd.DataFrame({'stock': stock_returns, 'market': market_returns})

  # Drop rows where either stock or market returns are NaN
  returns_df = returns_df.dropna()

  # Calculate rolling covariance and variance
  rolling_cov = returns_df['stock'].rolling(window=window_days).cov(
      returns_df['market'])
  rolling_market_var = returns_df['market'].rolling(window=window_days).var()

  # Calculate rolling beta
  rolling_beta = rolling_cov / rolling_market_var

  return rolling_beta


def filter_momo_conditions(indexcap, ticker_data_beta):
  """
    Filter instruments based on momentum conditions using the latest data point.

    Args:
        ticker_data_beta: List of tuples (ticker, data, beta)
    Returns:
        List of filtered tuples with updated Momo column
    """
  filtered_ticker_data = []
  logger.info(f'iloc: 5: filter momo conditions')  # 5
  is_us_index = 'us' in indexcap.lower()
  mcap_filter_by_index = {
      IndexCap.US_MICRO: 10 * 10**7,  # 100 million
      IndexCap.US_SMALL: 100 * 10**7,  # 1 billion
      IndexCap.US_MID: 500 * 10**7,  # 5 billion,
      IndexCap.MICRO_CAP: 550 * 10**7,  # 100 million,
      IndexCap.ALL: 550 * 10**7,  # 100 million,
      IndexCap.SMALL_CAP: 650 * 10**7,  # 1 billion,
      IndexCap.MID_CAP: 850 * 10**7  # 5 billion
  }
  # daily avg value, to be 10% of mcap

  for ticker, yf_ticker_info, data, beta in ticker_data_beta:
    # Get the last row for comparisons
    last_row = data.iloc[-1]
    logger.info(f'Ticker = {ticker}')

    # Initialize Momo column with AboveSma200 value from last row
    data['MomoPass'] = data['AboveSma200']

    # Check conditions using the last row's values
    fail_conditions = []

    # 10 * 10**7:  # Note: Changed ^ to * 10**7 for proper exponentiation
    if last_row['DailyAvgValue'] < mcap_filter_by_index[indexcap] * 0.015:
      fail_conditions.append('DailyAvgValue')
      data.loc[:, 'MomoPass'] = False

    if last_row['MCap'] < mcap_filter_by_index[
        indexcap]:  # Fixed exponentiation
      fail_conditions.append('MCap')
      data.loc[:, 'MomoPass'] = False

    if last_row['DistFromATH'] < -0.25:
      fail_conditions.append('DistFromATH')
      data.loc[:, 'MomoPass'] = False

    if last_row['Returns1Y'] < 0.065:
      fail_conditions.append('Returns1Y')
      data.loc[:, 'MomoPass'] = False

    # Log failed conditions if any
    if fail_conditions:
      conditions_str = ', '.join(fail_conditions)
      logger.info(
          f'{ticker} -> Failed filters: <{conditions_str}> with values:\n'
          f'\t DailyAvgValue: {last_row["DailyAvgValue"]:.1f}\n'
          f'\t MCap: {last_row["MCap"]:.1f}\n'
          f'\t DistFromATH: {last_row["DistFromATH"]:.2f}\n'
          f'\t Returns1Y: {last_row["Returns1Y"]:.2f}')
    else:
      logger.info(f'{ticker} -> PASSED filters with values:\n'
                  f'\t DailyAvgValue: {last_row["DailyAvgValue"]:.1f}\n'
                  f'\t MCap: {last_row["MCap"]:.1f}\n'
                  f'\t DistFromATH: {last_row["DistFromATH"]:.2f}\n'
                  f'\t Returns1Y: {last_row["Returns1Y"]:.2f}')

    filtered_ticker_data.append((ticker, yf_ticker_info, data, beta))

  return filtered_ticker_data


def momo_add_stats(data, yf_ticker_info, nifty_data=None):
  """Add daily returns, rolling Sharpe ratios, and Beta to the dataframe"""
  # Calculate daily returns
  data['Returns'] = data[HLOC.Col_CLOSE].pct_change()

  # calculate returns of one year
  data['Returns1Y'] = data[HLOC.Col_CLOSE].pct_change(periods=252)
  ticker = yf_ticker_info['symbol']
  impliedSharesOutstanding = 1
  if 'impliedSharesOutstanding' not in yf_ticker_info:
    if 'sharesOutstanding' in yf_ticker_info:
      impliedSharesOutstanding = yf_ticker_info['sharesOutstanding']
    else:
      logger.warning(f'impliedSharesOutstanding not found for {ticker}')
  else:
    impliedSharesOutstanding = yf_ticker_info['impliedSharesOutstanding']

  logger.info(
      f'Ticker: {ticker}; impliedSharesOutstanding = {impliedSharesOutstanding}'
  )
  data['DailyValue'] = data[HLOC.Col_CLOSE] * impliedSharesOutstanding

  # daily average value over the last 21 days
  data['DailyAvgValue'] = data['DailyValue'].rolling(window=21).mean()

  # Calculate daily returns
  data['Sma200'] = data[HLOC.Col_CLOSE].rolling(window=200).mean()

  # calculate market cap
  data['MCap'] = yf_ticker_info[
      'marketCap'] if 'marketCap' in yf_ticker_info else 0

  # Calculate all time high of stock
  data['ATH'] = data[HLOC.Col_CLOSE].max()

  # calculate distance from all time high, in %
  data['DistFromATH'] = (data[HLOC.Col_CLOSE] - data['ATH']) / data['ATH']
  logger.info(
      f'momo_add_stats: Ticker: {ticker}; DistFromATH={data["DistFromATH"]}')
  logger.info(f'momo_add_stats: Ticker: {ticker}; ATH={data["ATH"]}')

  # is the price above sma200
  data['AboveSma200'] = data[HLOC.Col_CLOSE] > data['Sma200']

  # Calculate rolling Sharpe ratios for different periods
  periods_days = {
      '12M_Sharpe': 252,  # 12 months ≈ 252 trading days
      '6M_Sharpe': 126,  # 6 months ≈ 126 trading days
      '3M_Sharpe': 63  # 3 months ≈ 63 trading days
  }

  for col_name, window in periods_days.items():
    data[col_name] = momo_calculate_rolling_sharpe(data['Returns'], window)

  # Calculate Beta if NIFTY data is provided
  if nifty_data is not None:
    # Calculate NIFTY returns
    nifty_returns = nifty_data[HLOC.Col_CLOSE].pct_change()

    logger.info(f'nifty_returns = {nifty_returns}; type={type(nifty_returns)}')
    stock_returns = data['Returns']
    logger.info(f'stock_returns = {stock_returns}; type={type(stock_returns)}')
    # Calculate point-in-time Beta
    data['Beta'] = momo_calculate_beta(ticker,
                                       stock_returns.reset_index(drop=True),
                                       nifty_returns.reset_index(drop=True))

    # Calculate rolling Beta for different periods
    for period, days in periods_days.items():
      period_name = period.replace('Sharpe', 'Beta')
      data[period_name] = momo_calculate_rolling_beta(data['Returns'],
                                                      nifty_returns, days)

  return data


def momo_process_single_ticker(
    ticker_with_index_with_dir_with_nifty_data_with_reference_date,
    periods=[12, 6, 3]):
  """Process a single ticker to calculate Sharpe ratios and Beta"""
  # Unpack the ticker and directory
  ticker, index, dir, nifty_data, reference_date = ticker_with_index_with_dir_with_nifty_data_with_reference_date
  # Download historical stock data
  yf_ticker_info, data = get_ticker_data(ticker, index, dir, reference_date)
  logger.info(f'ticker={ticker}; data =\n{data}')

  if data.empty:
    logger.warning(f'ticker: {ticker}; data is empty')
    return ticker, None, {}, pd.DataFrame()

  # Add all stats including rolling Sharpe ratios and Beta
  data = momo_add_stats(data, yf_ticker_info, nifty_data)
  logger.info(f'After momo stats. ticker = {ticker}; data = \n{data}')

  # Calculate point-in-time metrics for the last date
  metrics = {}
  for period in periods:
    # Get the last 'n' months of data
    end_date = data.index[-1]
    start_date = end_date - pd.DateOffset(months=period)

    # Filter the DataFrame using .loc
    period_data = data.loc[start_date:end_date]
    metrics[f'{period}M_Sharpe'] = momo_calculate_sharpe_ratio(
        period_data['Returns'])

    # Calculate Beta if NIFTY data is provided
    if nifty_data is not None:
      nifty_period_data = nifty_data.loc[start_date:end_date][
          HLOC.Col_CLOSE].pct_change()
      metrics[f'{period}M_Beta'] = momo_calculate_beta(ticker,
                                                       period_data['Returns'],
                                                       nifty_period_data)

  logger.info(f'ticker: {ticker}; metrics = {metrics}; data = \n{data}')

  return ticker, yf_ticker_info, metrics, data


def last_day_of_month_calendar(year_month):
  """
    Calculate last day using calendar module
    """
  if not year_month:
    return ''
  year_month = int(year_month)
  year = year_month // 100
  month = year_month % 100
  return date(year, month, calendar.monthrange(year, month)[1])


def last_day_of_next_month_calendar(year_month):
  """
    Calculate the last day of the next month from a YYYYMM format input

    Args:
        year_month (int): Year and month in format 202410

    Returns:
        date: Last date of the next month
    """
  year_month = int(year_month)
  year = year_month // 100
  month = year_month % 100

  # Handle December edge case
  if month == 12:
    next_year = year + 1
    next_month = 1
  else:
    next_year = year
    next_month = month + 1

  return date(next_year, next_month,
              calendar.monthrange(next_year, next_month)[1])


def get_sharpe_ratios_parallel(tickers,
                               index,
                               dir,
                               reference_date,
                               num_processes=None):
  """
    Calculate Sharpe ratios for multiple tickers in parallel

    Parameters:
    tickers (list): List of stock tickers
    dir (str): Directory path for data files
    num_processes (int): Number of processes to use. Defaults to CPU count - 1

    Returns:
    dict: Dictionary of Sharpe ratios for each ticker
    """
  # If num_processes not specified, use CPU count - 1 (leaving one core free)
  ref_index = '^SPX' if index.startswith('us') else '^NSEI'
  ref_data = yf.download(ref_index,
                         period='max',
                         progress=False,
                         multi_level_index=False,
                         end=reference_date if reference_date else None)
  # multi_level_index=False)
  logger.info(f'ref_data = {ref_data}')

  if num_processes is None:
    num_processes = max(1, cpu_count() - 1)

  # Create ticker-directory pairs
  ticker_dir_pairs = [(ticker, index, dir, ref_data, reference_date)
                      for ticker in tickers]

  # Create a process pool
  with Pool(processes=num_processes) as pool:
    # Map the process_single_ticker function to all tickers with their directory
    results = pool.map(momo_process_single_ticker, ticker_dir_pairs)

  logger.info(f'results = {results}')
  # print all results
  for ticker, _, ratios, data in results:
    logger.info(f'Result: {ticker} = {ratios}; data = \n{data}')

  # Convert results list to dictionary
  return {
      ticker: [yf_ticker_info, data]
      for ticker, yf_ticker_info, ratios, data in results if ratios
  }


def main():
  args = create_args()
  logger.info(f'args = {args}')

  # Get Sharpe ratios for the specified tickers
  tickers = get_tickers(index=args.index,
                        dry_run=args.dry_run,
                        num_to_pick=args.num_of_tickers)

  last_date_of_month = last_day_of_month_calendar(
      args.reference_date) if args.reference_date else ''
  last_day_of_next_month = last_day_of_next_month_calendar(
      args.reference_date) if args.reference_date else ''
  logger.info(
      f'last_date_of_month={last_date_of_month}; type={type(last_date_of_month)}'
  )
  logger.info(
      f'last_day_of_next_month={last_day_of_next_month}; type={type(last_day_of_next_month)}'
  )

  # if sharpe_results.pickle exists, load from there
  if args.use_pickle and os.path.exists('sharpe_results.pickle'):
    logger.info('sharpe_results.pickle exists, loading from there')
    sharpe_results = pickle.load(open('sharpe_results.pickle', 'rb'))
  else:
    logger.info('sharpe_results.pickle does not exist, calculating')

    sharpe_results = get_sharpe_ratios_parallel(tickers, args.index, args.dir,
                                                last_date_of_month)

    # save sharpe_results to pickle
    pickle.dump(sharpe_results, open('sharpe_results.pickle', 'wb'))

  logger.info(f'sharpe_results = {sharpe_results}')

  output_dir = 'data/momo-us' if args.index in [
      IndexCap.US_MICRO, IndexCap.US_SMALL, IndexCap.US_MID
  ] else 'data/momo-nse'

  print_sharpe_ratios(args.index, sharpe_results, output_dir,
                      args.reference_date, last_date_of_month,
                      last_day_of_next_month)


if __name__ == '__main__':
  main()
