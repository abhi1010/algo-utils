import os, pandas
from enum import Enum
import argparse
from datetime import date, datetime
import json
import ast
import glob
from pprint import pformat

from trading.common import utils

logger = utils.get_logger('bbwp')

import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.stats import percentileofscore

from trading.screener.screener_common import Markets
from trading.services import telegram_runner


def _send_msg(text):
  logger.info(f'Sending msg to telegram: {text}')
  telegram_runner.send_text([text])


def load_clean_csv(filepath):
  """
    Load a CSV file while removing metadata rows and handling duplicate headers.
    Renames 'Price' column to 'Date' if ticker metadata is present.

    Args:
        filepath (str): Path to the CSV file

    Returns:
        pd.DataFrame: Cleaned DataFrame with metadata rows removed
    """
  # Read the first few rows to check for metadata
  preview_df = pd.read_csv(filepath, nrows=5)
  has_ticker_metadata = preview_df.astype(str).apply(
      lambda x: x.str.contains('Ticker', na=False)).any().any()

  # Read the full file without headers first
  df = pd.read_csv(filepath, skiprows=[1, 2])
  columns_list = df.columns.to_list()
  logger.info(f'df.columns = {columns_list}')

  # Get the true header from the first row
  # header = list(df.iloc[0])
  if has_ticker_metadata:
    columns_list[0] = 'Date'
  df.columns = columns_list
  logger.info(f'previw df = \n{preview_df}')
  logger.info(f'has_ticker_metadata = {has_ticker_metadata}')
  logger.info(f'header = {columns_list}')
  logger.info(f'df = \n{df}')

  logger.info(f'filepath={filepath}; df = \n{df}')

  return df


def process_files(directory, limit=0, make_chart=False, only_one=False):
  dataframes = {}
  # # Get list of CSV files in the directory
  if only_one:
    csv_files = [
        file for file in os.listdir(directory)
        if file.endswith('.csv') and 'VGUARD' in file
    ]
  else:
    # Get list of CSV files in the directory
    csv_files = [
        file for file in os.listdir(directory) if file.endswith('.csv')
    ]
  if limit:
    csv_files = csv_files[:limit]

  logger.info(f'directory = {directory}')
  logger.info(f'csv_files = {csv_files} ; len = {len(csv_files)}')

  for filename in csv_files:
    # logger.info(filename)
    symbol = filename.split(".")[0]

    # df = pd.read_csv(os.path.join(directory, filename))

    df = load_clean_csv(os.path.join(directory, filename))
    # only keep data from 2024 jan 01 onwards
    # df = df[df['Date'] >= '2024-01-01']

    # logger.info(f'Read : {symbol}')
    if df.empty:
      continue

    dataframes[symbol] = df

  return dataframes


def filter_away_stale_data(dataframes, market, timeframe):
  date_fmt = '%Y-%m-%d %H:%M:%S' if market == Markets.CRYPTO and timeframe != '1d' else '%Y-%m-%d'

  def get_largest_dt(dataframes):
    date_s = '<Unknown>'
    largest_dt = datetime.min
    # 2023-04-04 20:00:00 or 2023-04-04

    for symbol, df in dataframes.items():
      logger.info(f'Processing symbol: {symbol}; df=\n{df}')
      try:
        date_s = df['Date'].iloc[-1]
        date_dt = datetime.strptime(date_s, date_fmt)
      except Exception as e:
        logger.info(f'exception: {str(e)}; sym: {symbol}; date_s = {date_s}')
        logger.info(f'Exception during symbol: {symbol}; df=\n{df}')
        continue
        raise e
      largest_dt = max(largest_dt, date_dt)
      logger.info(f'Sym: {symbol}; date_s = {date_dt}')

    return largest_dt

  largest_date = get_largest_dt(dataframes)
  logger.info(f'largest_date = {largest_date}')
  only_latest_dataframes = {}

  for symbol, df in dataframes.items():
    date_s = df['Date'].iloc[-1]
    date_dt = datetime.strptime(date_s, date_fmt)
    if date_dt >= largest_date:
      only_latest_dataframes[symbol] = df
    else:
      logger.info(f'Sym: {symbol} is too old, skipping; '
                  f'date_s = {date_s}; largest_date = {largest_date}')
  return only_latest_dataframes, largest_date


# Function to calculate the percentile
def rolling_percentile(series, window):
  # a = np.percentile(series, 50)
  a = series.rolling(window).rank(pct=True)
  return a
  # return series.rolling(
  #     window=window).apply(lambda x: percentileofscore(x, x[-1]))


def run_bb_stats(dataframes, window, threshold=0.1):
  cols_to_copy = ['BBL_20_2.0', 'BBU_20_2.0', 'width', 'BBB_20_2.0_percentile']
  low_vol_ticker_names = []
  for symbol, df in dataframes.items():
    # logger.info(f'Sym: {symbol}; init = \n{df}; shape={df.shape}')
    if df.shape[0] < window or df.shape[1] < 5:
      logger.info(f'Sym: {symbol} shape is too small, skipping: {df.shape}')
      continue

    sma_50 = ta.sma(df['Close'], length=50)
    # Calculate the Bollinger Bands
    sym_bb = ta.bbands(df['Close'], length=20, std=2)
    # logger.info(f'Sym: {symbol}; BB = \n{sym_bb}')

    # Calculate the 50-period rolling percentile for BBB_20_2.0
    sym_bb['width'] = sym_bb['BBU_20_2.0'] - sym_bb['BBL_20_2.0']
    sym_bb['BBB_20_2.0_percentile'] = rolling_percentile(
        sym_bb['width'], window)
    last_30_percentile = sym_bb.iloc[-30:]['BBB_20_2.0_percentile'].to_list()

    if sym_bb['BBB_20_2.0_percentile'].iloc[-1] <= threshold and df[
        'Close'].iloc[-1] >= sma_50.iloc[-1] * 0.97:
      df[cols_to_copy] = sym_bb[cols_to_copy]

      logger.info(f'Appending Sym: {symbol}; window={window}; '
                  f'sma={sma_50.iloc[-1]}; close={df["Close"].iloc[-1]}'
                  f'BB = \n{df.iloc[-30:].to_string()}; ')
      low_vol_ticker_names.append(symbol)
    # Copy the columns
    # df[cols_to_copy] = sym_bb[cols_to_copy]
  return low_vol_ticker_names


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--dir",
                      default='data/crypto/1h/csv',
                      help="Which directory to save to",
                      type=str)

  # arg called output_dir, default to be data/bbwp
  parser.add_argument('--output-dir', '-o', default='data/bbwp', type=str)

  # add tag as BBWP screener
  parser.add_argument('--tag', default='BBWP_screener', type=str)

  parser.add_argument('--timeframe',
                      '-t',
                      default='1h',
                      type=str,
                      help='What timeframe/interval to use')

  # add arg called window, with a default of 20
  parser.add_argument('--window', '-w', default=100, type=int, help='Window')

  parser.add_argument('--market',
                      type=Markets,
                      default=Markets.CRYPTO,
                      help='What market')

  parser.add_argument(
      "--pairs-file",
      default='',
      help=
      "Do we have a ready made list of pairs? If empty, we download everything",
      type=str)

  # add arg for showing charts
  parser.add_argument('--charts', action='store_true', help='Show charts')

  args = parser.parse_args()
  return args


def save_names_to_txt(names, output_dir, market, timeframe, largest_date):
  # file path in the format of data/bbwp/YYYY-MM-DD_tickers.txt
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  # output_file to be date based YYYY-MM-DD_tickers.txt
  # today_s = date.today().strftime("%Y-%m-%d")
  largest_dt_s = largest_date.strftime("%Y-%m-%d_%H-%M")
  output_file = os.path.join(
      output_dir, f'{largest_dt_s}_{market}-{timeframe}-tickers.txt')
  logger.info(f'output_file = {output_file}')
  with open(output_file, 'w') as f:
    f.write(str(names))
  logger.info(f'written into {output_file}')


def get_new_tickers_compared_to_older_file(names, output_dir, market,
                                           timeframe, largest_date):
  names = sorted(names)

  largest_dt_s = largest_date.strftime("%Y-%m-%d_%H-%M")
  file_name_fmt = f'{market}-{timeframe}-tickers.txt'
  # find the latest 2 files with the given format
  # use glob to find latest 2 files with the file_name_fmt format
  files = glob.glob(os.path.join(output_dir, f'*{file_name_fmt}'))

  if len(files) == 0:
    return names
  files.sort(key=os.path.getmtime)
  logger.info(f'files = {files}; file_name_fmt = {file_name_fmt}')
  if len(files) < 2:
    logger.info(f'Not enough files found for {file_name_fmt}')
    return []
  last_file = files[-2]

  # parse teh file into a list of tickers
  with open(last_file, 'r') as f:
    file_contents = f.read().splitlines()
    # file content it like this: "['KUCOIN:QIUSDT', 'KUCOIN:ARUSDT']". Parse it into a list
    logger.info(f'last_file={last_file}; file_contents = {file_contents}')

    last_file_names = sorted(ast.literal_eval(file_contents[0]))
  logger.info(f'last_file = {last_file}')
  logger.info(f'last_file_names = {last_file_names}')
  new_tickers = list(set(names) - set(last_file_names))
  logger.info(f'new_tickers = {new_tickers}')
  return new_tickers


def transform_tickers(tickers, market):
  if market in [Markets.NIFTY, Markets.SPX, Markets.NASDAQ]:
    return tickers
  kucoin_tickers = [
      'KUCOIN:' + item.split('-')[0].replace('_', '').replace('/', '')
      for item in tickers
  ]
  return kucoin_tickers


def send_telegram_msg(args, tickers_for_telegram, new_tickers, market):
  tickers_for_telegram = sorted(tickers_for_telegram)
  new_tickers = sorted(new_tickers)
  msg = ''
  if len(tickers_for_telegram) == 0:
    msg = f"Args={args} \n No tickers right now"
    return

  # Convert Namespace to dictionary
  namespace_dict = vars(args)
  # Pretty print the dictionary into a string
  namespace_str = pformat(namespace_dict)
  prefix_joiner = ',NSE.' if market == Markets.NIFTY else ','

  if market == Markets.NIFTY:
    prefix = 'NSE:'
    new_tickers_s = (prefix +
                     (',' + prefix).join(new_tickers).replace('&', '_')
                     ) if len(new_tickers) else ''
    all_tickers_s = (prefix +
                     (',' + prefix).join(tickers_for_telegram).replace(
                         '&', '_')) if len(tickers_for_telegram) else ''
  else:
    new_tickers_s = ','.join(new_tickers)
    all_tickers_s = ','.join(tickers_for_telegram)

  msg = f'''Args:
  ```python
{namespace_str}```

  New tickers:
  ```
{new_tickers_s}```

  All tickers:
  ```
{all_tickers_s}```'''

  _send_msg(msg)


if __name__ == '__main__':
  args = get_args()
  logger.info(f'args = {args}')
  utils.set_pandas_options()

  files_df = process_files(args.dir, make_chart=args.charts)
  dfs, largest_date = filter_away_stale_data(files_df, args.market,
                                             args.timeframe)
  low_vol_ticker_names = run_bb_stats(dfs, window=args.window)
  logger.info(f'low_vol_ticker_names = {low_vol_ticker_names}')

  tickers_for_tradingview = transform_tickers(low_vol_ticker_names,
                                              args.market)
  logger.info(f'tickers_for_tradingview = {tickers_for_tradingview}')

  save_names_to_txt(tickers_for_tradingview, args.output_dir, args.market,
                    args.timeframe, largest_date)
  new_tickers = get_new_tickers_compared_to_older_file(tickers_for_tradingview,
                                                       args.output_dir,
                                                       args.market,
                                                       args.timeframe,
                                                       largest_date)
  logger.info(f'new_tickers = {new_tickers}')
  send_telegram_msg(args, tickers_for_tradingview, new_tickers, args.market)
