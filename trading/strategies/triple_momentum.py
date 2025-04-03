# %%

from trading.common import utils

logger = utils.get_logger('dual-momentum', use_rich=True, should_add_ts=True)

import os, sys, json, argparse
import subprocess
import re
import pandas as pd
import time
import datetime
import plotly
import plotly.express as px
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from typing import Tuple, Dict

from trading.services import telegram_runner
'''

Usage

python trading/strategies/triple_momentum.py \
  --main-col NIFTY \
  --main-ticker ^NSEI \
  --alt-col USDINR \
  --alt-ticker USDINR=X \
  --output-dir data/triple-momentum \
  --initial-capital 100000


python trading/strategies/triple_momentum.py \
  --main-col INDA \
  --main-ticker INDA \
  --alt-col GOLD \
  --alt-ticker GC=F

'''


def parse_arguments() -> argparse.Namespace:
  """Parse command line arguments."""
  parser = argparse.ArgumentParser(
      description='Dual Momentum Trading Strategy')

  # what is the main ticker

  parser.add_argument('--main-col',
                      type=str,
                      default='NIFTY',
                      help='Column name for NIFTY data')
  parser.add_argument('--main-ticker',
                      type=str,
                      default='^NSEI',
                      help='Main ticker symbol')

  # what is the alternate ticker
  parser.add_argument('--alt-ticker',
                      type=str,
                      default='USDINR=X',
                      help='Alt ticker symbol')

  # what is the third ticker
  parser.add_argument('--third-ticker',
                      type=str,
                      default='GLD',
                      help='Third ticker symbol')

  parser.add_argument('--alt-col',
                      type=str,
                      default='USDINR',
                      help='Column name for alt data')

  parser.add_argument('--third-col',
                      type=str,
                      default='GLD',
                      help='Column name for third data')

  parser.add_argument('--initial-capital',
                      type=float,
                      default=100000,
                      help='Initial capital amount')

  parser.add_argument('--output-dir',
                      type=str,
                      required=False,
                      default='data/dual-momentum',
                      help='Directory to store output data')

  # add an arg for notifications
  parser.add_argument('--notify',
                      action='store_true',
                      help='Send notifications')

  return parser.parse_args()


def run_cmd(cmds):
  proc = subprocess.Popen(
      cmds,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
  )
  stdout, stderr = proc.communicate(timeout=1000)
  if stderr:
    stderr_s = stderr.decode('utf-8').split('\n')
    for s in stderr_s:
      logger.info(f'stderr = {s}')
  return stdout.decode('utf-8')


def read_data(ticker):
  df = yf.download(ticker,
                   progress=False,
                   period='5y',
                   multi_level_index=False)
  logger.info(f'df = \n{df}')
  return df


def setup_directory(output_dir: str) -> None:
  """Create output directory if it doesn't exist."""
  os.makedirs(output_dir, exist_ok=True)


def calc_returns(df, suffix=''):
  """Calculate returns and normalized returns metrics with optional suffix."""
  df = df.copy()  # Ensure we don't modify the original dataframe

  # Add suffix to all columns except the original price columns
  rename_dict = {col: f"{col}{suffix}" for col in df.columns}
  df = df.rename(columns=rename_dict)

  returns_col = f'returns{suffix}'
  mean_col = f'mean{suffix}'
  std_col = f'std{suffix}'
  norm_returns_col = f'norm_returns{suffix}'

  logger.info(f'df = \n{df}')
  logger.info(f'df columns found = {df.columns}; suffix={suffix}')

  df[returns_col] = df[f'Close{suffix}'].pct_change(21)
  df[mean_col] = df[returns_col].rolling(4).mean()
  df[std_col] = df[mean_col].rolling(4).std()
  df[norm_returns_col] = df[mean_col] / df[std_col]

  return df


def merge_three_dataframes(df_first: pd.DataFrame, df_second: pd.DataFrame,
                           df_third: pd.DataFrame) -> pd.DataFrame:
  """Merge three dataframes with appropriate suffixes."""
  # Calculate returns for each dataframe with appropriate suffixes
  df_first = calc_returns(df_first, '_first')
  df_second = calc_returns(df_second, '_second')
  df_third = calc_returns(df_third, '_third')

  # First merge
  df_merged = pd.merge(df_first,
                       df_second,
                       left_index=True,
                       right_index=True,
                       how='left')

  # Second merge
  df_merged = pd.merge(df_merged,
                       df_third,
                       left_index=True,
                       right_index=True,
                       how='left')

  return df_merged.iloc[::-1]  # Reverse order


def clean_merged_data(df_merged: pd.DataFrame) -> pd.DataFrame:
  """Clean and prepare merged dataframe for three assets."""
  # Create aligned series for comparisons
  norm_first = df_merged['norm_returns_first']
  norm_second = df_merged['norm_returns_second']
  norm_third = df_merged['norm_returns_third']

  # Calculate highest returns using aligned comparisons
  df_merged['norm_returns_first_is_highest'] = ((norm_first > norm_second) &
                                                (norm_first > norm_third))

  df_merged['norm_returns_second_is_highest'] = ((norm_second > norm_first) &
                                                 (norm_second > norm_third))

  df_merged['norm_returns_third_is_highest'] = ((norm_third > norm_first) &
                                                (norm_third > norm_second))

  # Drop unnecessary columns
  columns_to_drop = []
  for suffix in ['_first', '_second', '_third']:
    columns_to_drop.extend(
        [f'{s}{suffix}' for s in ['Open', 'High', 'Low', 'Volume']])

  return df_merged.drop(columns_to_drop, axis=1)


def get_parsed_dataframe(ticker, suffix=''):
  """Get dataframe with calculated returns using specified suffix."""
  df = read_data(ticker)
  df = calc_returns(df, suffix)
  return df


def process_status_changes(df: pd.DataFrame,
                           col_mappings: Dict[str, str]) -> pd.DataFrame:
  """Process status changes for three assets."""
  # Detect any change in position
  df['status_changed'] = (df.norm_returns_first_is_highest.ne(
      df.norm_returns_first_is_highest.shift())
                          | df.norm_returns_second_is_highest.ne(
                              df.norm_returns_second_is_highest.shift())
                          | df.norm_returns_third_is_highest.ne(
                              df.norm_returns_third_is_highest.shift()))

  d_filtered = df[df['status_changed']].iloc[::-1].round(2)

  columns_to_drop = [
      'returns_first',
      'mean_first',
      'std_first',
      'returns_second',
      'mean_second',
      'std_second',
      'returns_third',
      'mean_third',
      'std_third',
      # 'Adj Close_first', 'Adj Close_second', 'Adj Close_third'
  ]
  d_filtered.drop(columns_to_drop, axis=1, inplace=True)

  # Rename columns
  d_filtered.rename(columns=col_mappings, inplace=True)

  # Calculate previous values
  d_filtered['prev_first'] = d_filtered.shift(1)[col_mappings['Close_first']]
  d_filtered['prev_second'] = d_filtered.shift(1)[col_mappings['Close_second']]
  d_filtered['prev_third'] = d_filtered.shift(1)[col_mappings['Close_third']]
  col_names = d_filtered.columns.tolist()
  logger.info(f'd_filtered cols = \n{col_names}')

  return d_filtered


def calculate_positions_and_pl(data: pd.DataFrame, initial_capital: float,
                               first_col: str, second_col: str,
                               third_col: str) -> pd.DataFrame:
  """Calculate positions and profit/loss for three assets."""
  data = data.copy()
  logger.info(f'data = \n{data}')
  # Calculate positions based on highest normalized returns
  data['first_position'] = data['norm_returns_first_is_highest'].astype(int)
  data['second_position'] = data['norm_returns_second_is_highest'].astype(int)
  data['third_position'] = data['norm_returns_third_is_highest'].astype(int)

  # Calculate returns for each asset
  data['first_returns'] = data[first_col].pct_change() * 100
  data['second_returns'] = data[second_col].pct_change() * 100
  data['third_returns'] = data[third_col].pct_change() * 100

  # Calculate P/L for each asset
  for position, col, pl_name in [('first_position', first_col, 'first_pl'),
                                 ('second_position', second_col, 'second_pl'),
                                 ('third_position', third_col, 'third_pl')]:
    data[pl_name] = ((data[col] - data[col].shift(1)) * initial_capital *
                     data[position] / data[col].shift(1))

  # Apply P/L limits (adjust these as needed)
  data["first_pl"] = data["first_pl"].apply(lambda x: max(x, -5000))
  data["second_pl"] = data["second_pl"].apply(lambda x: max(x, -1000))
  data["third_pl"] = data["third_pl"].apply(lambda x: max(x, -1000))

  # Calculate cumulative P/L
  data['cumulative_first_pl'] = data['first_pl'].cumsum()
  data['cumulative_second_pl'] = data['second_pl'].cumsum()
  data['cumulative_third_pl'] = data['third_pl'].cumsum()
  data["total_pl"] = (data["cumulative_first_pl"] +
                      data["cumulative_second_pl"] +
                      data["cumulative_third_pl"])

  return data


def calculate_portfolio_metrics(df: pd.DataFrame,
                                initial_capital: float) -> pd.DataFrame:
  """Calculate portfolio metrics for three assets."""
  df = df.copy()

  # Set positions based on previous period's signals
  df['first_position'] = df['norm_returns_first_is_highest'].shift(1)
  df['second_position'] = df['norm_returns_second_is_highest'].shift(1)
  df['third_position'] = df['norm_returns_third_is_highest'].shift(1)

  # Calculate returns for each asset
  df['first_returns'] = df['Close_first'].pct_change()
  df['second_returns'] = df['Close_second'].pct_change()
  df['third_returns'] = df['Close_third'].pct_change()

  # Calculate strategy returns (only one position will be active at a time)
  df['strategy_returns'] = (
      df['first_position'].shift(1) * df['first_returns'] +
      df['second_position'].shift(1) * df['second_returns'] +
      df['third_position'].shift(1) * df['third_returns'])

  df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
  df['portfolio_value'] = initial_capital * df['cumulative_returns']

  return df


def main():
  # Parse arguments
  args = parse_arguments()
  setup_directory(args.output_dir)

  # Get data (assuming get_parsed_dataframe is defined elsewhere)
  df_main = get_parsed_dataframe(args.main_ticker)
  df_alt = get_parsed_dataframe(args.alt_ticker)
  df_third = get_parsed_dataframe(args.third_ticker)

  # Process data
  df_merged = merge_three_dataframes(df_main, df_alt, df_third)

  # Clean and process the data
  col_mappings = {
      'Close_first': args.main_col,
      'Close_second': args.alt_col,
      'Close_third': args.third_col
  }

  df_merged = clean_merged_data(df_merged)

  # col_mappings = {'Close_main': args.main_col, 'Close_alt': args.alt_col}

  d_filtered = process_status_changes(df_merged, col_mappings)

  # Calculate positions and P/L
  data = calculate_positions_and_pl(d_filtered, args.initial_capital,
                                    args.main_col, args.alt_col,
                                    args.third_col)

  # Calculate portfolio metrics
  portfolio_data = calculate_portfolio_metrics(df_merged, args.initial_capital)

  # Prepare and log results
  logger.info(f'Filtered data = \n{data.iloc[-20:].round(2)}')
  logger.info(f'Portfolio data = \n{portfolio_data.iloc[:10]}')
  # sample_results(data)
  # Prepare telegram update
  telegram_text = prepare_telegram_update(args, data)
  logger.info(telegram_text)

  if args.notify:
    telegram_runner.send_text([telegram_text])
  else:
    logger.info('Notifications disabled')


def prepare_telegram_update(args, data: pd.DataFrame) -> str:
  """Prepare text for Telegram update."""

  # remove rows containing NaN values
  data = data.dropna()

  # Select the last 10 rows and the relevant columns
  final = data[[
      args.main_col, args.alt_col, args.third_col,
      'norm_returns_first_is_highest', 'norm_returns_second_is_highest',
      'norm_returns_third_is_highest'
  ]][-10:].round(2)

  # Create a BUY column that shows which asset to buy
  final['BUY'] = None
  final.loc[final['norm_returns_first_is_highest'], 'BUY'] = args.main_col
  final.loc[final['norm_returns_second_is_highest'], 'BUY'] = args.alt_col
  final.loc[final['norm_returns_third_is_highest'], 'BUY'] = args.third_col

  # Drop the boolean columns
  final = final.drop([
      'norm_returns_first_is_highest', 'norm_returns_second_is_highest',
      'norm_returns_third_is_highest'
  ],
                     axis=1)

  return f'''Triple Momentum = \n```\n{final}\n```'''


if __name__ == '__main__':
  main()
