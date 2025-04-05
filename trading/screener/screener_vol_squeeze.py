import os, pandas
from enum import Enum
import argparse
import concurrent.futures

from trading.common import utils

logger = utils.get_logger('screener_vol_squeeze')
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from trading.screener.screener_common import Markets
from trading.screener import bollinger_band_width_percentile as bbwp
from trading.screener import screener_common
'''
Description:

  Screener for low volatility tickers that have very low volatility in the last 5-10 days

Usage

  python trading/screener/screener_vol_squeeze.py --timeframe 1h --dir data/crypto/1h/csv --market crypto
  python trading/screener/screener_vol_squeeze.py --timeframe 1d --dir data/spx-bbwp-screener --market spx --output-dir 'data/screener-vol-spx' --tag "Low Vol SPX"

'''


# Step 1: Calculate Absolute Range
def calculate_range(df):
  df['Range'] = (df['Open'] - df['Close']).abs()
  return df


# Step 2: Calculate Rolling Standard Deviation of the Range
def calculate_rolling_std(df, window=20):
  df['Range_Std'] = df['Range'].rolling(window=window).std()
  return df


# Function to check for the narrowest range in the last N days
def check_narrowest_range(df, window=5):
  # Rolling minimum of the Range over the last N days
  df['Narrowest_Range'] = df['Range'].rolling(window=window).min()

  # Check if the current range is the narrowest in the last N days
  df['Is_Narrowest'] = df['Range'] == df['Narrowest_Range']

  return df


# Step 3: Identify Range Narrowing (Squeeze)
def identify_squeeze(df):
  df['Is_Squeezing'] = df['Range_Std'].diff() < 0

  # Example usage with a DataFrame
  df = check_narrowest_range(df, window=20)  # for the last 5 days
  # df = check_narrowest_range(df, window=10)  # for the last 10 days
  vals = df[['Date', 'Range', 'Narrowest_Range', 'Is_Narrowest']].tail(10)
  # View the columns with narrowest ranges
  logger.info(f"Narrowest Ranges: \n {vals}")

  return df


# Function to plot the ranges and squeezes
def plot_range_and_squeeze(df):
  sns.set(style="whitegrid")

  plt.figure(figsize=(14, 8))

  # Plot Range (|Open - Close|)
  plt.plot(df['Date'],
           df['Range'],
           label='Range (|Open - Close|)',
           color='blue')

  # Plot Rolling Standard Deviation of the Range
  plt.plot(df['Date'],
           df['Range_Std'],
           label='Rolling Std of Range',
           color='orange')

  # Highlight Squeeze Points
  plt.scatter(df[df['Is_Squeezing']]['Date'],
              df[df['Is_Squeezing']]['Range'],
              color='red',
              label='Squeeze Points',
              zorder=5)

  plt.title('Range and Squeeze Detection')
  plt.xlabel('Date')
  plt.ylabel('Range')
  plt.legend()
  plt.xticks(rotation=45)
  plt.tight_layout()
  plt.show()


# Main function to process data
def bollinger_band_width_range_screener(symbol, df, window=20):
  df = calculate_range(df)
  df = calculate_rolling_std(df, window)
  df = identify_squeeze(df)
  return symbol, df


def find_volatility_squeeze(files_df, limit=0, make_chart=False):
  # df = bollinger_band_width_range_screener(df, window=20)
  # plot_range_and_squeeze(df)

  dataframes = {}
  squeezing_tickers = []

  with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(bollinger_band_width_range_screener, symbol, df):
        symbol
        for symbol, df in files_df.items()
    }

    for future in concurrent.futures.as_completed(futures):
      symbol, df = future.result()
      dataframes[symbol] = df
      # is last row is squeezing, then extend squeezing_tickers
      if df.iloc[-1]['Is_Narrowest'] and df.iloc[-1]['Is_Squeezing']:
        logger.info(f'Squeeze on for : {symbol}')
        # plot_range_and_squeeze(df)
        squeezing_tickers.append(symbol)

  return dataframes, squeezing_tickers


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--dir",
                      default='data/spx-bbwp-screener',
                      help="Which directory to save to",
                      type=str)

  # arg called output_dir, default to be data/bbwp
  parser.add_argument('--output-dir', '-o', default='data/ttm-spx', type=str)

  # add tag as ttm squeeze
  parser.add_argument('--tag', default='TTM Squeeze', type=str)

  parser.add_argument('--timeframe',
                      '-t',
                      default='1d',
                      type=str,
                      help='What timeframe/interval to use')

  parser.add_argument('--market',
                      type=Markets,
                      default=Markets.CRYPTO,
                      help='What market')

  # add arg for showing charts
  parser.add_argument('--charts', action='store_true', help='Show charts')

  args = parser.parse_args()
  return args


def main():
  args = get_args()
  logger.info(f'args = {args}')

  files_df = bbwp.process_files(args.dir)
  dfs, largest_date = bbwp.filter_away_stale_data(files_df, args.market,
                                                  args.timeframe)

  dataframes, squeezing_tickers = find_volatility_squeeze(
      dfs, make_chart=args.charts)

  logger.info(f'squeezing_tickers = {squeezing_tickers}')
  tickers_for_tradingview = bbwp.transform_tickers(squeezing_tickers,
                                                   args.market)

  logger.info(f'tickers_for_tradingview = {tickers_for_tradingview}')

  bbwp.save_names_to_txt(tickers_for_tradingview, args.output_dir, args.market,
                         args.timeframe, largest_date)
  new_tickers = bbwp.get_new_tickers_compared_to_older_file(
      tickers_for_tradingview, args.output_dir, args.market, args.timeframe,
      largest_date)

  logger.info(f'new_tickers = {new_tickers}')
  bbwp.send_telegram_msg(args, tickers_for_tradingview, new_tickers,
                         args.market)


if __name__ == '__main__':
  main()
