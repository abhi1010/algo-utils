import os, pandas
from enum import Enum
import argparse
import concurrent.futures

from trading.common import utils

logger = utils.get_logger('screener_range_squeeze')
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from trading.screener.screener_common import Markets
from trading.screener import bollinger_band_width_percentile as bbwp
from trading.screener import screener_common
'''
Description:

  Screener for low range tickers that have very low range in the last 5-10 days

Usage

  python trading/screener/screener_range_squeeze.py --timeframe 1h --dir data/crypto/1h/csv --market crypto
  python trading/screener/screener_range_squeeze.py --timeframe 1d --dir data/spx-bbwp-screener --market spx --output-dir 'data/screener-range-spx' --tag "Low Range SPX"

'''


def calculate_range_indicator(symbol, data, window):
  # Calculate the absolute value of (open - close)
  data['range'] = abs(data['Open'] - data['Close'])

  # Calculate the moving average of the range
  data['range_ma'] = data['range'].rolling(window=window).mean()

  # Calculate the ratio of current range to moving average
  data['range_ratio'] = data['range'] / data['range_ma']
  # logger.info(f'symbol: {symbol}; data = {data}')
  return data


def find_low_range_stocks(symbol, data, window, threshold):
  results = []

  # for symbol in symbols:
  #   # data = fetch_data(symbol, start_date, end_date)
  #   # if data.empty:
  #   #   continue

  data = calculate_range_indicator(symbol, data, window)

  # Check if the most recent range ratio is below the threshold
  if data['range_ratio'].iloc[-1] < threshold:
    results.append({
        'symbol': symbol,
        'last_date': data.index[-1],
        'last_range_ratio': data['range_ratio'].iloc[-1]
    })

  return results


def bollinger_band_width_range_screener(symbol, df):
  window = 20  # Number of periods for moving average
  threshold = 0.8  # Threshold for considering the range as getting smaller

  screener_results = find_low_range_stocks(symbol, df, window, threshold)

  return symbol, screener_results


def find_range_squeeze(files_df, limit=0, make_chart=False):
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
      symbol, screener_results = future.result()

      logger.info(f'symbol: {symbol}; results = {screener_results}')

      if screener_results:
        dataframes[symbol] = screener_results
        # TODO: Need to find the lowest rank amongst them

  return dataframes, squeezing_tickers


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--dir",
                      default='data/spx-bbwp-screener',
                      help="Which directory to save to",
                      type=str)

  # arg called output_dir, default to be data/bbwp
  parser.add_argument('--output-dir',
                      '-o',
                      default='data/screener-spx-range',
                      type=str)

  parser.add_argument('--tag', default='Range Squeeze', type=str)

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

  dataframes, squeezing_tickers = find_range_squeeze(dfs,
                                                     make_chart=args.charts)

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
