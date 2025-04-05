import os, pandas
from enum import Enum
import argparse
import concurrent.futures

from trading.common import utils

logger = utils.get_logger('ttm_squeeze')
import plotly.graph_objects as go
import pandas as pd

from trading.screener.screener_common import Markets
from trading.screener import bollinger_band_width_percentile as bbwp
from trading.screener import screener_common
'''
Description:

  Generates a chart of the TTM Squeeze
  Generate a list of all theTTM Squeeze, by market

Usage

  python trading/screener/ttm_squeeze.py --timeframe 1h --dir data/crypto/1h/csv --market crypto
  python trading/screener/ttm_squeeze.py --timeframe 1d --dir data/spx-bbwp-screener --market spx --output-dir 'data/ttm-spx' --tag "TTM Squeeze SPX"

'''


def process_single_file(symbol, df, make_chart):
  # Calculate rolling statistics
  df['20sma'] = df['Close'].rolling(window=20).mean()
  df['stddev'] = df['Close'].rolling(window=20).std()
  df['lower_band'] = df['20sma'] - (2 * df['stddev'])
  df['upper_band'] = df['20sma'] + (2 * df['stddev'])

  df['TR'] = abs(df['High'] - df['Low'])
  df['ATR'] = df['TR'].rolling(window=20).mean()

  df['lower_keltner'] = df['20sma'] - (df['ATR'] * 1.5)
  df['upper_keltner'] = df['20sma'] + (df['ATR'] * 1.5)

  # Determine squeeze condition
  df['squeeze_on'] = df.apply(lambda row: row['lower_band'] > row[
      'lower_keltner'] and row['upper_band'] < row['upper_keltner'],
                              axis=1)

  # Log the last 4 squeeze_on values
  last_sq_vals = df['squeeze_on'].iloc[-4:].to_list()
  logger.info(f'      symbol: {symbol}; last_sq_vals = {last_sq_vals}')

  squeezing_tickers = []
  if df.iloc[-3]['squeeze_on'] and not df.iloc[-1]['squeeze_on']:
    logger.info(f"{symbol} is coming out of the squeeze")
    squeezing_tickers.append(symbol)
    if make_chart:
      chart(df)  # Assuming chart function is defined elsewhere

  return symbol, df, squeezing_tickers


def process_files(files_df, limit=0, make_chart=False):
  dataframes = {}
  squeezing_tickers = []

  with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(process_single_file, symbol, df, make_chart): symbol
        for symbol, df in files_df.items()
    }

    for future in concurrent.futures.as_completed(futures):
      symbol, df, tickers = future.result()
      dataframes[symbol] = df
      squeezing_tickers.extend(tickers)

  return dataframes, squeezing_tickers


def chart(df):
  candlestick = go.Candlestick(x=df['Date'],
                               open=df['Open'],
                               high=df['High'],
                               low=df['Low'],
                               close=df['Close'])
  upper_band = go.Scatter(x=df['Date'],
                          y=df['upper_band'],
                          name='Upper Bollinger Band',
                          line={'color': 'red'})
  lower_band = go.Scatter(x=df['Date'],
                          y=df['lower_band'],
                          name='Lower Bollinger Band',
                          line={'color': 'red'})

  upper_keltner = go.Scatter(x=df['Date'],
                             y=df['upper_keltner'],
                             name='Upper Keltner Channel',
                             line={'color': 'blue'})
  lower_keltner = go.Scatter(x=df['Date'],
                             y=df['lower_keltner'],
                             name='Lower Keltner Channel',
                             line={'color': 'blue'})

  fig = go.Figure(
      data=[candlestick, upper_band, lower_band, upper_keltner, lower_keltner])
  fig.layout.xaxis.type = 'category'
  fig.layout.xaxis.rangeslider.visible = False
  fig.show()


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

  dataframes, squeezing_tickers = process_files(dfs, make_chart=args.charts)

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

  # if args.charts:
  #   chart(dataframes[first_ticker])
