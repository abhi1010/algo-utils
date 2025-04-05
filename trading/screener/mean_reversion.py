import os
import sys
import argparse
import yfinance as yf
import pandas_ta as ta
import pandas as pd
from pprint import pformat

from trading.common import utils

logger = utils.get_logger('mean-reversion', False)

from trading.screener import bollinger_band_width_percentile as bbwp
from trading.screener.screener_common import Markets
from trading.services import telegram_runner
# from trading.screener.screener_common import *


def fetch_stock_data(symbol, period='1y'):
  stock = yf.Ticker(symbol)
  df = stock.history(period=period)
  return df


def calculate_indicators(df):
  # Bollinger Bands
  df.ta.bbands(length=20, append=True)

  # Keltner Channels
  df.ta.kc(length=20, append=True)

  # ADX
  df.ta.adx(length=14, append=True)

  return df


def screen_stock_potential_downside(df, symbol):
  # df = fetch_stock_data(symbol)
  df = calculate_indicators(df)

  # Get the last row of data
  last_row = df.iloc[-1]

  # Check conditions
  price_touching_upper_bb = last_row['Close'] >= last_row['BBU_20_2.0']
  bb_wider_than_kc = (last_row['BBU_20_2.0'] -
                      last_row['BBL_20_2.0']) > (last_row['KCUe_20_2'] -
                                                 last_row['KCLe_20_2'])
  adx_trending_down = df['ADX_14'].diff().iloc[-1] < 0

  return price_touching_upper_bb and bb_wider_than_kc and adx_trending_down


def screen_stock_potential_upside(df, symbol):
  # df = fetch_stock_data(symbol)
  df = calculate_indicators(df)

  # Get the last row of data
  last_row = df.iloc[-1]

  # Check conditions
  price_touching_lower_bb = last_row['Close'] <= last_row['BBL_20_2.0']
  bb_wider_than_kc = (last_row['BBU_20_2.0'] -
                      last_row['BBL_20_2.0']) > (last_row['KCUe_20_2'] -
                                                 last_row['KCLe_20_2'])
  adx_trending_up = df['ADX_14'].diff().iloc[-1] > 0

  return price_touching_lower_bb and bb_wider_than_kc and adx_trending_up


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--dir",
                      default='data/spx-bbwp-screener/',
                      help="Which directory to save to",
                      type=str)

  # arg called output_dir, default to be data/bbwp
  parser.add_argument('--output-dir', '-o', default='data/mean', type=str)

  # add tag as BBWP screener
  parser.add_argument('--tag', default='MEAN_REVERSION', type=str)

  parser.add_argument('--timeframe',
                      '-t',
                      default='1h',
                      type=str,
                      help='What timeframe/interval to use')

  parser.add_argument('--market',
                      type=Markets,
                      default=Markets.SPX,
                      help='What market')

  args = parser.parse_args()
  return args


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
    nse_prefix = 'NSE:'
    prefixer_style = lambda tickers: (nse_prefix + (',' + nse_prefix).join(
        tickers)) if len(tickers) else ''
  else:
    prefixer_style = lambda tickers: ','.join(tickers)

  new_tickers_s = prefixer_style(new_tickers)
  all_tickers_s = prefixer_style(tickers_for_telegram)

  #   msg += f'\n\n All tickers: {all_tickers_s}'
  msg = f'''Args:
  ```python
{namespace_str}```

New tickers: ```
{new_tickers_s}```

All Tickers: ```
{all_tickers_s}```
'''
  bbwp._send_msg(msg)


def alert_for_tickers(tickers, largest_date, args):
  tickers_for_tradingview = bbwp.transform_tickers(tickers, args.market)

  logger.info(f'tickers_for_tradingview = {tickers_for_tradingview}')

  bbwp.save_names_to_txt(tickers_for_tradingview, args.output_dir, args.market,
                         args.timeframe, largest_date)
  new_tickers = bbwp.get_new_tickers_compared_to_older_file(
      tickers_for_tradingview, args.output_dir, args.market, args.timeframe,
      largest_date)

  logger.info(f'new_tickers = {new_tickers}')

  send_telegram_msg(args, tickers_for_tradingview, new_tickers, args.market)


def main():
  args = get_args()
  # create dir, args.output_dir, if it does not exist, using os.makedirs
  os.makedirs(args.output_dir, exist_ok=True)

  logger.info(f'args = {args}')

  files_df = bbwp.process_files(args.dir)
  dfs, largest_date = bbwp.filter_away_stale_data(files_df, args.market,
                                                  args.timeframe)

  screened_stocks = []

  for symbol, df in dfs.items():
    logger.info(f"Screening {symbol}...")
    if screen_stock_potential_downside(
        df, symbol) or screen_stock_potential_upside(df, symbol):
      screened_stocks.append(symbol)

  logger.info("\nStocks that passed the screening:")
  for stock in screened_stocks:
    logger.info(stock)

  alert_for_tickers(screened_stocks, largest_date, args)


if __name__ == "__main__":
  main()
