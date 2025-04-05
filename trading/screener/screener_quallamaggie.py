from datetime import datetime, timedelta
import concurrent.futures

from trading.common import utils

logger = utils.get_logger('screener_quallamaggie', use_rich=True)

import argparse
import yfinance as yf
import pandas as pd
import ccxt

from trading.common.utils import HLOC, Markets
from trading.screener import bollinger_band_width_percentile as bbwp
'''
Usage

python trading/screener/screener_quallamaggie.py --market spx
'''

# Define EP conditions
# gap_condition = gap_size >= 0.10  # 10% or more gap up
# volume_condition = volume_ratio >= 2.0  # 2x average volume


def scan_pattern_for_episodic_pivot(ticker, df):
  """
    Check if the given ticker has had a pivot today.

    Args:
        ticker (str): The stock ticker symbol
        df (pd.DataFrame): DataFrame with historical price data

    Returns:
        dict: Pivot data if found today, None otherwise
    """
  try:
    if df.empty:
      return None

    # Calculate required metrics
    df['Return'] = df['Close'].pct_change()
    df['SMA_Volume'] = df['Volume'].rolling(window=50).mean()

    # Get today's date
    today = df['Date'].iloc[-1]

    # Get indices for today and yesterday
    today_idx = -1  # Last row
    yesterday_idx = -2  # Second to last row

    # Check for gap up of 10% or more
    gap_size = df['Open'].iloc[today_idx] / df['Close'].iloc[yesterday_idx] - 1

    # Check volume conditions
    current_volume = df['Volume'].iloc[today_idx]
    avg_volume = df['SMA_Volume'].iloc[today_idx]
    volume_ratio = current_volume / avg_volume if not pd.isna(
        avg_volume) else 0

    # Define EP conditions
    gap_condition = gap_size >= 0.10  # 10% or more gap up
    volume_condition = volume_ratio >= 2.0  # 2x average volume

    if gap_condition and volume_condition:
      pivot_data = {
          'ticker': ticker,
          'date': today,
          'gap_percentage': round(gap_size * 100, 2),
          'volume_ratio': round(volume_ratio, 2),
          'price': round(df['Close'].iloc[today_idx], 2),
          'volume': int(current_volume)
      }
      return pivot_data

    return None

  except Exception as e:
    print(f"Error processing {ticker}: {str(e)}")
    raise e
    return None


def read_tickers_from_file(filename):
  """Read tickers from a file, one per line or comma-separated"""
  try:
    with open(filename, 'r') as f:
      content = f.read().strip()
      # Check if content contains commas
      if ',' in content:
        # Split by comma and clean each ticker
        tickers = [t.strip() for t in content.split(',')]
      else:
        # Split by lines and clean each ticker
        tickers = [t.strip() for t in content.splitlines()]
    return [t for t in tickers if t]  # Remove empty strings
  except FileNotFoundError:
    logger.info(f"Error: File '{filename}' not found.")
    sys.exit(1)
  except Exception as e:
    logger.info(f"Error reading file: {str(e)}")
    sys.exit(1)


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
                      default='data/quallamaggie-spx',
                      type=str)

  # add tag as ttm squeeze
  parser.add_argument('--tag', default='quallamaggie finds', type=str)

  parser.add_argument('--timeframe',
                      '-t',
                      default='1d',
                      type=str,
                      help='What timeframe/interval to use')

  parser.add_argument('--market',
                      type=Markets,
                      default=Markets.SPX,
                      help='What market')

  args = parser.parse_args()
  return args


# Main function to process data
def scan_sym_for_pivot(symbol, df, window=20):
  pivot_info = scan_pattern_for_episodic_pivot(symbol, df)
  return symbol, pivot_info


def find_episodic_pivots(files_df, limit=0):

  pivot_tickers = []

  with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(scan_sym_for_pivot, symbol, df): symbol
        for symbol, df in files_df.items()
    }

    for future in concurrent.futures.as_completed(futures):
      symbol, pivot_info = future.result()
      if pivot_info:
        logger.info(f'VALID received: {symbol} --> {pivot_info}')
        pivot_tickers.append(symbol)

  return pivot_tickers


def main():
  args = get_args()
  logger.info(f'args: {args}')

  files_df = bbwp.process_files(args.dir)

  dfs, largest_date = bbwp.filter_away_stale_data(files_df, args.market,
                                                  args.timeframe)

  pivot_tickers = find_episodic_pivots(dfs)

  logger.info(f'pivot_tickers = {pivot_tickers}')
  tickers_for_tradingview = bbwp.transform_tickers(pivot_tickers, args.market)

  logger.info(f'tickers_for_tradingview = {tickers_for_tradingview}')

  bbwp.save_names_to_txt(tickers_for_tradingview, args.output_dir, args.market,
                         args.timeframe, largest_date)
  new_tickers = bbwp.get_new_tickers_compared_to_older_file(
      tickers_for_tradingview, args.output_dir, args.market, args.timeframe,
      largest_date)

  logger.info(f'new_tickers = {new_tickers}')
  bbwp.send_telegram_msg(args, tickers_for_tradingview, new_tickers,
                         args.market)


# Example usage
if __name__ == "__main__":
  main()
