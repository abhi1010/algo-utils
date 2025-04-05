import os
import sys
from enum import Enum
import argparse

import pandas as pd

from trading.crypto import volume_selector
from trading.common import utils

logger = utils.get_logger()
"""
Description:

  Downloads crypto files in JSON format and saves the files in CSV format.
  Useful for testing again backtester related stuff.


Usage:
  python trading/tables/kucoin_download.py --download-data  --timeframe 4h --dir data/crypto/4h --kucoin-pairs-file data/kucoin-200.json --data-format-ohlcv json --days 500

  python trading/tables/kucoin_download.py --download-data  --timeframe 1h --dir data/crypto/1h --kucoin-pairs-file data/kucoin-200.json --data-format-ohlcv json --days 500

  python trading/tables/kucoin_download.py --download-data  --timeframe 1h --dir data/crypto/1h --kucoin-pairs-file data/kucoin-10.json  --data-format-ohlcv json --days 100

"""


class DownloadFormat(str, Enum):
  """
  Enum to represent the download format.
  """
  CSV = "csv"
  JSON = "json"


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--days', default=1, type=int, help='How many days to download data for')

  parser.add_argument(
      '--download-data',
      '-d',
      default=False,
      action='store_true',
      help='Should we download the data to find out this information?')

  parser.add_argument(
      '--timeframe',
      '-t',
      default='1d',
      type=str,
      help='What timeframe/interval to use')

  parser.add_argument(
      "--dir",
      default='data/crypto/',
      help="Which directory to save to",
      type=str)

  parser.add_argument(
      "--kucoin-pairs-file",
      default='',
      help="Do we have a ready made list of kucoin pairs? If empty, we download everything",
      type=str)

  parser.add_argument(
      '--format',
      type=DownloadFormat,
      default=DownloadFormat.CSV,
      help='What format')

  args = parser.parse_args()
  return args


def convert_json_to_csv(
    dir,
    format_to_save=DownloadFormat.JSON,
    interval=None,
    token_first_names=[]):

  list_json_files = [
      f for f in os.listdir(dir) if f.endswith('.json') and f'-{interval}' in f
      if f and any([t in f for t in token_first_names])
  ]
  logger.info(
      f'list_json_files = {list_json_files}; dir = {dir};'
      f'token_first_names={token_first_names}; interval  = {interval}')

  is_csv_updated = {}
  for f in list_json_files:

    df = pd.read_json(os.path.join(dir, f))
    df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']

    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    # df['Close'] = df['Close']

    df.set_index('Date', inplace=True)
    df = df.sort_index(ascending=True)
    logger.info(f'json = {f}; df = {df}')
    if format_to_save == DownloadFormat.CSV:
      utils.check_and_create_directory(os.path.join(dir, 'csv'))
      csv_path = os.path.join(dir, 'csv', f.replace('.json', '.csv'))

      del df['Timestamp']
      is_csv_updated[csv_path] = compare_rows_of_df_to_csv_data_last_row(
          df, csv_path)

      df.to_csv(csv_path)
      logger.info(f'csv path = {csv_path}')
  return is_csv_updated


def compare_rows_of_df_to_csv_data_last_row(df, csv_path):
  """
  Compare the last row of the DataFrame df with the last row of the CSV file csv_path
  :param df:
  :param csv_path:
  :return:
  """
  if not os.path.exists(csv_path):
    return False
  df_csv = pd.read_csv(csv_path, index_col=0, parse_dates=True)
  df_csv_idx = df_csv.index[-1]
  df_idx = df.index[-1]

  is_last_row_same = df_csv_idx == df_idx
  is_first_row_same = df_csv.index[0] == df.index[0]
  # is_data_diff = df_csv.compare(df)
  logger.info(
      f'is_data_same = {is_last_row_same}; '
      f'is_first_row_same = {is_first_row_same}; '
      f'df_csv_idx = {df_csv_idx}; df_idx={df_idx};'
      f';')
  return is_last_row_same


def main():
  args = get_args()
  logger.info(f'args = {args}')
  kucoin_pairs = volume_selector.read_kucoin_pairs(args.kucoin_pairs_file)
  logger.info(f'kucoin_pairs = {kucoin_pairs}; len={len(kucoin_pairs)}')
  first_names = [
      utils.get_token_name_without_quote_currency(t) for t in kucoin_pairs
  ]
  if args.download_data:
    volume_selector.download_data(
        kucoin_pairs, args.dir, args.days, args.timeframe)
  convert_json_to_csv(
      args.dir,
      args.format,
      interval=args.timeframe,
      token_first_names=first_names)


if __name__ == '__main__':
  main()
