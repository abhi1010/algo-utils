import sys
import os
import sys
from enum import Enum
import argparse

import pandas as pd
import numpy as np

np.set_printoptions(suppress=True)

from trading.common import utils

logger = utils.get_logger()
"""
Description:

  Provide a CSV file. It converts into json format that can be used by freqtrade.


Usage:
  python csv_to_json.py --timeframe d --resample        --file data/crypto/d/csv/BAX_USDT-1d.csv
  python csv_to_json.py --timeframe d  --dir data/crypto --file data/crypto/d/csv/BAX_USDT-1d.csv

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

  parser.add_argument("--dir",
                      default='data/crypto/',
                      help="Which directory to save to",
                      type=str)

  parser.add_argument('--file', default='', help='What file')

  parser.add_argument('--resample',
                      default=False,
                      action='store_true',
                      help='Do we need to resample?')

  parser.add_argument('--timeframe',
                      default=utils.TimeFrame.daily,
                      type=utils.TimeFrame,
                      help='The timeframe of conversion')

  args = parser.parse_args()
  return args


def read_data(file_path, timeframe, resample=False):
  logger.info(f'loading file: {file_path}')
  if os.path.exists(file_path):

    df_main = pd.read_csv(file_path)
    date_col_name = 'Date' if 'Date' in df_main.columns else 'Datetime'
    # df_main[date_col_name].apply(lambda x: x.replace(tzinfo=None))

    df_main['Date'] = pd.to_datetime(df_main[date_col_name], utc=True)
    logger.info(f'loaded data = {df_main}')
    # df_main['Date'].dt.tz_localize(None)
    df_main['Date'].apply(lambda x: x.replace(tzinfo=None))
    if date_col_name == 'Datetime':
      del df_main[date_col_name]
    logger.info(f'df = {df_main}; ')

    if resample:
      shape_before_reasmple = df_main.shape
      df_main = utils.resample_to_interval(df_main, timeframe)
      logger.info(f'Resampled data shape: {df_main.shape}; '
                  f'earlier: {shape_before_reasmple}')
    df_main = df_main.set_index(['Date'])

    return df_main
  else:
    logger.warning(f'File missing: {file_path}')
  return pd.DataFrame()


def convert_df_to_json_compatible(df):
  # df.columns = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
  # df['Timestamp'] = pd.Timestamp(df.index)
  df['Timestamp'] = df.index.values.astype(np.int64) // 10**9
  df = df.set_index('Timestamp')
  del df['Close']
  # del df['Date']
  return df


def save_as_json(df, dir, file_prefix):
  file_path = os.path.join(dir, f'{file_prefix}.json')
  df.to_json(file_path, orient='records')
  logger.info(f'saved json file: {file_path}')


def get_csv_df_as_json_lines(df, csv_file):

  initial_dir = os.path.dirname(csv_file)
  base_csv_name = os.path.basename(csv_file)
  dir_to_save = os.path.join(initial_dir, 'json')

  utils.check_and_create_directory(dir_to_save)
  json_path = os.path.join(dir_to_save, base_csv_name.replace('.csv', '.json'))

  json_lines = []
  r1 = df.to_records(column_dtypes={"Volume": "float64"})
  for row in r1:
    s = []
    for v in row:
      s.append(str(v))
      # logger.info(f'v = {v}')
    s_row = '[' + ','.join(s) + ']'
    json_lines.append(s_row)
  return json_path, json_lines


def save_csv_df_as_json(json_path, json_lines):
  json_s = '[' + ','.join(json_lines) + ']'
  with open(json_path, 'wt') as f:
    f.write(json_s)
  logger.info(f'json_path = {json_path}')


def main():
  args = get_args()
  logger.info(f'args = {args}')
  df = read_data(args.file, args.timeframe, args.resample)
  df = convert_df_to_json_compatible(df)
  logger.info(f'adjusted df = {df}')
  json_path, json_lines = get_csv_df_as_json_lines(df, args.file)
  save_csv_df_as_json(json_path, json_lines)


if __name__ == '__main__':
  main()
