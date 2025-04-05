from dateutil.parser import *
from datetime import date
import json
import csv
import pickle
import argparse
import os
from datetime import datetime, timedelta
import shutil

import yfinance as yf
from finviz.screener import Screener
import pandas as pd
import pandas_ta as ta
import requests
from prettytable import PrettyTable
import pandas as pd
from py_vollib_vectorized import price_dataframe, get_all_greeks
import py_vollib.black.implied_volatility
import py_vollib_vectorized
from openpyxl.styles import Color, PatternFill, Font, Border
from openpyxl.styles.differential import DifferentialStyle
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule, FormulaRule

from trading.data import nifty_data as nse
from nselib import derivatives
from trading.common import utils, util_files
from trading.options import option_utils
from trading.data import json_db
'''
Usage:

# run the greeks
alias r='python trading/options/option_greeks_deltas.py --dry-run --num-symbols 1 --timestamp max'

# load from a parquet file
alias p='python trading/options/option_greeks_deltas.py --dry-run --num-symbols 1 --timestamp max --load-latest-parquet'
'''


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--dry-run", "-d", action='store_true', help="No real buying")

  # add an argument called "load-latest-parquet"
  parser.add_argument(
      "--load-latest-parquet",
      "-l",
      action='store_true',
      help="Load latest parquet")

  parser.add_argument('--timestamp', '-t', default='max', help='expiry date')
  parser.add_argument(
      '--num-symbols',
      '-n',
      type=int,
      default=3,
      help='number of symbols to include')

  args = parser.parse_args()
  return args


args0 = get_args()

FILE_FORMAT = (
    "dry-" if args0.dry_run else "") + "greeks_deltas__%Y-%m-%d__%H-%M-%S"
now_s = datetime.now().strftime(FILE_FORMAT)
DIR_IV = 'data/iv'
SAVE_FOLDER = 'data/iv-dry' if args0.dry_run else 'data/iv'
ROUNDING = 2

pd.options.display.float_format = '{:.2f}'.format
logger = utils.get_logger(now_s)

logger.info(f'File = {now_s}; SAVE_FOLDER = {SAVE_FOLDER}')

# how to get FnO stock list
# https://archives.nseindia.com/content/fo/fo_mktlots.csv
# best gainers here: https://www.moneycontrol.com/stocks/fno/marketstats/futures/gainers/homebody.php?opttopic=&optinst=allfut&sel_mth=all&sort_order=0

# URL of the CSV file
URL = 'https://archives.nseindia.com/content/fo/fo_mktlots.csv'


def get_stock_list(num=0):
  return ['NIFTY']
  fno_list = option_utils.get_fno_list(URL)
  if num:
    return fno_list[:num]
  return fno_list


def get_daily_sum_xl_path(dry_run, add_suffix: bool):
  file_prefix = "dry-" if dry_run else ""
  backup_suffix = ''
  if add_suffix:
    today = datetime.today()
    backup_suffix = f"_{today.strftime('%Y%m%d')}"
  greeks_xl_file = os.path.join(
      SAVE_FOLDER, f"{file_prefix}greeks{backup_suffix}.xlsx")
  return greeks_xl_file


def create_backup(orig_file, new_file):
  logger.info(f'copying {orig_file} to {new_file}')
  shutil.copyfile(orig_file, new_file)


def add_greeks(eligible_options):
  sorted_data = []
  for option in eligible_options:
    hist_data = option.HistoricData

    if hist_data['Close'].shape[0] <= 4:
      logger.info(f'^^^^^ Skipping as shape is bad. option={option}')
      option.Valid = False
      continue

    # Calculate the average trading volume excluding zero values
    average_volume = hist_data[hist_data['TOT_TRADED_QTY'] >
                               0]['TOT_TRADED_QTY'].mean()

    df = hist_data
    # Convert date columns to datetime format
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y')
    df['EXPIRY_DT'] = pd.to_datetime(df['EXPIRY_DT'], format='%d-%b-%Y')

    # Calculate the business days to expiry for each row
    df['TTE_BDays'] = df.apply(
        lambda row: len(
            pd.date_range(
                start=row['TIMESTAMP'], end=row['EXPIRY_DT'], freq='B')),
        axis=1)

    # Calculate annualized time to expiry for each row based on business days
    trading_days_in_a_year = 252  # Adjust for your market
    df['T'] = df['TTE_BDays'] / trading_days_in_a_year
    df['R'] = 0.07
    df['IV'] = option.IV / 100

    hist_data['Flag'] = hist_data['OPTION_TYPE'][0][0].lower()
    hist_data = hist_data[hist_data['UNDERLYING_VALUE'].notna()]

    df = pd.DataFrame()
    df['Flag'] = hist_data['Flag'].to_list()
    # data['UNDERLYING_VALUE'] = data['UNDERLYING_VALUE'].fillna(0)
    if hist_data['UNDERLYING_VALUE'].isnull().values.any():
      logger.info(
          f'found some null values in UNDERLYING_VALUE: \n{hist_data.to_string()}'
      )
      continue
    df['S'] = hist_data['UNDERLYING_VALUE'].to_list()
    df['K'] = [x or 1 for x in hist_data['STRIKE_PRICE'].to_list()]
    df['TTE'] = hist_data['T'].to_list()
    df['R'] = hist_data['R'].to_list()
    df['IV'] = hist_data['IV'].to_list()
    logger.info(f'prices = {hist_data["UNDERLYING_VALUE"]}')
    logger.info(f'prices = {hist_data.to_string()}')

    py_vollib_vectorized.price_dataframe(
        df,
        flag_col='Flag',
        underlying_price_col='S',
        strike_col='K',
        annualized_tte_col='TTE',
        riskfree_rate_col='R',
        sigma_col='IV',
        model='black',
        inplace=True)

    # logger.info(f'bs_data = \n{bs_data.to_string()}')
    logger.info('-' * 80)
    logger.info(f'df= \n{df.to_string()}')
    logger.info(f'data 1= \n{hist_data.to_string()}')

    # calculate the greeks for the options data

  return sorted_data


def get_implied_volatility(data):
  data['Flag'] = data['OPTION_TYPE'][0][0].lower()

  df = pd.DataFrame()
  df['Flag'] = data['Flag'].to_list()
  # data['UNDERLYING_VALUE'] = data['UNDERLYING_VALUE'].fillna(0)
  if data['UNDERLYING_VALUE'].isnull().values.any():
    logger.info(
        f'found some null values in UNDERLYING_VALUE: \n{data.to_string()}')
  df['S'] = data['UNDERLYING_VALUE'].to_list()
  df['K'] = [x or 1 for x in data['STRIKE_PRICE'].to_list()]
  df['TTE'] = data['T'].to_list()
  df['R'] = data['R'].to_list()
  df['IV'] = data['IV'].to_list()
  py_vollib_vectorized.vectorized_implied_volatility_black(
      price, F, K, r, t, flag, return_as='numpy')


def save_option_df(df, ticker, dry_run):

  # Get the current timestamp
  timestamp = datetime.now().strftime('%Y%m%d_%H%M')

  # Create the directory if it doesn't exist
  os.makedirs(SAVE_FOLDER, exist_ok=True)

  # Save the list to a file using pickle with the timestamp in the filename
  save_filename = os.path.join(SAVE_FOLDER, f'{ticker}_{timestamp}.parquet')
  logger.info(f'Saving options df into: {save_filename}')
  with open(save_filename, 'wb') as f:
    df.to_parquet(save_filename, compression="gzip")


def main2(args):
  pass


def calc_iv_greeks_into_xl(args, data):
  t = 'NIFTY'

  data['Flag'] = data['OPTION_TYPE'].apply(lambda x: x[:1].lower())

  option_utils.set_tte(data)

  df = pd.DataFrame()
  flag = data['Flag'].to_list()
  # data['UNDERLYING_VALUE'] = data['UNDERLYING_VALUE'].fillna(0)
  if data['UNDERLYING_VALUE'].isnull().values.any():
    logger.info(
        f'found some null values in UNDERLYING_VALUE: \n{data.to_string()}')
  F = data['UNDERLYING_VALUE'].to_list()
  K = [x or 1 for x in data['STRIKE_PRICE'].to_list()]
  df['T'] = data['T'].to_list()
  # df['IV'] = data['IV'].to_list()
  price = data['Close'].to_list()
  # F = [95, 91]
  # K = [100, 90]
  t = df['T'].to_list()
  data['R'] = r = .07
  # flag = ['c', 'p']
  iv_series_bsm = py_vollib_vectorized.vectorized_implied_volatility(
      price,
      F,
      K,
      t,
      r,
      flag,
      q=0,
      model='black_scholes_merton',
      return_as='numpy')
  # iv_series_bs = py_vollib_vectorized.vectorized_implied_volatility(
  #     price, F, K, t, r, flag, q=0, model='black_scholes', return_as='numpy')
  # iv_series_b = py_vollib_vectorized.vectorized_implied_volatility(
  #     price, F, K, t, r, flag, q=0, model='black', return_as='numpy')
  # iv_series = py_vollib.black.implied_volatility.implied_volatility(
  #     price, F, K, r, t, flag, return_as='numpy')
  data['IV_bsm'] = iv_series_bsm

  df['S'] = data['UNDERLYING_VALUE'].to_list()
  df['K'] = [x or 1 for x in data['STRIKE_PRICE'].to_list()]

  def add_column(col_name):
    df[col_name] = data[col_name].to_list()

  for x in ['T', 'R', 'IV_bsm', 'TIMESTAMP', 'EXPIRY_DT', 'STRIKE_PRICE',
            'Flag', 'Close']:
    add_column(x)

  #  TIMESTAMP INSTRUMENT SYMBOL  EXPIRY_DT  STRIKE_PRICE OPTION_TYPE CLOSING_PRICE LAST_TRADED_PRICE  PREV_CLS SETTLE_PRICE  TOT_TRADED_QTY     TOT_TRADED_VAL  OPEN_INT  CHANGE_IN_OI MARKET_LOT  UNDERLYING_VALUE   Close     Low    High        Volume Flag  TTE_BDays    T    R  IV_bsm

  # data['IV_bs'] = iv_series_bs
  # data['IV_b'] = iv_series_b

  py_vollib_vectorized.price_dataframe(
      df,
      flag_col='Flag',
      underlying_price_col='S',
      strike_col='K',
      annualized_tte_col='T',
      riskfree_rate_col='R',
      sigma_col='IV_bsm',
      model='black_scholes',
      inplace=True)

  df_with_changes = calculate_changes(df)
  daily_sum = calc_daily_sum(df_with_changes, args.dry_run)

  greeks_xl_file_path = get_daily_sum_xl_path(args.dry_run, False)
  backup_xl_file_path = get_daily_sum_xl_path(args.dry_run, True)
  create_backup(greeks_xl_file_path, backup_xl_file_path)
  updated_daily_sum = get_filtered_daily_sum_by_ts(
      greeks_xl_file_path, daily_sum)
  write_updated_daily_sum_to_xl(updated_daily_sum, greeks_xl_file_path)


def calc_daily_sum(df, dry_run, save_tmp_excel=False):
  logger.info(f'df before daily_sum: \n{df.to_string()}')
  logger.info(f'df before daily_sum shape: {df.shape}')
  # Calculate the sum of changes across the whole day grouped by 'TIMESTAMP' and 'Flag'
  get_filepath = lambda f: os.path.join(DIR_IV, f)

  if save_tmp_excel:
    filepath = get_filepath('df_before_daily_sum.xlsx')
    df.to_excel(filepath, index=False)
    logger.info(f'file path  before daily_sum = {filepath}')
  daily_sum = df.groupby(['TIMESTAMP', 'Flag'
                         ])[['delta_change', 'theta_change',
                             'vega_change']].sum()
  logger.info(f'daily_sum before stack: c{daily_sum.to_string()}')

  if save_tmp_excel:
    filepath_daily_sum = get_filepath('df_daily_sum.xlsx')
    daily_sum.to_excel(filepath_daily_sum)
  # Unstack 'Flag' to pivot it into separate columns
  daily_sum = daily_sum.unstack('Flag')
  logger.info(f'daily_sum after stack: \n{daily_sum.to_string()}')

  # Print the daily sum results
  logger.info(
      "\nSum of changes across the whole day for the same timestamp and Flag:")

  logger.info('\n' + str(daily_sum))
  # daily_sum.to_parquet('logs/iv-dry/df3.parquet')
  return daily_sum


def calculate_changes(df):
  # Assuming df is your DataFrame
  # You may need to convert TIMESTAMP and EXPIRY_DT columns to datetime type
  # df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'])
  # df['EXPIRY_DT'] = pd.to_datetime(df['EXPIRY_DT'])

  # Sort the DataFrame by TIMESTAMP
  df = df.sort_values(by=['EXPIRY_DT', 'TIMESTAMP'])

  # Initialize columns for delta, theta, and vega changes
  df['delta_change'] = 0
  df['theta_change'] = 0
  df['vega_change'] = 0
  uniq_strike_price = df['STRIKE_PRICE'].unique()
  uniq_expiry_date = df['EXPIRY_DT'].unique()
  uniq_flag = df['Flag'].unique()

  # Iterate through each strike price and expiry date
  for strike_price in uniq_strike_price:
    for expiry_date in uniq_expiry_date:
      for flag in uniq_flag:
        # Select rows for the current strike price and expiry date
        subset = df[(df['STRIKE_PRICE'] == strike_price)
                    & (df['EXPIRY_DT'] == expiry_date)
                    & (df['Flag'] == flag)]

        # Calculate the changes compared to the previous timestamp
        subset['delta_change'] = subset['delta'].diff()
        subset['theta_change'] = subset['theta'].diff()
        subset['vega_change'] = subset['vega'].diff()
        df.update(subset)

        logger.info(
            f'strike_price = {strike_price};'
            f'expiry_date={expiry_date};'
            f'flag={flag}'
            f'\nsubset=\n{subset}')
  logger.info(f'Final df with changes: \n{df}')
  return df


def get_filtered_daily_sum_by_ts(greeks_df_file, df):

  # Load existing data from Excel
  existing_df = pd.read_excel(
      greeks_df_file, engine='openpyxl', sheet_name="Sheet1")
  logger.info(f'existing_df 1 = \n{existing_df}')
  # existing_df = existing_df.reset_index(drop=True)

  # existing_df_ts_name = 'Unnamed: 0'
  existing_df_ts_name = 'Flag'

  existing_df_ts = pd.to_datetime(existing_df[existing_df_ts_name].iloc[2:])
  logger.info(f'vals = {existing_df_ts}')
  existing_df_ts_type = type(existing_df_ts.iloc[0])
  logger.info(
      f'AA: existing_df_ts_type 0 : {existing_df_ts[2]}; '
      f'existing_df_ts_type = {existing_df_ts_type}')

  logger.info(f'existing_df  2= \n{existing_df}')
  logger.info(f'existing_df cols = {existing_df.columns}')
  logger.info(f'cols = {df.columns}')
  val0 = df.index
  logger.info(f'val0 = {val0}')

  # Find new timestamps that don't exist in the existing data
  new_timestamps = df[~df.index.isin(existing_df_ts)]
  logger.info(f'new_timestamps = {new_timestamps}')
  for idx, ts in new_timestamps.iterrows():
    values_list = [idx] + ts.tolist()
    existing_df.loc[len(existing_df.index)] = values_list

    logger.info(f'idx  ={idx}; row = {values_list}')

  # Append only the new rows to the existing DataFrame
  # df_to_add = pd.concat([existing_df, new_timestamps]).drop_duplicates()
  # logger.info(f'df_to_add = \n{df_to_add}')
  logger.info(f'existing_df 3 = \n{existing_df}')
  return existing_df


def write_updated_daily_sum_to_xl(updated_df, greeks_xl_file_path):
  does_file_exist = os.path.exists(greeks_xl_file_path)
  writer = None
  if does_file_exist:
    writer = pd.ExcelWriter(greeks_xl_file_path, engine="openpyxl", mode='w')
  else:
    writer = pd.ExcelWriter(greeks_xl_file_path, engine="openpyxl", mode='w')

  updated_df.columns = ['Flag', 'c', 'p', 'c', 'p', 'c', 'p']

  # Convert the dataframe to an openpyxl Excel object.
  updated_df.to_excel(writer, sheet_name="Sheet1", header=True, index=False)

  # Get the openpyxl workbook and worksheet objects.
  workbook = writer.book
  worksheet = workbook["Sheet1"]

  # Get the dimensions of the dataframe.
  (max_row, max_col) = updated_df.shape
  max_row += 3

  for name in 'BCDEFG':
    cell_range = f'{name}3:{name}{max_row}'
    worksheet.conditional_formatting.add(
        cell_range,
        ColorScaleRule(
            start_type='min',
            start_color='CC0000',
            mid_type='percentile',
            mid_value=50,
            mid_color='FFFFFF',
            end_type='percentile',
            end_value=100,
            end_color='006600'))

  writer.close()


def main(args):
  utils.set_pandas_options()
  stock_list = get_stock_list(args.num_symbols)
  eligible_options = []
  data_downloader = nse.NiftyDataDownloader()
  logger.info(f'stock_list = {stock_list}')
  for stock in stock_list:
    if args.load_latest_parquet:
      latest_file = util_files.get_latest_file_in_folder(
          '', SAVE_FOLDER, 'parquet')
      logger.info(f'Stock: {stock}; latest_file = {latest_file}')
      historic_data = historic_data = pd.read_parquet(latest_file)

      json_data = None
    else:
      # Check eligibility and get options data
      historic_data, json_data = option_utils.get_historic_and_json_data_for_option_instr(
          data_downloader, stock, type='indices', remove_columns=True)

      logger.info(f'historic_data = \n{historic_data.to_string()}')
      # logger.info(f'historic_data shape: {historic_data.shape}')

      save_option_df(historic_data, stock, args.dry_run)

    calc_iv_greeks_into_xl(args, historic_data)
    # logger.info(f'json_data = \n{json_data}')


if __name__ == '__main__':
  args = get_args()
  logger.info(f'Args = {args}')
  main(args)
  # main2(args)
