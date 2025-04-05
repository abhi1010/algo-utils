from dateutil.parser import *
from datetime import date
import json
import csv
import argparse
import os
from datetime import datetime, timedelta
import concurrent.futures
import pickle
import shutil

import yfinance as yf
from finviz.screener import Screener
import pandas as pd
import pandas_ta as ta
import requests
from prettytable import PrettyTable
from trading.data import nifty_data as nse
from nselib import derivatives
from openpyxl.styles import NamedStyle
from openpyxl.utils import get_column_letter, column_index_from_string
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule, FormulaRule

from trading.common import utils, util_files
from trading.options.option_utils import *
from trading.data import json_db


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--dry-run", "-d", action='store_true', help="No real buying")
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

# DIR_FOR_SAVING = 'logs/volumes'
DIR_FOR_SAVING = 'data/volumes'
DATA_DIR_FOR_SAVING = 'data/volumes'

FILE_FORMAT = (
    "dry-" if args0.dry_run else "") + "unusual_options__%Y-%m-%d__%H-%M-%S"
now_s = datetime.now().strftime(FILE_FORMAT)

# SAVE_FOLDER = 'data/volumes-dry' if args0.dry_run else 'data/volumes'
SAVE_FOLDER = 'data/volumes' if args0.dry_run else 'data/volumes'
COL_UNUSUAL_VOL = 'UnusualVol'
UNUSUAL_VOLUME_THRESHOLD = 4
ROUNDING = 2

pd.options.display.float_format = '{:.2f}'.format
logger = utils.get_logger(now_s)

logger.info(f'File = {now_s}; SAVE_FOLDER = {SAVE_FOLDER}')
'''
Usage:

python trading/options/unusual_volume.py --num-symbols 0 --timestamp "04-Oct-2023"
'''

# My version of Automated TA for NSE
# https://www.reddit.com/r/algotrading/comments/ldkt1z/options_trading_with_automated_ta/
# Summary: https://docs.google.com/document/d/1sm-fb_Jq-bF1UMkwflFBI5_ZN6WqfIJzaGeLVrGl4uM/edit
'''
 Few notes for myself
 1. Buy Today, Sell Tomorrow
 2. Buy >= 6 BS score
 3. Conditions for BS score
    RSI <=40 (14 days)
    Volume >=100
    Filled price <= Lower Bollinger band (5 days, with 2 SD)
    SMA ( 5 days) <= VWAP
    Spread >=0.05 (This might change in future) (Use historical spread perhaps for finding out)
    Filled price = Current Bid
    IV<=40
    Today gain <= 0
4. One interesting comment from fellow reader

    VWAP is not supposed to be calculated on daily data. This means you also need to pull 1min or 2min data from Yahoo (depending on your range) for each stock to calculate it properly.
    You don't seem to have the right settings for some of the TA indicators (e.g., RSI, SMA) based on what DJ has posted here.
    BBand Lower is supposed to be compared to the last filled price in the B-Score calc, not the prior close price.
    You're definitely not supposed to be summing the Volume numbers across several days. If you want to factor in more than one day I guess you could average, but I am pretty sure DJ is just taking the volume from either the prior day or the current day (probably the latter).
    The spread between his ideal buy/sell was on average 68% of the ask-bid spread from the data I examined, so using 60% for the value (as you do in your function) is a bit off. There is also contract-to-contract variation in how the spread is allocated that I have not been able to model, but this variation may not matter much given its small effect size.


    Similarly, you'll probably want to get current contract data (e.g., bid, ask, last, etc.) from your broker, rather than Yahoo since that is who you will be buying from.
    Finally, you do not seem to be filtering out options contracts with incomplete data (e.g., having no values for Bid, Ask, or IV), which needs to be done for the tests to apply properly.
5. Ignore everything with insider trading

'''

# how to get FnO stock list
# https://archives.nseindia.com/content/fo/fo_mktlots.csv
# best gainers here: https://www.moneycontrol.com/stocks/fno/marketstats/futures/gainers/homebody.php?opttopic=&optinst=allfut&sel_mth=all&sort_order=0

# URL of the CSV file
URL = 'https://archives.nseindia.com/content/fo/fo_mktlots.csv'


def get_stock_list(num=0):
  fno_list = get_fno_list(URL)
  if num:
    return fno_list[:num]
  return fno_list


def display_all_options_with_unusual_volume(options_holders):

  for option in options_holders:
    if not option.Valid:
      continue

    data = option.HistoricData

    # Check if data contains UnUsualVol column
    if COL_UNUSUAL_VOL not in data.columns:
      return

    rows = data[(data[COL_UNUSUAL_VOL] > UNUSUAL_VOLUME_THRESHOLD)]
    if len(rows):
      logger.info(f'UnusualVol = \n{rows.to_string()}')


def process_option(option):
  highlighted_rows = pd.DataFrame()
  data = option.HistoricData

  if data['Close'].shape[0] <= 4:
    logger.info(f'^^^^^ Skipping as shape is bad. option={option}')
    option.Valid = False
    return highlighted_rows

  # Assuming 'data' is your DataFrame
  data[COL_UNUSUAL_VOL] = 0  # Initialize the new column with 0

  # find the distance between STRIKE_PRICE and UNDERLYING_VALUE, in %
  data['Diff'] = (data['STRIKE_PRICE'] -
                  data['UNDERLYING_VALUE']) / data['UNDERLYING_VALUE'] * 100
  data['Change_OI_Pct'] = data['CHANGE_IN_OI'] * 100 / data['OPEN_INT']

  # Calculate the average trading volume excluding zero values
  average_volume = data[data['TOT_TRADED_QTY'] > 0]['TOT_TRADED_QTY'].mean()

  # Calculate the ratio of current trading volume to previous mean for filtered rows
  data[COL_UNUSUAL_VOL] = data.apply(
      lambda row: row['TOT_TRADED_QTY'] / average_volume
      if row['TOT_TRADED_QTY'] > 0 else 0.0,
      axis=1)

  highlighted_rows = data[(data[COL_UNUSUAL_VOL] > UNUSUAL_VOLUME_THRESHOLD)]
  return highlighted_rows


def check_for_volumes(eligible_options):
  highlighted_rows = pd.DataFrame()

  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit tasks for each option in eligible_options
    futures = {
        executor.submit(process_option, option): option
        for option in eligible_options
    }

    # Collect results
    for future in concurrent.futures.as_completed(futures):
      result = future.result()
      if not result.empty:
        highlighted_rows = pd.concat([highlighted_rows, result])

  sorted_data = highlighted_rows.sort_values(
      by=COL_UNUSUAL_VOL, ascending=False)
  return sorted_data


def update_summary_excel(df, dry_run):
  df_filtered = df[(df['SymbolCount'] < 4) & (abs(df['Diff']) > 8)
                   & (df['UnusualVol'] > 4)]

  unusual_vol_xl_file_path = get_unusual_vo_xl_path(dry_run, False)
  backup_xl_file_path = get_unusual_vo_xl_path(dry_run, True)
  create_backup(unusual_vol_xl_file_path, backup_xl_file_path)
  updated_unusual_volume = get_filtered_vol_by_ts(
      unusual_vol_xl_file_path, df_filtered)
  logger.info(f'updated_unusual_volume = {updated_unusual_volume}')
  logger.info(f'unusual_vol_xl_file_path = {unusual_vol_xl_file_path}')
  logger.info(f'backup_xl_file_path = {backup_xl_file_path}')
  if not updated_unusual_volume.empty:

    write_updated_volumes_to_xl(
        updated_unusual_volume, unusual_vol_xl_file_path)
  else:
    write_updated_volumes_to_xl(df_filtered, unusual_vol_xl_file_path)


def write_updated_volumes_to_xl(updated_df, volumes_xl_file_path):
  does_file_exist = os.path.exists(volumes_xl_file_path)
  writer = None
  if does_file_exist:
    writer = pd.ExcelWriter(volumes_xl_file_path, engine="openpyxl", mode='w')
  else:
    writer = pd.ExcelWriter(volumes_xl_file_path, engine="openpyxl", mode='w')

  # Set the number format for the entire column A to display dates as YYYY-MM-DD
  date_format = NamedStyle(name='date_format')
  date_format.number_format = 'YYYY-MM-DD'
  # Set the number format for the entire column B to display numbers with 2 decimals
  number_format = NamedStyle(name='number_format')
  number_format.number_format = '0.00'
  df_to_write = pd.DataFrame()
  df_to_write['TIMESTAMP'] = updated_df['TIMESTAMP']
  df_to_write['SYMBOL'] = updated_df['SYMBOL']
  df_to_write['EXPIRY_DT'] = updated_df['EXPIRY_DT']
  df_to_write['STRIKE_PRICE'] = updated_df['STRIKE_PRICE']
  df_to_write['OPTION_TYPE'] = updated_df['OPTION_TYPE']
  df_to_write['CLOSING_PRICE'] = updated_df['CLOSING_PRICE']
  df_to_write['OPEN_INT'] = updated_df['OPEN_INT']
  df_to_write['UnusualVol'] = updated_df['UnusualVol']
  df_to_write['Diff'] = updated_df['Diff']
  df_to_write['Change_OI_Pct'] = updated_df['Change_OI_Pct']
  df_to_write['SymbolCount'] = updated_df['SymbolCount']
  logger.info(f'updated_df  = \n{updated_df}')
  logger.info(f'df_to_write  = \n{df_to_write}')
  # Convert the dataframe to an openpyxl Excel object.
  df_to_write.to_excel(writer, sheet_name="Sheet1", header=True, index=False)

  # Get the openpyxl workbook and worksheet objects.
  workbook = writer.book
  worksheet = workbook["Sheet1"]

  # Get the dimensions of the dataframe.
  (max_row, max_col) = df_to_write.shape
  max_row += 3

  for cell in worksheet['A']:
    cell.style = date_format

  for ws_col in ['E', 'H', 'I', 'J']:
    for cell in worksheet[ws_col]:
      cell.style = number_format

  for ws_col in ['H', 'I']:
    if ws_col == 'AP':
      break
    cell_range = f'{ws_col}2:{ws_col}{max_row}'

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

  # Freeze the first row (row 2 as indexing starts from 1)
  worksheet.freeze_panes = 'A2'

  # Autofit the width of all columns based on content length
  for column_cells in worksheet.columns:
    max_length = 0
    column = column_cells[0].column_letter
    for cell in column_cells:
      try:
        if len(str(cell.value)) > max_length:
          max_length = len(cell.value)
      except TypeError:
        pass
    # adjusted_width = (max_length + 2) * 1.2  # Adjusted width for padding

    adjusted_width = (max_length) * 1.1  # Adjusted width for padding
    adjusted_width = max(adjusted_width, 10)
    worksheet.column_dimensions[column].width = adjusted_width

  worksheet.column_dimensions['A'].width = 20

  writer.close()


def get_filtered_vol_by_ts(volumes_df_file, df):
  if not os.path.exists(volumes_df_file):
    return pd.DataFrame()
  # Load existing data from Excel
  logger.info(f'volumes_df_file = {volumes_df_file}')
  existing_df = pd.read_excel(
      volumes_df_file, engine='openpyxl', sheet_name="Sheet1")
  # logger.info(f'existing_df 1 = \n{existing_df}')
  # logger.info(f'df = \n{df}; shape={df.shape}')
  # existing_df = existing_df.reset_index(drop=True)

  existing_df_ts_name = 'TIMESTAMP'

  existing_df_ts = pd.to_datetime(existing_df[existing_df_ts_name])
  logger.info(f'vals = {existing_df_ts}')
  existing_df_ts_type = type(existing_df_ts.iloc[0])
  logger.info(
      f'AA: existing_df_ts_type 0 : {existing_df_ts[2]}; '
      f'existing_df_ts_type = {existing_df_ts_type}')

  logger.info(f'existing_df  2= \n{existing_df}')
  val0 = df.TIMESTAMP
  logger.info(f'val0 = {val0}')

  # Find new timestamps that don't exist in the existing data
  new_timestamps = df[~df.TIMESTAMP.isin(existing_df_ts)]
  logger.info(f'new_timestamps = \n{new_timestamps}')
  columns_indices = existing_df.columns.to_list()

  for idx, ts in new_timestamps.iterrows():
    values_list = ts.tolist()
    len_values = len(values_list)
    len_cols = len(existing_df.columns)

    values_filtered = []
    for col in ['TIMESTAMP', 'SYMBOL', 'EXPIRY_DT', 'STRIKE_PRICE',
                'OPTION_TYPE', 'CLOSING_PRICE', 'OPEN_INT', 'UnusualVol',
                'Diff', 'Change_OI_Pct', 'SymbolCount']:
      values_filtered.append(ts[col])

    existing_df.loc[len(existing_df.index)] = values_filtered

  # Append only the new rows to the existing DataFrame
  # df_to_add = pd.concat([existing_df, new_timestamps]).drop_duplicates()
  # logger.info(f'df_to_add = \n{df_to_add}')
  return existing_df


def create_backup(orig_file, new_file):
  if os.path.exists(orig_file):
    logger.info(f'copying {orig_file} to {new_file}')
    shutil.copyfile(orig_file, new_file)


def get_unusual_vo_xl_path(dry_run, add_suffix: bool):
  file_prefix = "dry-" if dry_run else ""
  backup_suffix = ''
  if add_suffix:
    today = datetime.today()
    backup_suffix = f"_{today.strftime('%Y%m%d')}"
  volumes_xl_file = os.path.join(
      SAVE_FOLDER, f"{file_prefix}volumes_filtered{backup_suffix}.xlsx")
  return volumes_xl_file


def save_to_excel(df, dry_run):

  # Create a Pandas dataframe from some data.
  # df = pd.DataFrame({'Data': [10, 20, 30, 20, 15, 30, 45]})

  # Create a Pandas Excel writer using XlsxWriter as the engine.
  os.makedirs(DIR_FOR_SAVING, exist_ok=True)
  # date in string format: YYYY-MM-DD
  date_s = datetime.now().strftime('%Y-%m-%d__%H-%M')
  volume_df_file = os.path.join(
      DIR_FOR_SAVING,
      ("dry-" if dry_run else "") + f"unusual_volume_{date_s}.xlsx")

  symbol_counts = df['SYMBOL'].value_counts()
  logger.info(f'symbol_counts = {symbol_counts}')
  df['SymbolCount'] = df['SYMBOL'].map(symbol_counts)

  # Set the number format for the entire column A to display dates as YYYY-MM-DD
  date_format = NamedStyle(name='date_format')
  date_format.number_format = 'YYYY-MM-DD'
  # Set the number format for the entire column B to display numbers with 2 decimals
  number_format = NamedStyle(name='number_format')
  number_format.number_format = '0.00'

  writer = pd.ExcelWriter(volume_df_file, engine="openpyxl", mode='w')

  # Convert the dataframe to an XlsxWriter Excel object.
  df.to_excel(writer, sheet_name="Sheet1")

  # Get the xlsxwriter workbook and worksheet objects.
  workbook = writer.book
  worksheet = workbook["Sheet1"]

  # Get the dimensions of the dataframe.
  (max_row, max_col) = df.shape

  # Apply a conditional format to the required cell range.
  coloring = {
      'type': '3_color_scale',
      'min_value': 0,
      'mid_value': 50,
      'max_value': 100,
      'min_color': 'CC0000',
      'mid_color': 'white',
      'max_color': '006600'
  }

  # worksheet.conditional_format(f'Z2:Z{max_row}', coloring)
  # worksheet.conditional_format(f'AA2:AA{max_row}', coloring)
  # worksheet.conditional_format(f'AB2:AB{max_row}', coloring)
  # worksheet.conditional_format(f'AD2:AD{max_row}', coloring)

  # Define a style with number format set to two decimal places
  decimal_style = NamedStyle(name='decimal_style', number_format='0.00')

  # Autofit the width of all columns based on content length
  for column_cells in worksheet.columns:
    max_length = 0
    column = column_cells[0].column_letter
    for cell in column_cells:
      try:
        if len(str(cell.value)) > max_length:
          max_length = len(cell.value)
      except TypeError:
        pass
    # adjusted_width = (max_length + 2) * 1.2  # Adjusted width for padding

    adjusted_width = (max_length) * 1.1  # Adjusted width for padding
    adjusted_width = max(adjusted_width, 10)
    worksheet.column_dimensions[column].width = adjusted_width

  worksheet.column_dimensions['B'].width = 20

  # Columns to hide
  columns_to_hide = [
      'A', 'C', 'E', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
      'T', 'V', 'W', 'Y'
  ]

  # Get the column names from 'A' to 'BC'

  columns_for_num_format = [
      9, 10, 11, 12, 13, 14, 15, 17, 21, 22, 23, 24, 26, 27, 28
  ]
  # Get the column names from 'A' to 'BC'
  column_names_for_num_format = [
      get_column_letter(col_idx) for col_idx in columns_for_num_format
  ]
  for ws_col in column_names_for_num_format:
    for cell in worksheet[ws_col]:
      cell.style = number_format

  for cell in worksheet['A']:
    cell.style = date_format
  for cell in worksheet['B']:
    cell.style = date_format

  # Hide specified columns
  for col in columns_to_hide:
    col_index = column_index_from_string(col)  # Convert column letter to index
    worksheet.column_dimensions[col].hidden = True

  # Close the Pandas Excel writer and output the Excel file.
  writer.close()
  logger.info(f'Saved into excel file 1: {volume_df_file}')


def process_stock(stock):
  data_downloader = nse.NiftyDataDownloader()
  return get_eligible_option_data_for_stock(data_downloader, stock, debug=False)


def get_eligible_options():
  stock_list = get_stock_list(args.num_symbols)
  eligible_options = []

  with concurrent.futures.ThreadPoolExecutor() as executor:
    # Submit tasks for each stock in the stock list
    futures = {
        executor.submit(process_stock, stock): stock for stock in stock_list
    }

    # Collect results
    for future in concurrent.futures.as_completed(futures):
      stock_option_holders = future.result()
      eligible_options.extend(stock_option_holders)
  return eligible_options


def main(args):
  utils.set_pandas_options()

  if not args.dry_run:
    # if not False:
    eligible_options = get_eligible_options()
    logger.info(
        f'main: completed options data with len = {len(eligible_options)}')
    if len(eligible_options) == 0:
      logger.info("No data to use")
      return

    highlighted_rows = check_for_volumes(eligible_options)
    # json_db.save_df(highlighted_rows)

    display_all_options_with_unusual_volume(eligible_options)

    # What is the highest TIMESTAMP value
    highlighted_rows['TS'] = highlighted_rows['TIMESTAMP']
    highlighted_rows['TIMESTAMP'] = pd.to_datetime(
        highlighted_rows['TIMESTAMP'])

    highest_timestamp = highlighted_rows['TIMESTAMP'].max(
    ) if args.timestamp.lower() == 'max' else args.timestamp
    lowest_timestamp = highlighted_rows['TIMESTAMP'].min()
    logger.info(
        f'highest_timestamp = {highest_timestamp}; lowest_timestamp = {lowest_timestamp}'
    )
    filtered_highlighted_rows = highlighted_rows[(
        highlighted_rows['TIMESTAMP'] == highest_timestamp
    )] if args.timestamp.lower() == 'max' else highlighted_rows[(
        highlighted_rows['TS'] == args.timestamp)]
  else:
    # read from file
    os.makedirs(DATA_DIR_FOR_SAVING, exist_ok=True)
    latest_pickled_file = util_files.get_latest_file_in_folder(
        '', DATA_DIR_FOR_SAVING, 'pickle')
    logger.info(f'latest_pickled_file = {latest_pickled_file}')
    filtered_highlighted_rows = pd.read_pickle(latest_pickled_file)

  if not args.dry_run:
    # save filtered_highlighted_rows as pickle file where filename format is YYYY-MM-DD_HH-MM.pickle
    file_name_prefix = datetime.now().strftime('%Y-%m-%d_%H-%M')
    filtered_highlighted_rows.to_pickle(
        os.path.join(
            DATA_DIR_FOR_SAVING, f'{file_name_prefix}_unusual_volume.pickle'))
  try:
    logger.info(f'Saving to excel')
    save_to_excel(filtered_highlighted_rows, args.dry_run)
    logger.info(f'Saving to excel 2')
    update_summary_excel(filtered_highlighted_rows, args.dry_run)
    logger.info(f'Saving to excel 3')
  except Exception as ex:
    logger.exception(f'Exception in update_summary_excel : {str(e)}')

  logger.info(f'highlighted_rows = \n{filtered_highlighted_rows.to_string()}')


if __name__ == '__main__':
  args = get_args()
  logger.info(f'Args = {args}')
  main(args)
