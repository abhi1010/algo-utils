import os, pandas
from enum import Enum
import argparse

from trading.common import utils

logger = utils.get_logger('screener_common')
import pandas as pd

from trading.services import telegram_runner


class Markets(str, Enum):
  """
  Enum to represent the download format.
  """
  CRYPTO = "crypto"
  NIFTY = "nifty"
  SPX = 'spx'
  NASDAQ = 'nasdaq'


def read_etf_holdings(file_path):
  """
    Read ETF holdings from an Excel file with specific parsing requirements.
    
    Parameters:
    file_path (str): Path to the Excel file
    
    Returns:
    pandas.DataFrame: Cleaned DataFrame of ETF holdings
    """
  try:
    engine = 'calamine'
    # "xlrd", "openpyxl", "odf", "pyxlsb"
    # Read the Excel file from the "Holdings" sheet using xlrd engine for .xls files
    # Skip rows until the header row is found
    does_file_exist = os.path.exists(file_path)
    logger.info(f'file_path = {file_path}; does_file_exist = {does_file_exist}')
    df = pd.read_excel(
        file_path, sheet_name='Holdings', header=None, engine=engine)

    # Find the index of the header row (containing "Ticker Name Sector")
    header_index = df[df.apply(
        lambda row: 'Ticker' in str(row.values) and 'Name' in str(row.values)
        and 'Sector' in str(row.values),
        axis=1)].index[0]

    # Re-read the file, skipping rows before the header and using the correct header
    df = pd.read_excel(
        file_path, sheet_name='Holdings', header=header_index, engine=engine)

    # Clean up the DataFrame
    # Remove any rows with NaN in critical columns
    df = df.dropna(subset=['Ticker', 'Name'])
    # Convert Weight to numeric, removing % sign if present
    if 'Weight (%)' in df.columns:
      df['Weight (%)'] = pd.to_numeric(
          df['Weight (%)'].astype(str).str.rstrip('%'), errors='coerce')

    # Optional: Convert numeric columns
    numeric_columns = ['Market Value', 'Notional Value', 'Quantity', 'Price']
    for col in numeric_columns:
      df[col] = pd.to_numeric(df[col], errors='coerce')

    # Print basic info about the dataset
    print(f"Total number of holdings: {len(df)}")
    print("\nFirst few rows:")
    print(df.head())

    # Optional: Save to CSV
    output_path = file_path.replace('.xls', '_processed.csv')
    df.to_csv(output_path, index=False)
    print(f"\nProcessed holdings saved to {output_path}")

    return df

  except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    raise e
    return None


def process_files(directory, limit=0, make_chart=False, only_one=False):
  dataframes = {}
  # # Get list of CSV files in the directory
  if only_one:
    csv_files = [
        file for file in os.listdir(directory)
        if file.endswith('.csv') and 'VGUARD' in file
    ]
  else:
    # Get list of CSV files in the directory
    csv_files = [
        file for file in os.listdir(directory) if file.endswith('.csv')
    ]
  if limit:
    csv_files = csv_files[:limit]

  logger.info(f'directory = {directory}')
  logger.info(f'csv_files = {csv_files} ; len = {len(csv_files)}')

  for filename in csv_files:
    # logger.info(filename)
    symbol = filename.split(".")[0]

    df = pd.read_csv(os.path.join(directory, filename))
    # only keep data from 2024 jan 01 onwards
    # df = df[df['Date'] >= '2024-01-01']

    # logger.info(f'Read : {symbol}')
    if df.empty:
      continue

    dataframes[symbol] = df

  return dataframes


def filter_away_stale_data(dataframes, market, timeframe):
  date_fmt = '%Y-%m-%d %H:%M:%S' if market == Markets.CRYPTO and timeframe != '1d' else '%Y-%m-%d'

  def get_largest_dt(dataframes):
    largest_dt = datetime.min
    # 2023-04-04 20:00:00 or 2023-04-04

    for symbol, df in dataframes.items():
      try:
        date_s = df['Date'].iloc[-1]
        date_dt = datetime.strptime(date_s, date_fmt)
      except Exception as e:
        logger.info(f'exception: {str(e)}; sym: {symbol}; date_s = {date_s}')
        continue
        raise e
      largest_dt = max(largest_dt, date_dt)
      logger.info(f'Sym: {symbol}; date_s = {date_dt}')

    return largest_dt

  largest_date = get_largest_dt(dataframes)
  logger.info(f'largest_date = {largest_date}')
  only_latest_dataframes = {}

  for symbol, df in dataframes.items():
    date_s = df['Date'].iloc[-1]
    date_dt = datetime.strptime(date_s, date_fmt)
    if date_dt >= largest_date:
      only_latest_dataframes[symbol] = df
    else:
      logger.info(
          f'Sym: {symbol} is too old, skipping; '
          f'date_s = {date_s}; largest_date = {largest_date}')
  return only_latest_dataframes, largest_date


def save_names_to_txt(names, output_dir, market, timeframe, largest_date):
  # file path in the format of data/bbwp/YYYY-MM-DD_tickers.txt
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  # output_file to be date based YYYY-MM-DD_tickers.txt
  # today_s = date.today().strftime("%Y-%m-%d")
  largest_dt_s = largest_date.strftime("%Y-%m-%d_%H-%M")
  output_file = os.path.join(
      output_dir, f'{largest_dt_s}_{market}-{timeframe}-tickers.txt')
  logger.info(f'output_file = {output_file}')
  with open(output_file, 'w') as f:
    f.write(str(names))
  logger.info(f'written into {output_file}')


def get_new_tickers_compared_to_older_file(
    names, output_dir, market, timeframe, largest_date):
  names = sorted(names)

  largest_dt_s = largest_date.strftime("%Y-%m-%d_%H-%M")
  file_name_fmt = f'{market}-{timeframe}-tickers.txt'
  # find the latest 2 files with the given format
  # use glob to find latest 2 files with the file_name_fmt format
  files = glob.glob(os.path.join(output_dir, f'*{file_name_fmt}'))

  if len(files) == 0:
    return names
  files.sort(key=os.path.getmtime)
  logger.info(f'files = {files}; file_name_fmt = {file_name_fmt}')
  if len(files) < 2:
    logger.info(f'Not enough files found for {file_name_fmt}')
    return []
  last_file = files[-2]

  # parse teh file into a list of tickers
  with open(last_file, 'r') as f:
    file_contents = f.read().splitlines()
    # file content it like this: "['KUCOIN:QIUSDT', 'KUCOIN:ARUSDT']". Parse it into a list
    last_file_names = sorted(ast.literal_eval(file_contents[0]))
  logger.info(f'last_file = {last_file}')
  logger.info(f'last_file_names = {last_file_names}')
  new_tickers = list(set(names) - set(last_file_names))
  logger.info(f'new_tickers = {new_tickers}')
  return new_tickers


def transform_tickers(tickers, market):
  if market in [Markets.NIFTY, Markets.SPX, Markets.NASDAQ]:
    return tickers
  kucoin_tickers = [
      'KUCOIN:' + item.split('-')[0].replace('_', '').replace('/', '')
      for item in tickers
  ]
  return kucoin_tickers


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
    prefix = 'NSE:'
    new_tickers_s = (
        prefix + (',' + prefix).join(new_tickers).replace('&', '_')
    ) if len(new_tickers) else ''
    all_tickers_s = (
        prefix + (',' + prefix).join(tickers_for_telegram).replace('&', '_')
    ) if len(tickers_for_telegram) else ''
  else:
    new_tickers_s = ','.join(new_tickers)
    all_tickers_s = ','.join(tickers_for_telegram)

  msg = f'''Args:
  ```python
{namespace_str}```

  New tickers:
  ```
{new_tickers_s}```

  All tickers:
  ```
{all_tickers_s}```'''

  _send_msg(msg)
