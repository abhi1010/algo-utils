import os
import sys
import re
import datetime
import inspect
import logging
from logging.handlers import TimedRotatingFileHandler
from rich.logging import RichHandler
from collections import namedtuple

from email.message import EmailMessage

from markdown2 import Markdown
import pandas as pd
from jinja2 import Environment, FileSystemLoader

import os, sys, re
import locale
import logging
from enum import Enum
import traceback

from pandas import DataFrame, DatetimeIndex, merge, to_datetime, to_timedelta

from trading.common import mailer

from cryptography.fernet import Fernet

locale.setlocale(locale.LC_ALL, 'en_CA.UTF-8')
LOG_DIR = os.path.join(os.getcwd(), 'logs')

GROUP_NAMES = namedtuple('GroupName',
                         ['all', 'common', 'new', 'removed', 'name'])


class Markets(str, Enum):
  NIFTY = 'nifty'
  CRYPTO = 'crypto'
  SPX = 'spx'


class HLOC:
  Col_ADJ_CLOSE = 'Adj Close'
  Col_CLOSE = 'Close'
  Col_High = 'High'
  Col_Low = 'Low'
  Col_Open = 'Open'
  Col_Volume = 'Volume'


TICKER_INTERVAL_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "60m": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
    "m": 1,
    "h": 60,
    "d": 1440,
    "w": 10080,
}


def set_highlighted_excepthook():
  import sys, traceback
  from pygments import highlight
  from pygments.lexers import get_lexer_by_name
  from pygments.formatters import TerminalFormatter

  lexer = get_lexer_by_name("pytb" if sys.version_info.major < 3 else "py3tb")
  formatter = TerminalFormatter()

  def myexcepthook(type, value, tb):
    tbtext = ''.join(traceback.format_exception(type, value, tb))
    sys.stderr.write(highlight(tbtext, lexer, formatter))

  sys.excepthook = myexcepthook


def set_pandas_options():
  pd.options.display.float_format = '{:.2f}'.format
  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)


def check_and_create_directory(dir):
  if not os.path.isdir(dir):
    os.makedirs(dir)


def get_token_name_without_quote_currency(full_token_name):
  return full_token_name.split('-')[0].split('_')[0].split('/')[0]


def print_stack_trace():
  """
    Prints the current call stack trace using the logging module,
    without raising an exception.
    """
  stack_trace = traceback.extract_stack()

  # Log the full stack trace
  logger.info("Current Call Stack Trace:")
  for frame in stack_trace[:
                           -1]:  # Exclude the last frame (this function itself)
    logger.info(
        f"File: {frame.filename}, Line: {frame.lineno}, Function: {frame.name}"
    )
    logger.info(f"  Code: {frame.line.strip()}")


def get_logger(log_name='cpr-btc', should_add_ts=False, use_rich=False):
  logger = logging.getLogger('root')
  if not len(logger.handlers):
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s, %(levelname)s : [%(filename)s:%(lineno)s - %(funcName)20s() ] - %(message)s'
    )

    if use_rich:

      rich_handler = RichHandler(rich_tracebacks=True, )
      rich_handler.setLevel(logging.DEBUG)
      logger.addHandler(rich_handler)
    else:
      # Stream handler for printing to console
      stdout_handler = logging.StreamHandler()
      stdout_handler.setLevel(logging.INFO)
      stdout_handler.setFormatter(formatter)
      logger.addHandler(stdout_handler)

    check_and_create_directory(LOG_DIR)

    if should_add_ts:
      log_name += datetime.datetime.now().strftime("__%Y-%m-%d__%H-%M")

    # Rotating file handler for saving logs to file
    logger_path = os.path.join(LOG_DIR, f'{log_name}.log')
    file_handler = TimedRotatingFileHandler(filename=logger_path,
                                            when='midnight',
                                            backupCount=10)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f'Logger path = {logger_path}')
    print_stack_trace()
  return logger


logger = logging.getLogger('root')


class TimeFrame(str, Enum):
  monthly = 'm'
  weekly = 'w'
  daily = 'd'
  half = '4h'
  hourly = '1h'
  minute_5 = '5m'


# write a function that takes in a date time and returns whether it is 60 minutes of the current time
def is_within_minutes_of_the_current_time(date_time: datetime.datetime,
                                          interval,
                                          offset=datetime.timedelta(hours=9)):

  minutes_to_use = TICKER_INTERVAL_MINUTES[interval]
  current_time = datetime.datetime.now()
  given_time_offset = date_time + offset

  is_it_ok = (current_time - given_time_offset
              ).total_seconds() / minutes_to_use < minutes_to_use
  total_secs = abs((current_time - given_time_offset).total_seconds())
  return total_secs / minutes_to_use < minutes_to_use


def resample_to_interval(dataframe: DataFrame, interval):
  """
    Resamples the given dataframe to the desired interval.
    Please be aware you need to use resampled_merge to merge to another dataframe to
    avoid lookahead bias

    :param dataframe: dataframe containing close/high/low/open/volume
    :param interval: to which ticker value in minutes would you like to resample it
    :return:
    """
  if isinstance(interval, str):
    interval = TICKER_INTERVAL_MINUTES[interval]

  df = dataframe.copy()
  df = df.set_index(DatetimeIndex(df["Date"]))
  ohlc_dict = {
      "Open": "first",
      "High": "max",
      "Low": "min",
      "Close": "last",
      "Volume": "sum",
      "Adj Close": "last"
  }
  # Resample to "left" border as dates are candle open dates
  df = df.resample(str(interval) + "min", label="left").agg(ohlc_dict).dropna()
  df.reset_index(inplace=True)

  return df


def show_numbers(value):
  # return "{:,}".format(value)

  return locale.currency(value, symbol=False, grouping=True)


def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)


def unix_to_time(unix_ts):

  dt = datetime.datetime.fromtimestamp(unix_ts / 1000000000)
  return dt


def compare_lists(lst_today, lst_prev):
  today = set(lst_today)
  prev = set(lst_prev)
  common = today.intersection(prev)
  new = today - common
  removed = prev - common
  return common, new, removed


def find_days_since_given_date(far_date):
  now = datetime.datetime.now()
  days_since_far_date = (now - far_date).days
  return days_since_far_date


def markdown_to_html(text):
  markdowner = Markdown()
  return markdowner.convert(text)


def create_markdown_table(ticker_list, header):
  header_text = f'''| {header} |
| -- |
'''
  row_fmt = '''| {} |
'''
  rows = ''.join([row_fmt.format(ticker) for ticker in ticker_list])
  return header_text + rows


# Create an offset of 10 Business days and 10 hours


def get_business_days_offset(days=-56):
  today = datetime.datetime.today()
  ts = pd.Timestamp(today).date()
  bd = pd.tseries.offsets.BusinessDay(offset=datetime.timedelta(days=days))
  return (bd + ts).to_pydatetime()


def get_crypto_days_offset(days=-56):
  today = datetime.datetime.today().date()

  bd = today + datetime.timedelta(days=days)
  # convert bd into pd datatime
  return pd.to_datetime(bd)


def disable_third_party_logging():
  logging.getLogger('yahoo_fin').setLevel(logging.ERROR)
  logging.getLogger('requests').setLevel(logging.ERROR)
  logging.getLogger('urllib3').setLevel(logging.ERROR)


def compare_lists_to_get_detailed_info(tickers, backup_file):
  tickers = sorted(tickers)
  prev_tickers = []

  if os.path.exists(backup_file):
    with open(backup_file, 'r') as f:
      lines = tmp_tickers = f.readlines()
      if len(lines):
        tmp_tickers = lines[0].split(',')
        print(f'backup_file={backup_file}; tickers orig= {tmp_tickers}')
        prev_tickers = sorted([ticker.strip() for ticker in tmp_tickers])

  logger.info(f'Comparing lists: {tickers}; \n\t prev_tickers={prev_tickers}')
  common, new, removed = compare_lists(tickers, prev_tickers)
  logger.info(f'Comparing lists: {common}; \n\t '
              f'new={new} \n\t removed={removed}')

  with open(backup_file, 'w') as f:
    f.writelines(','.join(tickers))
  return common, new, removed


def names_to_tradingview_naming(tickers, exchange='NSE'):
  if exchange == 'NSE':
    return ','.join([
        'NSE:' +
        x.replace('.NS', '').replace('-', '_').replace('M&MF', 'M_MFIN')
        for x in tickers
    ])
  else:
    return ','.join(['KUCOIN:' + x.replace('/USDT', 'USDT') for x in tickers])


def get_html_for_groups(groups):

  # Set up the Jinja2 environment to load templates from the "templates" directory
  env = Environment(loader=FileSystemLoader('data/templates'))

  # Load the "tickers_table_template.html" template from the "templates" directory
  template = env.get_template('tickers_table_template.html')

  # Render the template with the tickers variable
  html_output = template.render(GROUPS=groups)
  print(f'html_output = {html_output}')

  return html_output


def create_jinja2_data_for_relative_strength(pairs, output_json_file):

  # Create a Jinja2 environment
  env = Environment(loader=FileSystemLoader('trading/resources'))

  # Load the template file
  template = env.get_template("pairlists.template")

  # Render the template with the list of pairs
  output = template.render(pairs=pairs)

  # Write the output to a file
  with open(output_json_file, "w") as f:
    f.write(output)


def send_mail_with_html(html_output, subject):
  mail_util = mailer.Mailer()
  mail_util.send_message(subject, html_output)


def get_date(offset=datetime.timedelta(days=0)):
  dt = datetime.datetime.today() + offset
  return dt


def date_get_short(given_dt=None):
  if not given_dt:
    return datetime.datetime.now().strftime('%Y-%m-%d')
  else:
    return given_dt.strftime('%Y-%m-%d')

  # tickers = ['AAPL', 'GOOGL']

  # compare_and_update_list(tickers, 'data/tickers_tabled.ticks')


def generate_key():
  """Generates and returns a new encryption key."""
  return Fernet.generate_key()


def encrypt_string(key, plaintext):
  """Encrypts a string using the provided key."""
  fernet = Fernet(key)
  encrypted = fernet.encrypt(plaintext.encode())
  return encrypted


def decrypt_string(key, encrypted):
  """Decrypts an encrypted string using the provided key."""
  fernet = Fernet(key)
  decrypted = fernet.decrypt(encrypted).decode()
  return decrypted


if __name__ == '__main__':
  # dt = date_get_short()
  # print(f'dt = {dt}')
  pass
