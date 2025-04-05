from dateutil.parser import *
from datetime import date
import json
import csv
import argparse
import os
import logging
from datetime import datetime, timedelta
import traceback

import yfinance as yf
from finviz.screener import Screener
import pandas as pd
import pandas_ta as ta
import requests
from prettytable import PrettyTable
from trading.data import nifty_data as nse
from nselib import derivatives
from tabulate import tabulate

from trading.common import utils, util_prints

pd.options.display.float_format = '{:.2f}'.format

# logger = utils.get_logger(__name__)
logger = logging.getLogger(__name__)

INSTR_TYPE_MAPPING = {'equities': 'OPTSTK', 'indices': 'OPTIDX'}


def tabulate_option_holders(option_holders):

  headers = [
      "Ticker", "Strike", "Expiry", "OptionType", "FilledPrice", "Bid/Ask",
      "IV", "Delta", "Theta", "Gamma"
  ]

  option_holders = sorted(option_holders, key=lambda x: x.Strike)
  table_calls = [opt.to_list() for opt in option_holders if opt.IsCall]
  table_puts = [opt.to_list() for opt in option_holders if not opt.IsCall]
  logger.info(
      '\n' + tabulate(table_calls, headers=headers, tablefmt="rounded_grid"))
  logger.info(
      '\n' + tabulate(table_puts, headers=headers, tablefmt="rounded_grid"))


class OptionHolder:

  def __init__(self):
    self.Valid = True
    self.Ticker = ""
    self.Strike = 0.0
    self.Expiry = None
    self.YahooString = ""
    self.IsCallCE_Or_PE = None

    self.BidPrice = 0.0
    self.AskPrice = 0.0
    self.FilledPrice = 0.0
    self.FilledAmount = 0.0
    self.CurrentSpread = 0.0
    self.Volume = 0
    self.OpenInterest = 0
    self.PercentChange = 0.0
    self.IV = 0.0
    self.VWAP = 0.0

    self.IdealBuy = 0.0
    self.IdealSell = 0.0

    self.BBandHigh = 0.0
    self.BBandLow = 0.0
    self.RSI = 0.0
    self.SMA = 0.0
    self.BuyScore = 0

    self.PrevClose = 0.0

    self.HistoricData = None

    self.Theta = 0.0
    self.Delta = 0.0
    self.Gamma = 0.0

  def to_list(self):
    return [
        self.Ticker, self.Strike, self.Expiry, 'C' if self.IsCall else 'P',
        f"{self.FilledPrice:.2f}", f"{self.BidPrice:.2f}/{self.AskPrice:.2f}",
        self.IV, f"{self.Delta:.2f}", f"{self.Theta:.2f}", f"{self.Gamma:.2f}"
    ]

  def to_dict(self):
    return {
        "Ticker": self.Ticker,
        "Strike": self.Strike,
        "Expiry": self.Expiry,
        "OptionType": str(self.IsCall),
        "FilledPrice": f"{self.FilledPrice:.2f}",
        "Bid/Ask": f"{self.BidPrice:.2f}/{self.AskPrice:.2f}",
        "IV": self.IV,
        "Delta": f"{self.Delta:.2f}",
        "Theta": f"{self.Theta:.2f}",
        "Gamma": f"{self.Gamma:.2f}"
    }

  def __repr__(self):
    return str(self)

  def simple_str(self):
    return (
        f"OptionHolder<S>(Ticker={self.Ticker}, Strike={self.Strike} , Expiry={self.Expiry}, "
        f"OptionType={self.IsCallCE_Or_PE},  "
        f"FilledPrice={self.FilledPrice:.2f}, "
        f"Bid/Ask={self.BidPrice:.2f}/{self.AskPrice:.2f}, "
        f"IV={self.IV}, "
        f"Delta={self.Delta:.2f}, "
        f"Theta={self.Theta:.2f}, "
        f"Gamma={self.Gamma:.2f}, ")

  def __str__(self):
    return (
        f"OptionHolder(Ticker={self.Ticker}, Strike={self.Strike}, Expiry={self.Expiry}, "
        f"YahooString={self.YahooString}, OptionType={self.IsCallCE_Or_PE}, BidPrice={self.BidPrice}, "
        f"AskPrice={self.AskPrice}, FilledPrice={self.FilledPrice}, PrevClose={self.PrevClose}, "
        f"FilledAmount={self.FilledAmount}, "
        f"CurrentSpread={self.CurrentSpread}, Volume={self.Volume}, OpenInterest={self.OpenInterest}, "
        f"PercentChange={self.PercentChange}, IV={self.IV}, VWAP={self.VWAP}, "
        f"IdealBuy={self.IdealBuy}, IdealSell={self.IdealSell}, "
        f"BBandHigh={self.BBandHigh}, BBandLow={self.BBandLow}, RSI={self.RSI}, SMA={self.SMA}, "
        f"BuyScore={self.BuyScore})")


def get_yfinance_ticker_name(ticker):
  return ticker if ticker.endswith('.NS') else ticker + '.NS'


def get_nse_simple_name(ticker):
  return ticker if not ticker.endswith('.NS') else ticker.replace('.NS', '')


def get_fno_list(url):
  # Download the CSV file
  response = requests.get(url)
  if response.status_code == 200:
    content = response.content.decode('utf-8')

    # Parse the CSV content
    lines = content.split('\n')
    reader = csv.reader(lines)

    # Retrieve the second column into a list
    second_column_data = [
        row[1].strip() if row[1].endswith('.NS') else row[1].strip() + '.NS'
        for row in reader
        if row
    ][6:]

    full_fno_list = get_average_trading_values(second_column_data)
    top_tickers = get_top_tickers_for_avg_value(full_fno_list)
    logger.info(f'top_tickers = {top_tickers}; ')
    logger.info(
        f'full_fno_list len = {len(full_fno_list)}; '
        f'len top_tickers= {len(top_tickers)}')

    # Print the second column data
    # return second_column_data

    # return ['INDIACEM']
    return [get_nse_simple_name(t) for t in top_tickers]

    return ['INDIACEM', 'ABFRL', 'BHARATFORG', 'IBULHSGFIN', 'TRENT']
  else:
    logger.info("Failed to download the file.")
  return None


def get_average_trading_values(tickers):
  trading_values = dict()
  for ticker in tickers:

    stockTicker = yf.Ticker(
        ticker if ticker.endswith('.NS') else (ticker + '.NS'))

    df = stockTicker.history(period="10d", interval="1d")

    # Calculate trading value
    df['Trading Value'] = df['Close'] * df['Volume']

    # Calculate average trading volume
    df['Avg Vol'] = df['Volume'].rolling(window=5).mean()

    # Calculate average trading value
    df['Avg Value'] = df['Trading Value'].rolling(window=5).mean()
    latest_avg_trading_value = df.iloc[-1]['Avg Value']
    trading_values[ticker] = latest_avg_trading_value
  return trading_values


def get_top_tickers_for_avg_value(tickers_with_avg_trading_values):
  data = tickers_with_avg_trading_values
  # Calculate the threshold for the top 60% values
  threshold = sorted(data.values())[int(len(data) * 0.4)]

  # Filter the dictionary to keep only entries with values above the threshold
  filtered_dict = {
      ticker: value for ticker, value in data.items() if value >= threshold
  }
  return filtered_dict


def get_eligible_option_data_for_stock(data_downloader, ticker, debug=True):
  return get_eligible_option_data_for_instrument_as_option_holders(
      data_downloader, ticker, 'equities', debug)


def get_eligible_option_data_for_index(data_downloader, ticker, debug=True):
  return get_eligible_option_data_for_instrument_as_option_holders(
      data_downloader, ticker, 'indices', debug)


def get_historic_data_for_option(
    json_data, ticker, type, remove_columns, days=45):
  expiry_dates = json_data["expiryDates"]

  # Get the current date
  current_date = datetime.now().date()

  # Calculate the date from 45 days agoß
  days_ago = timedelta(days=days)
  date_45_days_ago = (current_date - days_ago).strftime('%d-%m-%Y')

  # Calculate the date for tomorrow
  date_tomorrow = (current_date + timedelta(1)).strftime('%d-%m-%Y')

  historicData = get_daily_data_for_option(
      ticker, date_tomorrow, date_45_days_ago, type)
  return fix_historic_data_columns(historicData, remove_columns)


def fix_historic_data_columns(historicData, remove_columns):
  historicData['STRIKE_PRICE'] = historicData['STRIKE_PRICE'].astype(float)
  historicData['Close'] = historicData['CLOSING_PRICE'].astype(float)
  historicData['Low'] = historicData['TRADE_LOW_PRICE'].astype(float)
  historicData['High'] = historicData['TRADE_HIGH_PRICE'].astype(float)
  historicData['Volume'] = historicData['TOT_TRADED_QTY'].astype(float)
  historicData['PREV_CLS'] = historicData['PREV_CLS'].astype(float)
  historicData['TOT_TRADED_QTY'] = historicData['TOT_TRADED_QTY'].astype(float)
  historicData['CHANGE_IN_OI'] = historicData['CHANGE_IN_OI'].astype(float)
  historicData['OPEN_INT'] = historicData['OPEN_INT'].astype(int)

  historicData.index = pd.to_datetime(historicData.index)

  if remove_columns:
    reduce_historic_data_columns(
        historicData,
        ['MARKET_TYPE', 'OPENING_PRICE', 'TRADE_HIGH_PRICE', 'TRADE_LOW_PRICE'])
  # MARKET_TYPE OPENING_PRICE TRADE_HIGH_PRICE TRADE_LOW_PRICE CLOSING_PRICE
  # LAST_TRADED_PRICE  PREV_CLS SETTLE_PRICE  TOT_TRADED_QTY TOT_TRADED_VAL
  # OPEN_INT  CHANGE_IN_OI MARKET_LOT  UNDERLYING_VALUE
  # Close     Low    High  Volume

  return historicData


def reduce_historic_data_columns(df, columns):
  # remove all columns in dataframe
  df.drop(columns, axis=1, inplace=True)


def get_historic_and_json_data_for_option_instr(
    data_downloader, ticker, type, remove_columns):

  ticker_data_s = data_downloader.get_ticker_option_data(ticker, type=type)

  try:
    json_data = json.loads(ticker_data_s)['records']
  except Exception as e:
    tb_info = traceback.format_exc()
    logger.info(
        f'Exception: {str(e)}; ticker: {ticker}; ticker_data_s={ticker_data_s}; '
        f'Exception tb: {tb_info}')
    raise e
  # logger.info(f'json_data = {json_data}')

  historicData = get_historic_data_for_option(
      json_data, ticker, type, remove_columns=remove_columns)

  return historicData, json_data


def get_eligible_option_data_for_instrument_as_option_holders(
    data_downloader, ticker, type, debug):
  """Populates a list of OptionHolder objects from the given JSON data.

  Args:
    json_data: The JSON data containing the option chain data.

  Returns:
    A list of OptionHolder objects.
  """
  assert type in [
      'equities', 'indices'
  ], 'type must be either equities or indices'

  option_holders = []
  historicData, json_data = get_historic_and_json_data_for_option_instr(
      data_downloader, ticker, type, remove_columns=False)
  logger.info(f'historicData = \n{historicData.to_string()}')
  logger.info(f'\n\n')

  for data_dict in json_data["data"]:
    logger.info(f'-' * 80)
    logger.info(f'data_dict =  {data_dict.keys()}')
    logger.info(f'pretty print = {util_prints.get_pretty_print(data_dict)}')

    for option_type in ["CE", "PE"]:

      strike_price = data_dict['strikePrice']
      expiry_date = data_dict['expiryDate']
      option_data = data_dict.get(option_type, {})

      opt_data_strk_exp = historicData.loc[
          (historicData['EXPIRY_DT'].str.upper() == expiry_date.upper())
          & (historicData['STRIKE_PRICE'] == strike_price)]

      if option_data is not None and len(option_data):
        option_type_df = opt_data_strk_exp[opt_data_strk_exp['OPTION_TYPE'] ==
                                           option_type].iloc[::-1]

        if debug:
          logger.info(
              f'Ticker: {ticker}; strike: {strike_price}; '
              f'expiry: {expiry_date}; option_type  = {option_type}; '
              f'Data = \n{option_type_df.to_string()}')

        option_holder = OptionHolder()
        option_holder.Ticker = ticker
        option_holder.Strike = strike_price
        option_holder.Expiry = expiry_date
        option_holder.YahooString = option_data["identifier"]
        option_holder.IsCallCE_Or_PE = option_type
        option_holder.HistoricData = option_type_df

        option_holder.BidPrice = option_data["lastPrice"]
        option_holder.AskPrice = option_data["askPrice"]
        option_holder.FilledPrice = option_data['lastPrice']
        option_holder.FilledAmount = 0
        option_holder.CurrentSpread = option_holder.AskPrice - option_holder.BidPrice
        option_holder.Volume = option_data["totalTradedVolume"]
        option_holder.OpenInterest = option_data["openInterest"]
        option_holder.PercentChange = option_data["pChange"]
        option_holder.IV = option_data["impliedVolatility"]

        option_holder.VWAP = 0

        option_holder.IdealBuy = 0
        option_holder.IdealSell = 0

        option_holder.BBandHigh = 0
        option_holder.BBandLow = 0
        option_holder.RSI = 0
        option_holder.SMA = 0
        option_holder.BuyScore = 0

        option_holders.append(option_holder)

  logger.info(f'option_holders len = {len(option_holders)}')
  logger.info(f'historicData = \n{historicData.to_string()}')

  return option_holders


def get_daily_data_for_option(ticker, date_tomorrow, date_45_days_ago, type):
  instrument = INSTR_TYPE_MAPPING[type]
  data = derivatives.option_price_volume_data(
      symbol=ticker,
      instrument=instrument,
      from_date=date_45_days_ago,
      to_date=date_tomorrow)

  return data


def set_tte(df):
  # Convert date columns to datetime format
  df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%d-%b-%Y')
  df['EXPIRY_DT'] = pd.to_datetime(df['EXPIRY_DT'], format='%d-%b-%Y')

  # Calculate the business days to expiry for each row
  df['TTE_BDays'] = df.apply(
      lambda row: len(
          pd.date_range(start=row['TIMESTAMP'], end=row['EXPIRY_DT'], freq='B')
      ),
      axis=1)

  # Calculate annualized time to expiry for each row based on business days
  trading_days_in_a_year = 252  # Adjust for your market
  df['T'] = df['TTE_BDays'] / trading_days_in_a_year
