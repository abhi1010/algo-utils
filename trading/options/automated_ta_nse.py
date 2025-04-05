from dateutil.parser import *
from datetime import date
import json
import csv
import argparse
import pickle
import os
from datetime import datetime, timedelta

import yfinance as yf
from finviz.screener import Screener
import pandas as pd
import pandas_ta as ta
import requests
from prettytable import PrettyTable
from trading.data import nifty_data as nse
from nselib import derivatives

from trading.common import utils


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--dry-run",
                      "-d",
                      action='store_true',
                      help="No real buying")

  args = parser.parse_args()
  return args


args = get_args()

FILE_FORMAT = ("dry-" if args.dry_run else
               "") + "automated_ta_nse__%Y-%m-%d__%H-%M-%S"
now_s = datetime.now().strftime(FILE_FORMAT)

SAVE_FOLDER = 'data/reports-dry' if args.dry_run else 'data/reports'

ROUNDING = 2

pd.options.display.float_format = '{:.2f}'.format
logger = utils.get_logger(now_s)

logger.info(f'File = {now_s}; SAVE_FOLDER = {SAVE_FOLDER}')

from trading.options.option_utils import *

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


def _find_biggest_gainers(fno_list):
  adj_close = 'Close'
  open = 'Open'
  lst_tickers = [get_yfinance_ticker_name(t) for t in fno_list]
  logger.info(f'tickers to use : {lst_tickers}')

  # Download historical data for the tickers
  data = yf.download(lst_tickers, period='1d')
  # data = yf.get_data(lst_tickers)

  # Reshape the DataFrame
  logger.info(f'-' * 80)
  reshaped_data = pd.DataFrame(data.stack(), columns=[adj_close, open])
  reshaped_data.index.names = ["Date", "Ticker"]

  # Print the reshaped DataFrame
  logger.info(f'reshaped_data = \n{reshaped_data}')

  reshaped_data['percent_change'] = (
      reshaped_data[adj_close] -
      reshaped_data[open]) / reshaped_data[open] * 100
  biggest_gainers = reshaped_data.sort_values(by='percent_change',
                                              ascending=False)

  # Print the tickers with the biggest gainers
  gainers_list = biggest_gainers.head().index.get_level_values(1).to_list()
  return [g.replace('.NS', '') for g in gainers_list]


def get_stock_list(num=3):
  fno_list = get_fno_list(URL)
  gainers_list = _find_biggest_gainers(fno_list)
  logger.info(f"Biggest Gainers: {gainers_list}")
  return gainers_list[:num]


def get_ideal_buy_sell_magic(option):
  option.IdealBuy = round(option.BidPrice + (option.CurrentSpread * .2), 3)
  option.IdealSell = round(option.AskPrice - (option.CurrentSpread * .2), 3)


INDEX_LAST_ROW = -1


def main(args):
  utils.set_pandas_options()
  stock_list = get_stock_list(5)
  prev_valid_options = load_latest_options_holders()
  prev_tickers = list(set([
      x.Ticker for x in prev_valid_options
  ])) if prev_valid_options and len(prev_valid_options) else []

  eligible_options = []
  data_downloader = nse.NiftyDataDownloader()
  for stock in stock_list + prev_tickers:
    # Check eligibility and get options data
    eligible_options.extend(
        get_eligible_option_data_for_stock(data_downloader, stock))

  if len(eligible_options) == 0:
    logger.info("No data to use")
    return

  for option in eligible_options:
    if option.HistoricData['Close'].shape[0] <= 4:
      logger.info(f'^^^^^ Skipping as shape is bad. option={option}')
      option.Valid = False
      continue
    calculate_indicators(option)
    # Additional calculations

    update_option_scores(option)
    logger.info(
        f'option after calculations = {option}; data=\n{option.HistoricData.to_string()}'
    )

  valid_options = [o for o in eligible_options if o.Valid]

  # Sort this shit somehow
  valid_options.sort(key=lambda x: x.BuyScore, reverse=True)

  curr_output_table = create_output_table(valid_options, prev_valid_options)

  save_latest_option_holders(valid_options)
  get_updated_info_from_prev_options_run(eligible_options, prev_valid_options)

  if prev_valid_options:
    prev_output_table = create_output_table(prev_valid_options, [])
    logger.info(f'Previous Update...')
    logger.info(prev_output_table)

  logger.info(f'Final list of options...')
  logger.info(curr_output_table)


def get_updated_info_from_prev_options_run(curr_eligible_options,
                                           prev_valid_options):
  if not prev_valid_options:
    return

  for index, option in enumerate(prev_valid_options):
    list_curr_option = [
        o for o in curr_eligible_options
        if o.Ticker == option.Ticker and o.Strike == option.Strike and
        o.IsCallCE_Or_PE == option.IsCallCE_Or_PE and o.Expiry == option.Expiry
    ] if curr_eligible_options else []
    curr_option = None
    if len(list_curr_option) == 0:
      logger.info(f'Prev: Missing prev info for {option}')
    elif len(list_curr_option) == 1:
      curr_option = list_curr_option[0]
      logger.info(f'Prev: Found prev info for {option}')
    elif len(list_curr_option) > 1:
      logger.error(
          f'Prev: Expected only one option. found multiple. {list_curr_option}'
      )
    if curr_option:
      option.PercentChange = curr_option.PercentChange
      option.BidPrice = curr_option.BidPrice
      option.AskPrice = curr_option.AskPrice
      option.FilledPrice = curr_option.FilledPrice
      option.PrevClose = curr_option.PrevClose
      option.Volume = curr_option.Volume


def calculate_indicators(option):
  close_series_px = option.HistoricData['Close']
  logger.info(f'options = {option};')
  option.HistoricData["SMA"] = ta.sma(option.HistoricData["Close"], 5)
  option.HistoricData["RSI"] = ta.rsi(option.HistoricData["Close"], 5)
  option.HistoricData["VWAP"] = ta.vwap(low=option.HistoricData["Low"],
                                        high=option.HistoricData["High"],
                                        close=option.HistoricData["Close"],
                                        volume=option.HistoricData["Volume"])
  bband_data = ta.bbands(option.HistoricData["Close"], 5, 2)
  option.HistoricData["BBU"] = bband_data["BBU_5_2.0"]
  option.HistoricData["BBL"] = bband_data["BBL_5_2.0"]
  option.BBandHigh = round(option.HistoricData["BBU"].iloc[-1], ROUNDING)
  option.BBandLow = round(option.HistoricData["BBL"].iloc[-1], ROUNDING)
  option.RSI = round(option.HistoricData["RSI"].iloc[-1], ROUNDING)
  option.SMA = round(option.HistoricData["SMA"].iloc[-1], ROUNDING)
  option.PrevClose = round(option.HistoricData["PREV_CLS"].iloc[-1], ROUNDING)
  option.VWAP = round(option.HistoricData["VWAP"].iloc[-1], ROUNDING)
  option.Volume = round(option.HistoricData["Volume"].sum(), ROUNDING)
  option.CurrentSpread = round(
      max(option.HistoricData["Close"].iloc[-5:]) -
      min(option.HistoricData["Close"].iloc[-5:]), ROUNDING)
  get_ideal_buy_sell_magic(option)


def update_option_scores(option):
  # Your scoring calculations here
  if 0 < option.RSI <= 40:
    option.BuyScore += 1

  if option.Volume >= 100:
    option.BuyScore += 1

  if option.HistoricData["Close"].iloc[-1] <= option.BBandLow:
    option.BuyScore += 1

  if option.SMA <= option.VWAP:
    option.BuyScore += 1

  if option.CurrentSpread >= 0.05:
    option.BuyScore += 1

  if option.FilledPrice == option.BidPrice:
    option.BuyScore += 1

  if option.IV <= 40:
    option.BuyScore += 1

  if option.PercentChange <= 0:
    option.BuyScore += 1


def create_output_table(valid_options, prev_valid_options):

  outputTable = PrettyTable()
  outputTable.field_names = [
      "Sr. no.", "Ticker", "Strike", "Type", "Expiry", "Bid", "Filled",
      "PrvCls", "Ask", "Ideal (Buy/Sell)", "Spread", "Vol / OI", "BB (S/R)",
      "RSI", "VWAP", "SMA(5)", "Today Gain", "IV", "B-Score", "Prev B-Score"
  ]

  outputTable.align["Sr. no."] = "c"
  outputTable.align["Ticker"] = "c"
  outputTable.align["Strike"] = "r"
  outputTable.align['IsCall'] = 'c'
  outputTable.align["Expiry"] = "c"
  outputTable.align["Bid"] = "r"
  outputTable.align["Filled"] = "r"
  outputTable.align["PrvCls"] = "r"
  outputTable.align["Ask"] = "r"
  outputTable.align["Ideal (Buy/Sell)"] = "r"
  outputTable.align["Spread"] = "r"
  outputTable.align["Vol / OI"] = "r"
  outputTable.align["BB (S/R)"] = "r"
  outputTable.align["RSI"] = "r"
  outputTable.align["VWAP"] = "r"
  outputTable.align["SMA(5)"] = "r"
  outputTable.align["Today Gain"] = "r"
  outputTable.align["IV"] = "r"
  outputTable.align["B-Score"] = "r"
  outputTable.align["Prev B-Score"] = "r"

  # Sort this shit somehow

  for index, option in enumerate(valid_options):

    # find the prev option by comparing option and checking within prev_valid_options
    list_prev_options = [
        o for o in prev_valid_options
        if o.Ticker == option.Ticker and o.Strike == option.Strike and
        o.IsCallCE_Or_PE == option.IsCallCE_Or_PE and o.Expiry == option.Expiry
    ] if prev_valid_options else []
    prev_option = None
    if len(list_prev_options) == 0:
      logger.info(f'Now: Missing prev info for {option}')
    elif len(list_prev_options) == 1:
      prev_option = list_prev_options[0]
      logger.info(f'Now: Found prev info for {option}')
    elif len(list_prev_options) > 1:
      logger.error(
          f'Now: Expected only one option. found multiple. {list_prev_options}'
      )

    outputTable.add_row([
        index, option.Ticker, option.Strike, option.IsCallCE_Or_PE,
        option.Expiry, '{:.2f}'.format(option.BidPrice),
        '{:.2f}'.format(option.FilledPrice), '{:.2f}'.format(option.PrevClose),
        '{:.2f}'.format(option.AskPrice), '{:.2f}'.format(option.IdealBuy) +
        " / " + '{:6.2f}'.format(option.IdealSell),
        '{:.2f}'.format(round(option.CurrentSpread,
                              2)), '{:.0f}'.format(option.Volume) + " / " +
        '{:4.0f}'.format(option.OpenInterest),
        '{:.2f}'.format(option.BBandLow) + " / " +
        '{:6.2f}'.format(option.BBandHigh), '{:.2f}'.format(option.RSI),
        '{:.2f}'.format(option.VWAP), '{:.2f}'.format(option.SMA),
        '{:.2f}'.format(option.PercentChange), option.IV,
        "#" * option.BuyScore + " " + str(option.BuyScore) + " / 8",
        "TBC" if not prev_option else "#" * prev_option.BuyScore + " " +
        str(prev_option.BuyScore) + " / 8"
    ])

  return outputTable
  # Populate your table with data from valid_options
  # using a for loop or list comprehension


def get_latest_file_in_folder(folder_path):
  files = [
      os.path.join(SAVE_FOLDER, f) for f in os.listdir(folder_path)
      if f.endswith('.pkl')
  ]
  if len(files):
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def load_latest_options_holders():

  # Find the latest file in the folder
  latest_file = get_latest_file_in_folder(SAVE_FOLDER)
  logger.info(f'latest dumped filed: {latest_file}')

  if not latest_file:
    return
  # Load the latest file
  with open(latest_file, 'rb') as f:
    loaded_option_holders = pickle.load(f)
    return loaded_option_holders


def save_latest_option_holders(option_holders):
  for options in option_holders:
    options.historicData = None

  # Get the current timestamp
  timestamp = datetime.now().strftime('%Y%m%d_%H%M')

  # Create the directory if it doesn't exist
  os.makedirs(SAVE_FOLDER, exist_ok=True)

  # Save the list to a file using pickle with the timestamp in the filename
  save_filename = os.path.join(SAVE_FOLDER, f'{timestamp}.pkl')
  with open(save_filename, 'wb') as f:
    pickle.dump(option_holders, f)


if __name__ == '__main__':

  args = get_args()
  main(args)
