import os
import sys
from datetime import date
import yaml
from datetime import datetime, timedelta

from trading.common import utils

now_s = datetime.now().strftime("yf_option_ic__%Y-%m-%d__%H-%M-%S")

logger = utils.get_logger(now_s)

from collections import defaultdict
import pickle
from dateutil.parser import *
import yfinance as yf
from finviz.screener import Screener
import pandas as pd
import argparse

from scipy.optimize import brentq
import pandas_ta as ta
from prettytable import PrettyTable
import QuantLib as ql
import numpy as np
from tabulate import tabulate
import scipy.stats as stats
from math import exp, sqrt
from tastytrade.utils import get_tasty_monthly

from trading.options.option_utils import OptionHolder
from trading.common import util_prints
from trading.brokers import tasty as tt
from trading.common import util_files
from trading.services import telegram_runner
'''
Tihs is meant to calculate the delta, and iron condor pricing for me.
But yfinance doesnt give good data on options, as mucha s tastylive for e.g.
'''

FILE_FMT_LIQUID_TICKERS = 'liquid_tickers_with_high_iv'
BPE_THRRESHOLD = 350
RISK_REWARD_THRESHOLD = 0.33


def get_stock_list():
  # read the latest liquid tickers from file FILE_FMT_LIQUID_TICKERS
  # get the latest file: FILE_FMT_LIQUID_TICKERS

  latest_file = util_files.get_latest_file_in_folder_by_pattern(
      'data/tasty', FILE_FMT_LIQUID_TICKERS)
  logger.info(f'latest dumped filed: {latest_file}')

  # read yaml from latest file
  # get the list of stocks
  # list of tickers

  # latest_file = get_latest_file_in_folder('data')
  # logger.info(f'latest dumped filed: {latest_file}')
  if not latest_file:
    logger.info(f'file not found. returning')
    sys.exit(1)

  df = pd.read_csv(latest_file)
  # convert this into a list of dictionaries, per line
  df_dict = df.to_dict('records')
  logger.info(f'df_dict = {df_dict}')
  return df_dict

  # return [
  #     'DKNG', 'JBLU', 'MSFT', 'O', 'OPEN', 'PTON', 'TSLA', 'MU', 'PLTR', 'SBUX',
  #     'AES', 'AGL', 'CSCO', 'HUN', 'LCID', 'STWD', 'WBA', 'AMD', 'APH', 'BMY',
  #     'CLF', 'DAL', 'MP', 'RIVN', 'AAPL', 'AMZN', 'HPE', 'INTC', 'NVDA', 'SOFI',
  #     'U', 'WBD', 'LUV', 'LYFT', 'MRVL', 'NWL', 'PYPL', 'UBER', 'ZI', 'DIS',
  #     'F', 'QS', 'VZ', 'DVN', 'HOOD', 'KMI', 'PFE', 'PLUG', 'BABA', 'BSX',
  #     'GOLD', 'IWM', 'KDP', 'MARA', 'OLPX', 'QQQ', 'RIOT', 'SQQQ', 'TLRY',
  #     'WMT', 'XOM', 'GDX', 'GLD', 'HYG', 'KRE', 'MO', 'NIO', 'SLV', 'SNAP',
  #     'SPY', 'TQQQ', 'TSM', 'XLF', 'BOIL', 'BYND', 'ET', 'GRTS', 'JD', 'NU',
  #     'NXE', 'PBR', 'RIG', 'SAVE', 'SHOP', 'SNDL', 'XLU', 'XPEV', 'XRX'
  # ]
  # QQQ, DIG, EWJ, EFA, EWZ, USO, GLD, GDX, XLE,
  # XLF, XRT, IWM, EEM, DIA, SPY, SLV, UNG.
  # return ['XLK']


# # Black-Scholes pricing model for European call option
# def black_scholes_call_price(S, K, T, r, sigma):
#   d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
#   d2 = d1 - sigma * np.sqrt(T)
#   call_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
#   return call_price

# # Function to find the implied volatility
# def get_implied_volatility(S, K, T, r, market_price):

#   def objective_function(sigma):
#     return black_scholes_call_price(S, K, T, r, sigma) - market_price

#   a, b = 1e-6, 10
#   if objective_function(a) * objective_function(b) > 0:
#     raise ValueError(
#         "f(a) and f(b) must have different signs. Try expanding the interval.")

#   implied_vol = brentq(objective_function, a, b)
#   return implied_vol


# Function to calculate greeks
def calculate_greeks(stock_price,
                     strike_price,
                     maturity_date,
                     risk_free_rate,
                     option_type,
                     implied_volatility,
                     market_price=None):

  def d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

  def d2(S, K, T, r, sigma):
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

  def delta(S, K, T, r, sigma, option_type):
    D1 = d1(S, K, T, r, sigma)
    if option_type == 'call':
      return stats.norm.cdf(D1)
    elif option_type == 'put':
      return stats.norm.cdf(D1) - 1

  def gamma(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return stats.norm.pdf(D1) / (S * sigma * np.sqrt(T))

  def vega(S, K, T, r, sigma):
    D1 = d1(S, K, T, r, sigma)
    return S * stats.norm.pdf(D1) * np.sqrt(T)

  def theta(S, K, T, r, sigma, option_type):
    D1 = d1(S, K, T, r, sigma)
    D2 = d2(S, K, T, r, sigma)
    if option_type == 'call':
      return (-S * stats.norm.pdf(D1) * sigma / (2 * np.sqrt(T)) -
              r * K * exp(-r * T) * stats.norm.cdf(D2))
    elif option_type == 'put':
      return (-S * stats.norm.pdf(D1) * sigma / (2 * np.sqrt(T)) +
              r * K * exp(-r * T) * stats.norm.cdf(-D2))

  def rho(S, K, T, r, sigma, option_type):
    D2 = d2(S, K, T, r, sigma)
    if option_type == 'call':
      return K * T * exp(-r * T) * stats.norm.cdf(D2)
    elif option_type == 'put':
      return -K * T * exp(-r * T) * stats.norm.cdf(-D2)

  # Convert maturity date to time to maturity in years
  today = date.today()
  T = (maturity_date.date() - today).days / 365.25

  if market_price is None:
    raise ValueError(
        "Market price is required to calculate implied volatility.")

  greeks = {
      'delta':
      delta(stock_price, strike_price, T, risk_free_rate, implied_volatility,
            option_type),
      'gamma':
      gamma(stock_price, strike_price, T, risk_free_rate, implied_volatility),
      'vega':
      vega(stock_price, strike_price, T, risk_free_rate, implied_volatility),
      'theta':
      theta(stock_price, strike_price, T, risk_free_rate, implied_volatility,
            option_type),
      'rho':
      rho(stock_price, strike_price, T, risk_free_rate, implied_volatility,
          option_type)
  }

  logger.info(f'stock_price={stock_price}, strike_price={strike_price}'
              f' maturity_date={maturity_date},'
              f'option_type={option_type};'
              f'greeks: {greeks}')

  return greeks['delta'], greeks['gamma'], greeks['theta']


def get_option_data_for_stock(risk_free_rate, stock=""):
  logger.info(f'stock ={stock}')
  stock_ticker = yf.Ticker(stock)
  option_holder_return = []
  logger.info(f'risk_free_rate = {risk_free_rate}')
  tasty_mnth_exp = get_tasty_monthly()

  matching_expiries = lambda exp: exp.date() == tasty_mnth_exp

  for x in stock_ticker.options:
    expiration_time = parse(x)
    is_monthly_expiry = matching_expiries(expiration_time)
    logger.info(f'[{stock}]:: Expiration time = {expiration_time};  '
                f'tasty_mnth_exp={tasty_mnth_exp};  '
                f'matched_expiry={is_monthly_expiry}')
    if is_monthly_expiry:
      for index, row in stock_ticker.option_chain(x).calls.iterrows():
        convert_yf_opt_row_to_holder(row, stock, expiration_time,
                                     option_holder_return, risk_free_rate,
                                     "call")

      for index, row in stock_ticker.option_chain(x).puts.iterrows():
        convert_yf_opt_row_to_holder(row, stock, expiration_time,
                                     option_holder_return, risk_free_rate,
                                     "put")

  return option_holder_return


def convert_yf_opt_row_to_holder(row, stock, expiration_time,
                                 option_holder_return, risk_free_rate,
                                 option_type):
  historic_data = get_daily_data_for_option(row["contractSymbol"])
  contr_sym = row['contractSymbol']

  opt_holder = OptionHolder()
  opt_holder.Ticker = stock
  opt_holder.IsCall = (option_type == "call")
  opt_holder.Expiry = expiration_time.date()
  opt_holder.FilledPrice = row["lastPrice"]
  opt_holder.BidPrice = row["bid"]
  opt_holder.AskPrice = row["ask"]
  opt_holder.Strike = row["strike"]
  opt_holder.CurrentSpread = opt_holder.AskPrice - opt_holder.BidPrice
  opt_holder.PercentChange = round(row["change"], 3)
  opt_holder.OpenInterest = row["openInterest"]
  opt_holder.IV = round(row["impliedVolatility"], 3)
  opt_holder.YahooString = row["contractSymbol"]
  opt_holder.HistoricData = historic_data

  stock_price = yf.Ticker(stock).history(period='1d')['Close'].iloc[-1]
  logger.info(
      f'[{stock}]::  {option_type.capitalize()} contractSymbol = {contr_sym}; '
      f'Expiry: {expiration_time}; Px: {stock_price}; IV={opt_holder.IV}; ')

  opt_holder.Delta, opt_holder.Gamma, opt_holder.Theta = calculate_greeks(
      stock_price, opt_holder.Strike, expiration_time, risk_free_rate,
      option_type, opt_holder.IV,
      (opt_holder.AskPrice + opt_holder.BidPrice) / 2)

  option_holder_return.append(opt_holder)


def get_daily_data_for_option(option=""):
  return []
  stockTicker = yf.Ticker(option)

  hist_data = stockTicker.history(period="1mo", interval="1d")
  return hist_data


def get_eligible_options(eligibleOptions_file, use_pickle=False):
  # if file exists, then load, else create
  liquid_ticker_metrics = get_stock_list()
  if os.path.exists(eligibleOptions_file) and use_pickle:

    with open(eligibleOptions_file, 'rb') as f:
      logger.info(f'Reading from file: {eligibleOptions_file}')
      return pickle.load(f), liquid_ticker_metrics
  else:
    logger.info(
        f'Either file is unavailable or use_pickle is False: {use_pickle}')

  eligibleOptions = defaultdict(list)
  # tasty_bite = tt.TastyBite()
  risk_free_rate = 0.055
  stockslist = [x['symbol'] for x in liquid_ticker_metrics]

  for x in stockslist:
    logger.info("Checking if stock is eligible: " + x)
    eligibleOptions[x] = get_option_data_for_stock(risk_free_rate, x)

  with open(eligibleOptions_file, 'wb') as f:
    pickle.dump(eligibleOptions, f)

  return eligibleOptions, liquid_ticker_metrics


def print_table(eligibleOptions):

  outputTable = PrettyTable()

  outputTable.field_names = [
      "Sr. no.",
      "Ticker",
      "Strike",
      "Expiry",
      "Bid",
      "Filled",
      "Ask",
      # "Ideal (Buy/Sell)",
      # "Spread",
      "Vol / OI",
      # "BB (S/R)",
      "Δ",
      "𝛩",
      "ɣ",
      # "Today Gain",
      "IV",
      # "B-Score"
  ]

  outputTable.align["Sr. no."] = "c"
  outputTable.align["Ticker"] = "c"
  outputTable.align["Strike"] = "r"
  outputTable.align["Expiry"] = "c"
  outputTable.align["Bid"] = "r"
  outputTable.align["Filled"] = "r"
  outputTable.align["Ask"] = "r"
  # outputTable.align["Ideal (Buy/Sell)"] = "c"
  # outputTable.align["Spread"] = "r"
  outputTable.align["Vol / OI"] = "c"
  # outputTable.align["BB (S/R)"] = "c"
  outputTable.align["Δ"] = "r"
  outputTable.align["𝛩"] = "r"
  outputTable.align["ɣ"] = "r"
  # outputTable.align["Today Gain"] = "r"
  outputTable.align["IV"] = "r"
  outputTable.align["B-Score"] = "c"

  # Sort this shit somehow
  eligibleOptions.sort(key=lambda x: x.BuyScore, reverse=True)

  for index, option in enumerate(eligibleOptions):
    outputTable.add_row([
        index,
        option.Ticker,
        f'{option.Strike}{" C" if option.IsCall else " P"}',
        option.Expiry,
        '{:.3f}'.format(option.BidPrice),
        '{:.3f}'.format(option.FilledPrice),
        '{:.3f}'.format(option.AskPrice),
        # '{:.3f}'.format(option.IdealBuy) +" / " + '{:.3f}'.format(option.IdealSell),
        # '{:.3f}'.format(round(option.CurrentSpread, 3)),
        str(option.Volume) + " / " + str(option.OpenInterest),
        # '{:.3f}'.format(option.BBandLow) + " / " + '{:.3f}'.format(option.BBandHigh),
        '{:.2f}'.format(option.Delta),
        '{:.2f}'.format(option.Theta),
        '{:.2f}'.format(option.Gamma),
        # '{:.3f}'.format(option.PercentChange),
        option.IV,
        # str(option.BuyScore) + " / 8"
    ])
  logger.info(outputTable)


def filter_options(option_holders, is_call, max_delta=0.50):
  return [
      opt for opt in option_holders
      if opt.IsCall == is_call and abs(opt.Delta) <= max_delta
  ]


def find_option_by_delta(options, target_delta):
  return min(options, key=lambda x: abs(x.Delta - target_delta))


def log_options(ticker, calls, puts):
  logger.info(f'[{ticker}]:: {len(calls)} calls and {len(puts)} puts')
  for call in calls:
    logger.info(f'[{ticker}]:: call = {call.simple_str()}')
  for put in puts:
    logger.info(f'[{ticker}]:: put = {put.simple_str()}')


def create_option_table(puts, calls):
  headers = [
      "Ticker", "Strike", "Expiry", "OptionType", "FilledPrice", "Bid/Ask",
      "IV", "Delta", "Theta", "Gamma"
  ]
  table = [opt.to_list() for opt in puts] + [opt.to_list() for opt in calls]
  return tabulate(table,
                  headers=headers,
                  tablefmt="plain",
                  colalign=("left", "right", "right", "center", "right",
                            "center", "left", "left", "left", "left"))


def find_buy_options(options, sell_index):
  buy_index = sell_index + 1 if sell_index < len(options) - 1 else None
  return options[buy_index] if buy_index is not None else None


def log_iron_condor_options(ticker, put_to_buy, put_to_sell, call_to_sell,
                            call_to_buy):
  logger.info(f'[{ticker}]:: Upper Put  = {put_to_buy.simple_str()}')
  logger.info(f'[{ticker}]:: Lower Put  = {put_to_sell.simple_str()}')
  logger.info(f'[{ticker}]:: Upper Call = {call_to_sell.simple_str()}')
  logger.info(f'[{ticker}]:: Lower Call = {call_to_buy.simple_str()}')

  opt_str = lambda optn, is_bid: f'{optn.Strike} {"C" if optn.IsCall else "P"} ->  ${-optn.BidPrice if is_bid else optn.AskPrice}'
  logger.info(f'[{ticker}]:: Upper Put  = {opt_str(put_to_buy, False)}')
  logger.info(f'[{ticker}]:: Lower Put  = {opt_str(put_to_sell, True)}')
  logger.info(f'[{ticker}]:: Upper Call = {opt_str(call_to_sell, True)}')
  logger.info(f'[{ticker}]:: Lower Call = {opt_str(call_to_buy, False)}')


def calculate_expected_credit(put_to_buy, put_to_sell, call_to_sell,
                              call_to_buy):
  return (put_to_buy.AskPrice + put_to_buy.BidPrice + call_to_buy.AskPrice +
          call_to_buy.BidPrice -
          (put_to_sell.AskPrice + put_to_sell.BidPrice +
           call_to_sell.AskPrice + call_to_sell.BidPrice)) / 2


def get_calls_and_puts_for_ic(option_holders, ticker, delta_for_search):
  no_results = [None, None, None, None]
  calls = filter_options(option_holders, is_call=True)
  puts = filter_options(option_holders, is_call=False)

  if not calls or not puts:
    return no_results

  calls = sorted(calls, key=lambda x: x.Strike)
  puts = sorted(puts, key=lambda x: x.Strike)

  log_options(ticker, calls, puts)
  logger.info('\n' + create_option_table(puts, calls))

  call_to_sell = find_option_by_delta(calls, delta_for_search)
  put_to_sell = find_option_by_delta(puts, -delta_for_search)

  if call_to_sell is None or put_to_sell is None:
    logger.warning('Missing call or put with delta of 50 for IC')
    return no_results

  call_to_buy = find_buy_options(calls, calls.index(call_to_sell))
  put_to_buy = find_buy_options(puts[::-1],
                                len(puts) - 1 - puts.index(put_to_sell))

  if call_to_buy is None or put_to_buy is None:
    logger.info(
        f'[{ticker}]:: Missing call or put with delta of 50 for IC for other side'
    )
    return no_results

  log_iron_condor_options(ticker, put_to_buy, put_to_sell, call_to_sell,
                          call_to_buy)

  return call_to_sell, call_to_buy, put_to_sell, put_to_buy


def get_iron_condor_credit(option_holders, ticker, delta_for_search=0.2):
  call_to_sell, call_to_buy, put_to_sell, put_to_buy = get_calls_and_puts_for_ic(
      option_holders, ticker, delta_for_search)

  expected_credit = calculate_expected_credit(put_to_buy, put_to_sell,
                                              call_to_sell, call_to_buy)
  logger.info(f'[{ticker}]:: expected_credit = {expected_credit}')

  max_loss, bpr = calc_buying_power(call_to_sell, call_to_buy, put_to_buy,
                                    put_to_sell, expected_credit)

  ic_results = {
      "ticker": call_to_sell.Ticker,
      "expiration_date": call_to_sell.Expiry,
      "call_30_delta_upper": call_to_sell,
      "call_30_delta_lower": call_to_buy,
      "put_30_delta_upper": put_to_buy,
      "put_30_delta_lower": put_to_sell,
      "credit": -1 * expected_credit,
      "max_loss": max_loss,
      "buying_power_reduction": bpr,
  }
  logger.info(f'[{ticker}]:: ic_results = {ic_results}')
  return ic_results


def calc_buying_power(call_30_delta_upper, call_30_delta_lower,
                      put_30_delta_upper, put_30_delta_lower, credit):

  # Calculate the buying power reduction
  call_spread_width = call_30_delta_lower.Strike - call_30_delta_upper.Strike
  put_spread_width = put_30_delta_lower.Strike - put_30_delta_upper.Strike
  max_spread_width = max(call_spread_width, put_spread_width)
  max_loss = max_spread_width + credit
  buying_power_reduction = max_loss * 100  # Assuming 100 shares per option contract

  logger.info(f'max_spread_width={max_spread_width}; '
              f'call_spread_width={call_spread_width}; '
              f'put_spread_width={put_spread_width}; '
              f'max_loss = {max_loss} and bpr = {buying_power_reduction}')
  return max_loss, buying_power_reduction


def generate_iron_condor_stats(eligibleOptions):
  ticker_iron_condor_info = {}
  for ticker in eligibleOptions:
    ticker_options = eligibleOptions[ticker]
    iron_condor_data = get_iron_condor_credit(ticker_options, ticker)
    if iron_condor_data:
      ticker_iron_condor_info[ticker] = iron_condor_data

      logger.info(
          f'{ticker} -> iron_condor_data = {util_prints.get_pretty_print(iron_condor_data)}'
      )

    if len(ticker_options) == 0:
      logger.info(f"{ticker} -> No data to use")
    else:
      logger.info(f'{ticker} -> ticker_options = {ticker_options}')

    print_table(ticker_options)
  return ticker_iron_condor_info


def format_value(value):
  return "{:.2f}".format(value)


def tabulate_ic_data(iron_condor_data, liquid_ticker_metrics):

  def get_metrics(ticker, metric_field):
    metric_to_find = None
    for x in liquid_ticker_metrics:
      if x['symbol'] == ticker:
        metric_to_find = x[metric_field]
        break
    if metric_to_find:
      return format_value(metric_to_find)
    else:
      logger.info(
          f'[{ticker}]:: {metric_field} not found in liquid_ticker_metrics')
      logger.info(f'liquid_ticker_metrics = {liquid_ticker_metrics}')
      return ticker

  list_of_values = [[
      x['ticker'],
      format_value(x['credit']),
      format_value(x['max_loss']),
      format_value(x['credit'] * 100 / x['buying_power_reduction']),
      format_value(x['buying_power_reduction']),
      get_metrics(x['ticker'], 'implied_volatility_percentile'),
      get_metrics(x['ticker'], 'tw_implied_volatility_index_rank')
  ] for x in iron_condor_data.values()]

  # sort by 4th value
  list_of_values.sort(key=lambda x: x[3], reverse=True)

  def print_table(list_vals, tag, send_telegram=False):
    # Use tabulate to print
    table = tabulate(
        list_vals,
        headers=['Ticker', 'Credit', 'Max Loss', 'RR', 'BPE', 'IVP', 'IVR'],
        floatfmt=".2f",
        colalign=("left", "right", "right", "right", "right", "right",
                  "right"))

    logger.info(f'{tag}: \n {table}')
    if send_telegram:
      _send_msg(f'{tag}: \n ```\n{table}```')

  print_table(list_of_values, 'Full List of tickers', send_telegram=False)

  # BPE should be less < 170 or RR > 1
  filtered_list_vals = [
      x for x in list_of_values
      if float(x[4]) < BPE_THRRESHOLD and float(x[3]) > RISK_REWARD_THRESHOLD
      or float(x[3]) > 1
  ]
  if len(filtered_list_vals):
    print_table(
        filtered_list_vals,
        f'Filtered List of tickers with BPE < {BPE_THRRESHOLD} or RR > 1',
        send_telegram=True)
  else:
    logger.info(f'empty table. ')


def _send_msg(text):
  logger.info(f'Sending msg to telegram: {text}')
  telegram_runner.send_text([text])


if __name__ == '__main__':
  utils.set_pandas_options()

  parser = argparse.ArgumentParser()
  parser.add_argument('--use-pickle', '-up', action='store_true')
  args = parser.parse_args()
  logger.info(f'args = {args}')

  options_list, liquid_ticker_metrics = get_eligible_options(
      'data/eligibleOptions.pkl', use_pickle=args.use_pickle)

  iron_condor_data = generate_iron_condor_stats(options_list)
  tabulate_ic_data(iron_condor_data, liquid_ticker_metrics)
