import os
import logging
import pickle
import sys

from trading.common import utils

logger = utils.get_logger("tasty_liquid_tickers", True)
from devtools import pprint, PrettyFormat

from tastytrade import Session, Account
from tastytrade.metrics import get_market_metrics
from tastytrade.instruments import get_option_chain, get_future_option_chain, OptionType
from tastytrade.utils import get_tasty_monthly
from tastytrade.order import InstrumentType
import pandas as pd
import yaml

from decimal import Decimal
from tabulate import tabulate
from datetime import date

from devtools import pprint
from tastytrade.streamer import EventType
import yfinance as yf

from trading.common import utils
from trading.brokers import tasty as tt
from trading.common import util_files

FILE_FMT_LIQUID_TICKERS = 'data/tasty/liquid_tickers_with_high_iv'
FILE_FMT_LIQUID_TICKERS_LIST = 'data/tasty/liquid_tickers_list'
MAX_PRICE_THRESHOLD = 300
MIN_PRICE_THRESHOLD = 5


def _rename_key(data, old_key, new_key):
  if old_key in data:
    data[new_key] = data.pop(old_key)
  return data


def _get_watchlist_as_df(tasty_ins,
                         watchlists_to_filter=[],
                         pvt_watchlists=[]):
  wl_lists = []

  symbols_universe = set()

  def get_symbols(wl_filter, url):
    sym_list = set()
    for wl_list in [
        tasty_ins.session.get(f'/{url}/{name}',
                              params={'counts-only':
                                      False})['watchlist-entries']
        for name in wl_filter
    ]:
      symbols = [x['symbol'] for x in wl_list]
      sym_list.update(symbols)
      logger.info(f'Symbols = {symbols}')
    return sym_list

  public_symbols = get_symbols(watchlists_to_filter, 'public-watchlists')
  logger.info(f'public_symbols = {public_symbols}')
  symbols_universe.update(public_symbols)

  pvt_symbols = get_symbols(pvt_watchlists, 'watchlists')
  logger.info(f'pvt_symbols = {pvt_symbols}')
  symbols_universe.update(pvt_symbols)
  symbols_universe = list(symbols_universe)
  logger.info(
      f'symbols_universe = {symbols_universe}; len={len(symbols_universe)}')
  return symbols_universe


def _log_metrics(metrics, label):
  logger.info(f'{label} metrics: {metrics}')
  for metric in metrics:
    logger.info(f'{label}:: metric: {str(metric)}')


def _find_liquid_tickers_with_high_iv(metrics_list,
                                      iv_threshold=0.5,
                                      liquid_threshold=4):

  # get all metrics with high IV and liquidity
  metrics = [
      metric for metric in metrics_list
      if metric.tw_implied_volatility_index_rank and metric.liquidity_rating
      and float(metric.tw_implied_volatility_index_rank) > iv_threshold
      and float(metric.liquidity_rating) >= liquid_threshold
  ]


def print_liquid_tickers_with_high_iv(metrics, label):
  """Print liquid tickers with high IV using tabulate."""
  # use tabulate to print symbol, tw_implied_volatility_index_rank, implied_volatility_percentile
  table_data = []
  for metric in metrics:
    table_data.append([
        metric.symbol, metric.tw_implied_volatility_index_rank,
        metric.implied_volatility_percentile
    ])
  table = tabulate(table_data,
                   headers=[
                       "Symbol", "tw_implied_volatility_index_rank",
                       "implied_volatility_percentile"
                   ],
                   tablefmt="plain")

  logger.info(f'[[{label}]] Liquid tickers with high IV:\n{table}')


def _find_liquid_tickers_with_high_iv(metrics_list,
                                      iv_threshold=0.5,
                                      liquid_threshold=4,
                                      label=''):

  # get all metrics with high IV and liquidity
  metrics = [
      metric for metric in metrics_list
      if metric.tw_implied_volatility_index_rank and metric.liquidity_rating
      and float(metric.tw_implied_volatility_index_rank) > iv_threshold
      and float(metric.liquidity_rating) >= liquid_threshold
  ]

  debug_str = f'{label} IV:{iv_threshold}; Liq:{liquid_threshold}'
  print_liquid_tickers_with_high_iv(metrics, debug_str)

  return metrics


def save_metrics_to_csv(metrics):
  filename = f'{FILE_FMT_LIQUID_TICKERS}_{date.today().strftime("%Y_%m_%d")}.csv'

  metrics_list = []
  for metric in metrics:
    metrics_list.append({
        'symbol':
        metric.symbol,
        'tw_implied_volatility_index_rank':
        float(metric.tw_implied_volatility_index_rank),
        'implied_volatility_percentile':
        float(metric.implied_volatility_percentile)
    })

  # Convert the list of dictionaries to a DataFrame
  df = pd.DataFrame(metrics_list)

  # Save the DataFrame to a CSV file
  df.to_csv(filename, index=False)


def read_yaml_file(filename):
  with open(filename, 'r') as file:
    data = yaml.safe_load(file)
  return data


def fetch_latest_market_prices(symbols):
  tickers = yf.Tickers(symbols)
  market_data = dict()
  for ticker in symbols:
    ticker_history = tickers.tickers[ticker].history(period="1d")
    if not ticker_history.empty:
      market_data[ticker] = ticker_history['Close'][0]
  return market_data


def get_unique_symbols_with_price_within_thresholds(yaml_data, market_data,
                                                    max_px, min_px):
  unique_symbols = set()

  for item in yaml_data:
    symbol = item['symbol']
    latest_price = market_data.get(symbol, None)
    if latest_price is not None and min_px <= latest_price <= max_px:
      unique_symbols.add(symbol)

  return unique_symbols


def print_symbols(symbols):
  for symbol in symbols:
    logger.info(f'Symbol: {symbol}')
  all_syms = ','.join(symbols)
  logger.info(f'all_syms = {all_syms}')


def test_yaml():

  FILE_FMT_LIQUID_TICKERS = 'liquid_tickers_with_high_iv'

  latest_file = util_files.get_latest_file_in_folder_by_pattern(
      'data', FILE_FMT_LIQUID_TICKERS)
  # Example usage
  filename = latest_file

  yaml_data = read_yaml_file(filename)
  symbols_list = [item['symbol'] for item in yaml_data]
  market_data = fetch_latest_market_prices(symbols_list)
  symbols = get_unique_symbols_with_price_within_thresholds(
      yaml_data, market_data, MAX_PRICE_THRESHOLD, MIN_PRICE_THRESHOLD)
  print_symbols(symbols)


def _exclude_tickers_with_price_outside_threshold(symbols, threshold_high,
                                                  threshold_low):
  market_data = fetch_latest_market_prices(symbols)
  filtered_symbols_list = list()
  for symbol in symbols:
    latest_price = market_data.get(symbol, None)
    if not latest_price:
      filtered_symbols_list.append(symbol)
    else:
      logger.info(f'Ticker For Price {symbol} = {latest_price}')
      if latest_price >= threshold_high or latest_price <= threshold_low:
        logger.info(
            f'Ticker outside threshold {symbol} = {latest_price}; '
            f'threshold_high = {threshold_high}; threshold_low = {threshold_low}'
        )
      else:
        filtered_symbols_list.append(symbol)

  # symbols = get_unique_symbols_with_price_within_thresholds(
  #     yaml_data, market_data, 200, 5)
  # logger.info(f'FIltered symbols = {symbols}')
  return filtered_symbols_list


def get_tickers_from_watchlists(tasty_ins):

  # These tickers are not very important, let's pick out only the ones with high IV
  reg_wl_tickers = _get_watchlist_as_df(tasty_ins,
                                        watchlists_to_filter=[
                                            'High Options Volume', 'S&P 500',
                                            'Russell 1000', 'tasty IVR'
                                        ],
                                        pvt_watchlists=['Extra'])
  reg_wl_tickers = _exclude_tickers_with_price_outside_threshold(
      reg_wl_tickers, MAX_PRICE_THRESHOLD, MIN_PRICE_THRESHOLD)
  logger.info(f'reg_wl_tickers len = {len(reg_wl_tickers)}; '
              f'reg_wl_tickers={reg_wl_tickers}')

  # These tickers are very important and I definitely want all numbers
  # There's no need to filter them
  my_etf_tickers = _get_watchlist_as_df(tasty_ins,
                                        watchlists_to_filter=[],
                                        pvt_watchlists=[
                                            'Defaults',
                                            'Futures',
                                        ])
  logger.info(f'my_etf_tickers len = {len(my_etf_tickers)}; '
              f'my_etf_tickers = {my_etf_tickers}')

  all_tickers = reg_wl_tickers + my_etf_tickers

  # break tickers_list into calls of 200, each time getting metrics
  all_tickers = [
      all_tickers[i:i + 200] for i in range(0, len(all_tickers), 200)
  ]
  return all_tickers, reg_wl_tickers, my_etf_tickers


def get_liq_tickers_from_tickers(tasty_ins, tickers, iv_threshold,
                                 liquid_threshold, label):
  tickers_metrics = []

  # break tickers into a subgroup of 100 items
  tickers = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
  for lst in tickers:
    tickers_metrics.extend(get_market_metrics(tasty_ins.session, lst))

  _log_metrics(tickers_metrics, label)

  liquid_metrics = _find_liquid_tickers_with_high_iv(
      tickers_metrics,
      iv_threshold=iv_threshold,
      liquid_threshold=liquid_threshold,
      label=label)

  return liquid_metrics


def get_iv_report():
  """
  Retrieves the IV report for potential liquid tickers with high IV.

  This function retrieves the IV report for potential liquid tickers with high IV
  by performing the following steps:

  1. Initializes a `TastyBite` object named `tasty_ins`.
  2. Calls the `_get_watchlist_as_df` function to obtain a DataFrame named `uniq_df`
     containing unique watchlist entries.
  3. Extracts the list of tickers from the `symbol` column of `uniq_df` and assigns
     it to the variable `tickers_list`.
  4. Calls the `get_market_metrics` function with `tasty_ins.session` and
     `tickers_list` as arguments to obtain a list of market metrics named
     `tickers_metrics`.
  5. Calls the `_log_metrics` function with `tickers_metrics` and the label
     `'Watchlist'` as arguments.
  6. Calls the `_find_liquid_tickers_with_high_iv` function with `tickers_metrics`
     as an argument to obtain a list of potential liquid tickers named
     `potential_liquid_tickers` and a list of liquid metrics named `liquid_metrics`.
  7. Logs the `potential_liquid_tickers` and `tickers_list` using the `logger.info`
     function.
  8. Returns the `potential_liquid_tickers` and `liquid_metrics`.

  Returns:
      Tuple[List[str], List[MarketMetric]]: A tuple containing the list of potential
      liquid tickers and the list of liquid metrics.
  """

  tasty_ins = tt.TastyBite()
  all_tickers, reg_wl_tickers, my_etf_tickers = get_tickers_from_watchlists(
      tasty_ins)

  all_metrics = get_liq_tickers_from_tickers(tasty_ins, reg_wl_tickers, 0.5, 4,
                                             'Def Watchlists')
  all_metrics += get_liq_tickers_from_tickers(tasty_ins, my_etf_tickers, 0.2,
                                              1, 'ETF Watchlists')

  # save_metrics_to_yaml(liquid_metrics)
  save_metrics_to_csv(all_metrics)


def main(post_to_tg: bool = True):
  get_iv_report()
  # test_yaml()


if __name__ == '__main__':
  main()
'''
watchlist names

52 Week Near High
52 Week Near Low
A.I. Stocks
All Earnings
BAT's Watchlist
Basic Materials
CRE Hospitality Price Return Index
CRE Office Price Return Index
CRE Residential Price Return Index
CRE Retail Price Return Index
Communication Services
Consumer Defensive
Consumer Discretionary
Crypto
Crypto ETFs
Dividend Aristocrats
Dividend Champions
Dow Jones Industrial Average
Energy
Financial Services
Futures: All
Futures: CME
Futures: Micros
Futures: Small Exchange
Futures: With Options
Healthcare
High Options Volume
ISE Homebuilders Index
Industrials
Liquid ETFs
Liquid Symbols
Market
Market Indicators
NASDAQ 100
NASDAQ Golden Dragon China Index
NASDAQ-100 Target 25 Index
PHLX Gold/Silver Sector
PHLX Housing Sector
PHLX Oil Service Sector
PHLX Semiconductor
PHLX Utility Sector
Real Estate
Russell 1000
Russell Microcap
Russell Midcap
S&P 100
S&P 500
Technology
Tom's Watchlist
Utilities
Volatility Indexes
tasty Earnings
tasty Fast Movers
tasty IVR
tasty default


========= Watchlist with good tickers ========

High Options Volume
Liquid ETFs
S&P 500
Technology
NASDAQ 100
tasty IVR
tasty default
Russell 1000
'''
