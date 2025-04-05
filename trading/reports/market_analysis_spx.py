from dhanhq import dhanhq
from enum import Enum
import datetime
from datetime import date
import math
import re
import shutil
import argparse
import warnings

import pandas as pd
import yfinance as yf
import talib
from openpyxl.styles import NamedStyle
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule, FormulaRule

COMPANIES_LIST_FILE = 'trading/resources/all-us-companies.csv'
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

from trading.common import utils

logger = utils.get_logger('market-analysis-spx', use_rich=True)

from trading.common.util_prints import *
from trading.common import telegram_helper
from trading.portfolio.tracker import *

SAVE_FOLDER = 'data/spx'
DRY_RUN_NUM = 100
DRY_RUN_PERIOD = '5y'
PARQUET_FILE = 'data/spx/tickers_data.parquet'
pd.set_option("display.precision", 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

COLUMN_NAMES = [
    "Date/Time",  #0
    "S&P",  #1
    "IJR",  #2
    "Total Universe",  #3
    "4% Up",  # 4
    "4% Down",  #5
    "High Vol",  #6
    "Low Vol",  #7
    "Range <3%",  #8
    "Range 5.01% +",  #9
    "Close Upper Half",  #10
    "Close Lower Half",  #11
    "4.01% Breakouts",  #12
    "Breakout Sustained",  #13
    "Breakout Failure",  #14
    "4.01% Breakdowns",  #15
    "Breakdown Sustained",  #16
    "Breakdown Failure",  #17
    f"15% up in 5 days",  #18
    f"15% down in 5 days",  #19
    f"25% up in 20 days",  #20
    f"25% down in 20 days",  #21
    f"10% above 10 DEMA",  #22
    f"10% below 10 DEMA",  #23
    "Above 10 DEMA",  #24
    "Above 20 DEMA",  #25
    "Above 50 DEMA",  #26
    "Above 200 DEMA",  #27
    "52 Week High",  #28
    "52 Week Low",  #29
    f"15% from 52WH",  #30
    f"30% from 52WH",  #31
    f"50% from 52WH",  #32
    f"70% from 52WH",  #33
    f"70% Plus From 52WH",  #34
    f"15% from 52WL",  #35
    f"30% from 52WL",  #36
    f"50% from 52WL",  #37
    f"90% from 52WL",  #38
    f"150% from 52WL",  #39
    f"150% Plus From 52WL",  #40
    # "Nifty Auto",  #41
    # "Nifty Bank",  #42
    # "Nifty Commodities",  #43
    # "Nifty Consumption",  #44
    # "Nifty CPSE",  #45
    # "Nifty Energy",  #46
    # "Nifty Fin Service",  #47
    # "Nifty FMCG",  #48
    # "Nifty Infra",
    # "Nifty IT",
    # "Nifty Media",
    # "Nifty Metal",
    # "Nifty MNC",
    # "Nifty Pharma",
    # "Nifty PSE",
    # "Nifty PSU Bank",
    # "Nifty Pvt. Bank",
    # "Nifty Realty",
    # "Nifty Service Sector",
    # "5 Day Range",
]
'''
Usage:

python ./trading/stocks/market_analysis.py --date 20231218 -pf -dry

python market_analysis.py --dry_run --read_parquet_file --filename trading/resources/all-us-companies.csv



Analyse:

* 4% ADVANCE: net change >4%
* 4% DECLINE: net change <4%
* NET BREADTH: 4% ADVANCE - 4% DECLINE
* 5 DAY RATIO: Sum of advance of 4% over the last 5 days / Sum of decline of 4% over the last 5 days
* 10 DAY RATIO: Sum of advance of 4% over the last 10 days / Sum of decline of 4% over the last 10 days
* RANGE EXPANSION: Range >= 5%
* RANGE CONTRACTION: Range <= 3%
* 5 DAY RANGE: 5 day high - low / low
* VOLATILITY RATIO: RANGE expansion / Range contraction
* ABOVE AVG VOL: Vol > 1.5 x 20 DMA
* BELOW AVG VO: Vol < 0.5 x 20 DMA
* CLOSE > 50%: Close > 50% of Daily Range on a Range Expansion candle
* CLOSE < 50%: Close < 50% of Daily Range on a Range Expansion candle
* UH/LH RATIO: CLOSE > 50% / CLOSE < 50%
* BREAKOUT: Today's High > 4% from Prev Close
* BREAKDOWN: Today's Low < 4% from Prev Close
* BO/BD RATIO: Breakout / Breakdown
* UP CLOSE %: 4% advance / Breakout
* DOWN CLOSE %: 4% decline / breakdown
* BREAKOUT SUSTAINED: Closes within 40% (of range) from highs on breakout day
* BREAKOUT FAILED: Closes below 40% (of range) from highs on breakout day
* BO S/F RATIO: Breakout sustained / Breakout failed
* BREAKDOWN SUSTAINED: Closes within 40% (of range) from highs on breakdown day
* BREAKDOWN FAILED: Closes below 40% (of range) from highs on breakdown day. BD S/F RATIO: Breakdown sustained / Breakdown failed
. WITHIN 15% FROM 52WH: Close > 15% of 52WH
. WITHIN 15% FROM 52WL: Close < 15% of 52WL
. 15% H/L RATIO: Within 15% from 52 WH / Within 15% from 52 WL
. WITHIN 30% FROM 52WH: Close > 30% of 52WH
. WITHIN 30% FROM 52WL: Close < 30% of 52WL
. 30% H/L RATIO: Within 30% from 52 WH / Within 30% from 52 WL


'''


class HLOCs:
  HIGH = 'High'
  LOW = 'Low'
  CLOSE = 'Close'
  OPEN = 'Open'
  VOLUME = 'Volume'
  ADJ_CLOSE = 'Adj Close'


def process_args():
  parser = argparse.ArgumentParser()

  # create today's today in "%Y%m%d" format
  today_s = date.today().strftime("%Y%m%d")
  logger.info(f'today_s = {today_s}')

  parser.add_argument("--dry-run",
                      '-dry',
                      action='store_true',
                      help="No real buying")

  parser.add_argument("--read-parquet-file",
                      '-pf',
                      action='store_true',
                      help="Read parquet file")

  parser.add_argument("--date",
                      type=str,
                      required=False,
                      default=today_s,
                      help=f'Date like "{today_s}"')

  # get filename
  parser.add_argument("--filename",
                      '-f',
                      type=str,
                      default='',
                      help='What file name to read?')
  return parser.parse_args()


def get_equities_data(all_tickers, dry_run):
  # df = pd.read_csv(file)
  # df = df[(df['SERIES'] == 'EQ') | (df['SERIES'] == 'BE')]
  return all_tickers
  # if dry_run:
  #   df = yf.download(
  #       tickers=all_tickers[:DRY_RUN_NUM], period='1d', progress=False)
  # else:
  #   df = yf.download(all_tickers, period='550d', progress=False)

  # return df


def get_tickers_data(tickers,
                     dry_run,
                     read_parquet_file,
                     period=DRY_RUN_PERIOD):
  if dry_run:
    if read_parquet_file and os.path.exists(PARQUET_FILE):
      # Load the Parquet file into a DataFrame
      logger.info(f'Reading the parquel file: {PARQUET_FILE}')
      data = pd.read_parquet(PARQUET_FILE)
    else:
      data = yf.download(tickers[:DRY_RUN_NUM],
                         period=f'{DRY_RUN_PERIOD}',
                         progress=False)
  else:
    data = yf.download(tickers, period=period, progress=False)
    data.to_parquet(PARQUET_FILE)

  data = data.drop([HLOCs.CLOSE, HLOCs.OPEN], axis=1)
  # remove any data before 2019-12-30
  logger.info(f'data.index = {data.index}')
  data = data[data.index >= pd.to_datetime('2019-12-30')]
  return data


# def get_nse_indices_data(df):
#   # indices_info = [
#   #     ('^NSEI', 1), ('^CNXSC', 2), ('^CNXAUTO', 41), ('^NSEBANK', 42),
#   #     ('^CNXCMDT', 43), ('^CNXCONSUM', 44), ('^CNXENERGY', 46),
#   #     ('NIFTY_FIN_SERVICE.NS', 47), ('^CNXFMCG', 48), ('^CNXINFRA', 49),
#   #     ('^CNXIT', 50), ('^CNXMEDIA', 51), ('^CNXMETAL', 52), ('^CNXMNC', 53),
#   #     ('^CNXPHARMA', 54), ('^CNXPSE', 55), ('^CNXPSUBANK', 56),
#   #     ('^CNXREALTY', 58), ('^CNXSERVICE', 59)
#   # ]
#   # indices = [x[0] for x in indices_info]
#   # data = yf.download(indices, period='550d')
#   # closes = data[HLOCs.ADJ_CLOSE]
#   # logger.info(f'NSE INDICES closes: \n{closes}')
#   # for nse_index_name, df_index in indices_info:
#   #   df[COLUMN_NAMES[df_index]] = closes[nse_index_name]
#   # return data
#   return df


def get_plain_tickers(tickers):
  return tickers


def get_yfinance_ticker_names(tickers):
  return tickers


class TickDataCalculations:

  def __init__(self, tickers_data):
    self.tickers_data = tickers_data

  def calc_returns(self):
    tickers_data = self.tickers_data
    df_raw = pd.DataFrame()

    simple_sum = lambda frames: frames.iloc[:, 1:].sum(axis=1)

    returns_n = lambda n: tickers_data[HLOCs.ADJ_CLOSE].pct_change(periods=n)

    def get_returns(periods, return_threshold):
      returns = returns_n(periods)
      # Select the row where returns are greater than 4%
      returns_over_n_percent = returns[returns > return_threshold]
      returns_over_n_percent = returns_over_n_percent.count(axis=1)

      returns_under_n_percent = returns[returns < -return_threshold]
      returns_under_n_percent = returns_under_n_percent.count(axis=1)
      return returns_over_n_percent, returns_under_n_percent

    # Count the number of tickers with returns over 4% on each date
    returns_over_4_percent, returns_under_4_percent = get_returns(1, .04)
    df_raw['returns_over_4_percent'] = returns_over_4_percent
    df_raw['returns_under_4_percent'] = returns_under_4_percent
    logger.info(f'METRICS: returns_over_4_percent = {returns_over_4_percent}')
    logger.info(
        f'METRICS: returns_under_4_percent = {returns_under_4_percent}')

    # Count the number of tickers with returns over 15% in the last 5 days
    returns_over_15_percent, returns_under_15_percent = get_returns(5, .15)
    df_raw['returns_over_15_percent'] = returns_over_15_percent
    df_raw['returns_under_15_percent'] = returns_under_15_percent
    logger.info(
        f'METRICS: returns_over_15_percent = {returns_over_15_percent}')
    logger.info(
        f'METRICS: returns_under_15_percent = {returns_under_15_percent}')

    # Count the number of tickers with returns over 25% in the last 20 days
    returns_over_25_percent, returns_under_25_percent = get_returns(20, .25)
    df_raw['returns_over_25_percent'] = returns_over_25_percent
    df_raw['returns_under_25_percent'] = returns_under_25_percent
    logger.info(
        f'METRICS: returns_over_25_percent = {returns_over_25_percent}')
    logger.info(
        f'METRICS: returns_under_25_percent = {returns_under_25_percent}')

    net_breadth = returns_over_4_percent - returns_under_4_percent
    df_raw['net_breadth'] = net_breadth
    logger.info(f'METRICS: net_breadth = {net_breadth}')

    def _get_ratio(num):
      sum_of_last_N_days_over_4 = returns_over_4_percent.rolling(num).sum()
      sum_of_last_N_days_under_4 = returns_under_4_percent.rolling(num).sum()

      # Avoid division by zero and handle NaN and inf values
      with np.errstate(divide='ignore', invalid='ignore'):
        ratios = np.where(
            sum_of_last_N_days_under_4 != 0,
            sum_of_last_N_days_over_4 / sum_of_last_N_days_under_4, np.nan)
        ratios = np.where(np.isinf(ratios), np.nan, ratios)

      return ratios

    ratios_5 = _get_ratio(5)
    df_raw['ratios_5'] = ratios_5
    logger.info(f'METRICS: ratio 5 : {ratios_5}')

    ratios_10 = _get_ratio(10)
    df_raw['ratios_10'] = ratios_10
    logger.info(f'METRICS: ratio 10 : {ratios_10}')

    # Calculate daily range
    daily_range = (tickers_data[HLOCs.HIGH] -
                   tickers_data[HLOCs.LOW]) / tickers_data[HLOCs.ADJ_CLOSE]

    logger.info(f'METRICS: daily_range = \n{daily_range}')

    # df_raw['daily_range'] = daily_range

    dr_gt_5 = range_expansion = daily_range > 0.05
    range_expansion = simple_sum(range_expansion)

    close_gt_50_tmp = tickers_data[HLOCs.ADJ_CLOSE] > (
        tickers_data[HLOCs.LOW] +
        0.5 * daily_range * tickers_data[HLOCs.ADJ_CLOSE])
    range_plus_close_greater_than_5 = dr_gt_5 & close_gt_50_tmp

    range_plus_close_greater_than_5 = simple_sum(
        range_plus_close_greater_than_5)
    df_raw['close_gt_50'] = range_plus_close_greater_than_5

    logger.info(f'METRICS: close_gt_50 = \n{range_plus_close_greater_than_5}')

    close_lt_50 = range_expansion - range_plus_close_greater_than_5
    df_raw['close_lt_50'] = close_lt_50
    logger.info(f'METRICS: close_lt_50 = \n{close_lt_50}')

    uh_lh_ratio = range_plus_close_greater_than_5 / close_lt_50
    df_raw['uh_lh_ratio'] = uh_lh_ratio
    logger.info(f'METRICS: uh_lh_ratio = {uh_lh_ratio}')

    range_less_than_3 = simple_sum(daily_range < 0.03)
    df_raw['range_less_than_3'] = range_less_than_3

    df_raw['range_expansion'] = range_expansion
    df_raw['range_more_than_5'] = range_expansion
    logger.info(f'METRICS: range_expansion = \n{range_expansion}')

    range_contraction = daily_range < 0.03
    range_contraction = simple_sum(range_contraction)
    df_raw['range_contraction'] = range_contraction
    logger.info(f'METRICS: range_contraction = \n{range_contraction}')

    volatility_ratio = range_expansion / range_contraction
    df_raw['volatility_ratio'] = volatility_ratio
    logger.info(f'METRICS: volatility_ratio = \n{volatility_ratio}')

    five_day_high = tickers_data[HLOCs.HIGH].rolling(window=5, axis=0).max()
    # df_raw['five_day_high'] = five_day_high
    logger.info(f'METRICS: five_day_high = \n{five_day_high}')

    five_day_low = tickers_data[HLOCs.LOW].rolling(window=5, axis=0).min()
    # df_raw['five_day_low'] = five_day_low
    logger.info(f'METRICS: five_day_low = \n{five_day_low}')

    day_range_5 = (five_day_high - five_day_low) / five_day_low

    logger.info(f'METRICS: day_range_5 = \n{day_range_5}')
    # df_raw['day_range_5'] = day_range_5

    breakout = tickers_data[
        HLOCs.HIGH] > tickers_data[HLOCs.ADJ_CLOSE].shift() * 1.04
    breakout = simple_sum(breakout)
    df_raw['breakout'] = breakout
    logger.info(f'METRICS: breakout = \n{breakout}')

    breakdown = tickers_data[
        HLOCs.LOW] < tickers_data[HLOCs.ADJ_CLOSE].shift() * 0.96
    logger.info(f'METRICS: breakdown 00 = \n{breakdown}')

    breakdown = simple_sum(breakdown)
    df_raw['breakdown'] = breakdown
    logger.info(f'METRICS: breakdown = \n{breakdown}')

    bo_bd_ratio = breakout / breakdown
    df_raw['bo_bd_ratio'] = bo_bd_ratio
    logger.info(f'METRICS: bo_bd_ratio = \n{bo_bd_ratio}')

    up_close = returns_over_4_percent / breakout
    df_raw['up_close'] = up_close
    logger.info(f'METRICS: up_close = \n{up_close}')

    down_close = returns_under_4_percent / breakdown
    df_raw['down_close'] = down_close
    logger.info(f'METRICS: down_close = \n{down_close}')

    breakout_sustained = tickers_data[HLOCs.ADJ_CLOSE] > (
        tickers_data[HLOCs.HIGH] -
        tickers_data[HLOCs.ADJ_CLOSE] * 0.4 * daily_range)
    breakout_sustained = simple_sum(breakout_sustained)
    df_raw['breakout_sustained'] = breakout_sustained
    logger.info(f'METRICS: breakout_sustained = \n{breakout_sustained}')

    breakdown_sustained = tickers_data[HLOCs.ADJ_CLOSE] < (
        tickers_data[HLOCs.HIGH] - 0.4 * daily_range)
    breakdown_sustained = simple_sum(breakdown_sustained)
    df_raw['breakdown_sustained'] = breakdown_sustained
    logger.info(f'METRICS: breakdown_sustained = \n{breakdown_sustained}')

    # highs_52wk = tickers_data[HLOCs.HIGH].max()
    # df_raw['highs_52wk'] = highs_52wk
    # logger.info(f'METRICS: highs_52wk = \n{highs_52wk}')

    # lows_52wk = tickers_data[HLOCs.HIGH].min()
    # df_raw['lows_52wk'] = lows_52wk
    # logger.info(f'METRICS: lows_52wk = \n{lows_52wk}')

    # high_within_15_pct = tickers_data[HLOCs.ADJ_CLOSE] > (0.85 * highs_52wk)
    # logger.info(f'DDEBUG: high_within_15_pct = {high_within_15_pct}')
    # high_within_15_pct = simple_sum(high_within_15_pct)
    # df_raw['high_within_15_pct'] = high_within_15_pct
    # logger.info(f'METRICS: high_within_15_pct = \n{high_within_15_pct}')

    # low_within_15_pct = tickers_data[HLOCs.ADJ_CLOSE] < (1.15 * lows_52wk)
    # low_within_15_pct = simple_sum(low_within_15_pct)
    # df_raw['low_within_15_pct'] = low_within_15_pct
    # logger.info(f'METRICS: low_within_15_pct = \n{low_within_15_pct}')

    # hl_ratio_15_pct = high_within_15_pct / low_within_15_pct
    # df_raw['hl_ratio_15_pct'] = hl_ratio_15_pct
    # logger.info(f'METRICS: hl_ratio_15_pct = \n{hl_ratio_15_pct}')

    # high_within_30_pct = tickers_data[HLOCs.ADJ_CLOSE] > (0.7 * highs_52wk)
    # high_within_30_pct = simple_sum(high_within_30_pct)
    # df_raw['high_within_30_pct'] = high_within_30_pct
    # logger.info(f'METRICS: high_within_30_pct = \n{high_within_30_pct}')

    # low_within_30_pct = tickers_data[HLOCs.ADJ_CLOSE] < (1.3 * lows_52wk)
    # low_within_30_pct = simple_sum(low_within_30_pct)
    # df_raw['low_within_30_pct'] = low_within_30_pct
    # logger.info(f'METRICS: low_within_30_pct = \n{low_within_30_pct}')

    # hl_ratio_30_pct = high_within_30_pct / low_within_30_pct
    # df_raw['hl_ratio_30_pct'] = hl_ratio_30_pct
    # logger.info(f'METRICS: hl_ratio_30_pct = \n{hl_ratio_30_pct}')

    return df_raw

  def calc_highs_and_lows(self, tickers, df_raw, dry_run):
    tickers_data = self.tickers_data
    periods_in_year = 260 if dry_run else 260
    simple_sum = lambda frames: frames.iloc[:, 1:].sum(axis=1)

    closes = tickers_data[HLOCs.ADJ_CLOSE].copy()
    highs = tickers_data[HLOCs.HIGH].copy()
    lows = tickers_data[HLOCs.LOW].copy()
    week_52_highs = highs.rolling(window=periods_in_year).max()
    logger.info(f'week_52_highs = {week_52_highs}')
    week_52_lows = lows.rolling(window=periods_in_year).min()
    df_raw['52_week_high'] = simple_sum(week_52_highs == closes)
    df_raw['52_week_low'] = simple_sum(week_52_lows == closes)

    pct_within_high = lambda pct: simple_sum(closes >
                                             (100 - pct) * week_52_highs / 100)
    pct_within_low = lambda pct: simple_sum(closes <
                                            (100 + pct) * week_52_lows / 100)

    df_raw['15_pct_from_52_wh'] = pct_within_high(15)
    df_raw['30_pct_from_52_wh'] = pct_within_high(
        30) - df_raw['15_pct_from_52_wh']
    df_raw['50_pct_from_52_wh'] = pct_within_high(
        50) - df_raw['15_pct_from_52_wh'] - df_raw['30_pct_from_52_wh']
    df_raw['70_pct_from_52_wh'] = pct_within_high(
        70) - df_raw['15_pct_from_52_wh'] - df_raw[
            '30_pct_from_52_wh'] - df_raw['50_pct_from_52_wh']

    df_raw['15_pct_from_52_wl'] = pct_within_low(15)
    df_raw['30_pct_from_52_wl'] = pct_within_low(
        30) - df_raw['15_pct_from_52_wl']
    df_raw['50_pct_from_52_wl'] = pct_within_low(
        50) - df_raw['15_pct_from_52_wl'] - df_raw['30_pct_from_52_wl']
    df_raw['90_pct_from_52_wl'] = pct_within_low(
        90) - df_raw['15_pct_from_52_wl'] - df_raw[
            '30_pct_from_52_wl'] - df_raw['50_pct_from_52_wl']
    df_raw['150_pct_from_52_wl'] = pct_within_low(150) - df_raw[
        '15_pct_from_52_wl'] - df_raw['30_pct_from_52_wl'] - df_raw[
            '50_pct_from_52_wl'] - df_raw['90_pct_from_52_wl']
    logger.info(f'DDEBUG: closes = {closes}')

    logger.info(f'DDEBUG: week_52_highs = {week_52_highs}')
    logger.info(f'DDEBUG: week_52_lows = {week_52_lows}')

  def calc_volumes(self, tickers, df_raw):
    tickers_data = self.tickers_data
    simple_sum = lambda frames: frames.iloc[:, 1:].sum(axis=1)

    vol_count = 0
    volumes = tickers_data[HLOCs.VOLUME].copy()
    closes = tickers_data[HLOCs.ADJ_CLOSE].copy()
    close_d10 = tickers_data[HLOCs.ADJ_CLOSE].copy()
    close_d20 = tickers_data[HLOCs.ADJ_CLOSE].copy()
    vol_d20 = tickers_data[HLOCs.VOLUME].copy()
    close_d50 = tickers_data[HLOCs.ADJ_CLOSE].copy()
    close_d200 = tickers_data[HLOCs.ADJ_CLOSE].copy()

    # Calculate DEMA for each column of ticker's volume data
    for ticker in volumes.columns:
      vol_col = volumes[ticker]
      close_col = closes[ticker]

      close_dema_10 = talib.DEMA(close_col, timeperiod=10)
      close_dema_20 = talib.DEMA(close_col, timeperiod=20)
      vol_dema_20 = talib.DEMA(vol_col, timeperiod=20)

      close_dema_50 = talib.DEMA(close_col, timeperiod=50)
      close_dema_200 = talib.DEMA(close_col, timeperiod=200)

      close_d10[ticker] = close_dema_10
      close_d20[ticker] = close_dema_20
      vol_d20[ticker] = vol_dema_20
      close_d50[ticker] = close_dema_50
      close_d200[ticker] = close_dema_200

    def compute_metric(field_values, comparator, multiplier, d_n, debug=False):
      results = None
      if comparator == '>':
        results = field_values > multiplier * d_n
      elif comparator == '<':
        results = field_values < multiplier * d_n
      results = results.sum(axis=1)
      return results

    df_raw['below_avg_vol'] = compute_metric(volumes, '<', 0.5, vol_d20)
    df_raw['above_avg_vol'] = compute_metric(volumes, '>', 1.5, vol_d20, True)

    df_raw['10_pct_above_10_dema'] = compute_metric(closes, '>', 1.1,
                                                    close_d10)
    df_raw['10_pct_below_10_dema'] = compute_metric(closes, '<', 0.9,
                                                    close_d10)

    df_raw['above_10_dema'] = compute_metric(closes, '>', 1, close_d10)
    df_raw['above_20_dema'] = compute_metric(closes, '>', 1, close_d20)
    df_raw['above_50_dema'] = compute_metric(closes, '>', 1, close_d50)
    df_raw['above_200_dema'] = compute_metric(closes, '>', 1, close_d200)

  def check_stats(self, tickers):
    yf_tickers = get_yfinance_ticker_names(tickers)
    self.tickers_data = interpolate_vals(self.tickers_data)
    df_raw = self.calc_returns()

    return df_raw


def interpolate_vals(df):
  df[HLOCs.VOLUME] = df[HLOCs.VOLUME].interpolate(method='linear')
  df[HLOCs.ADJ_CLOSE] = df[HLOCs.ADJ_CLOSE].interpolate(method='linear')
  df[HLOCs.HIGH] = df[HLOCs.HIGH].interpolate(method='linear')
  df[HLOCs.LOW] = df[HLOCs.LOW].interpolate(method='linear')
  return df


def map_df_raw_to_excel(df_raw, tickers_data):
  df_raw = df_raw.fillna(0)

  # Create an empty DataFrame
  df = pd.DataFrame(columns=COLUMN_NAMES)
  df[COLUMN_NAMES[0]] = df_raw.index
  df.index = df_raw.index

  closes_above_1 = tickers_data[HLOCs.ADJ_CLOSE] > 1
  # univ_count = closes_above_1.count(axis=1)
  univ_count = closes_above_1.iloc[:, 1:].sum(axis=1)
  # pct = lambda x: df_raw[x] * 100 / univ_count # percent values
  pct = lambda x: df_raw[x]
  logger.info(f'univ_count = {univ_count.to_string()}')

  df[COLUMN_NAMES[3]] = univ_count
  df[COLUMN_NAMES[4]] = pct('returns_over_4_percent')
  df[COLUMN_NAMES[5]] = pct('returns_under_4_percent')
  df[COLUMN_NAMES[6]] = pct('above_avg_vol')
  df[COLUMN_NAMES[7]] = pct('below_avg_vol')
  df[COLUMN_NAMES[8]] = pct('range_less_than_3')
  df[COLUMN_NAMES[9]] = pct('range_more_than_5')

  df[COLUMN_NAMES[10]] = pct('close_gt_50')
  df[COLUMN_NAMES[11]] = pct('close_lt_50')
  df[COLUMN_NAMES[12]] = pct('breakout')
  df[COLUMN_NAMES[13]] = pct('breakout_sustained')
  df[COLUMN_NAMES[14]] = None  # Missing

  df[COLUMN_NAMES[15]] = pct('breakdown')
  df[COLUMN_NAMES[16]] = pct('breakdown_sustained')
  df[COLUMN_NAMES[17]] = None  # Missing

  df[COLUMN_NAMES[18]] = pct('returns_over_15_percent')
  df[COLUMN_NAMES[19]] = pct('returns_under_15_percent')
  df[COLUMN_NAMES[20]] = pct('returns_over_25_percent')
  df[COLUMN_NAMES[21]] = pct('returns_under_25_percent')

  df[COLUMN_NAMES[22]] = pct('10_pct_above_10_dema')
  df[COLUMN_NAMES[23]] = pct('10_pct_below_10_dema')
  df[COLUMN_NAMES[24]] = pct('above_10_dema')
  df[COLUMN_NAMES[25]] = pct('above_20_dema')
  df[COLUMN_NAMES[26]] = pct('above_50_dema')
  df[COLUMN_NAMES[27]] = pct('above_200_dema')

  df[COLUMN_NAMES[28]] = pct('52_week_high')
  df[COLUMN_NAMES[29]] = pct('52_week_low')
  df[COLUMN_NAMES[30]] = pct('15_pct_from_52_wh')
  df[COLUMN_NAMES[31]] = pct('30_pct_from_52_wh')
  df[COLUMN_NAMES[32]] = pct('50_pct_from_52_wh')
  df[COLUMN_NAMES[33]] = pct('70_pct_from_52_wh')

  df[COLUMN_NAMES[35]] = pct('15_pct_from_52_wl')
  df[COLUMN_NAMES[36]] = pct('30_pct_from_52_wl')
  df[COLUMN_NAMES[37]] = pct('50_pct_from_52_wl')
  df[COLUMN_NAMES[38]] = pct('90_pct_from_52_wl')
  df[COLUMN_NAMES[39]] = pct('150_pct_from_52_wl')

  # add indices prices too
  # get_nse_indices_data(df)
  '''

  df_raw['net_breadth'] = net_breadth
  df_raw['ratios_5'] = ratios_5
  df_raw['ratios_10'] = ratios_10
  df_raw['uh_lh_ratio'] = uh_lh_ratio
  df_raw['volatility_ratio'] = volatility_ratio
  df_raw['bo_bd_ratio'] = bo_bd_ratio
  df_raw['up_close'] = up_close
  df_raw['down_close'] = down_close

      "Breakout Failure",  #14
      "Breakdown Failure",  #17

      "70% Plus From 52WH",  #34
      "150% Plus From 52WL",  #40
  '''

  # df[column_names[6]] = df_raw['vol_res']
  # df[column_names[7]] = df_raw['vol_res']

  return df


def write_df_to_excel(excel_df, dry_run):
  breadth_xl_file_path = get_mkt_breadth_xl_path(dry_run, False)
  backup_xl_file_path = get_mkt_breadth_xl_path(dry_run, True)
  create_backup(breadth_xl_file_path, backup_xl_file_path)
  updated_mkt_breadth = get_filtered_mkt_breadth_by_ts(breadth_xl_file_path,
                                                       excel_df)
  if not updated_mkt_breadth.empty:
    logger.info(f'writing updated_mkt_breadth to {breadth_xl_file_path}; '
                f'updated_mkt_breadth=\n{updated_mkt_breadth}')
    write_updated_mkt_breadth_to_xl(updated_mkt_breadth, breadth_xl_file_path)
  else:
    logger.info(f'empty updated_mkt_breadth. Writing excel_df instead'
                f'; excel_df=\n{excel_df}')
    write_updated_mkt_breadth_to_xl(excel_df, breadth_xl_file_path)


def create_backup(orig_file, new_file):
  if os.path.exists(orig_file):
    logger.info(f'copying {orig_file} to {new_file}')
    shutil.copyfile(orig_file, new_file)


def get_filtered_mkt_breadth_by_ts(greeks_df_file, df):
  if not os.path.exists(greeks_df_file):
    return pd.DataFrame()
  # Load existing data from Excel
  existing_df = pd.read_excel(greeks_df_file,
                              engine='openpyxl',
                              sheet_name="raw")
  logger.info(f'existing_df 1 = \n{existing_df}')
  logger.info(f'df = \n{df}; shape={df.shape}')
  # existing_df = existing_df.reset_index(drop=True)

  existing_df_ts = pd.to_datetime(existing_df['Date/Time'])
  logger.info(f'vals = {existing_df_ts}')
  existing_df_ts_type = type(existing_df_ts.iloc[0])
  logger.info(f'AA: existing_df_ts_type 0 : {existing_df_ts[2]}; '
              f'existing_df_ts_type = {existing_df_ts_type}')

  logger.info(f'existing_df  2= \n{existing_df}')
  logger.info(f'existing_df cols = {existing_df.columns}')
  logger.info(f'cols = {df.columns}')
  val0 = df.index
  logger.info(f'val0 = {val0}')

  # Find new timestamps that don't exist in the existing data
  new_timestamps = df[~df.index.isin(existing_df_ts)]
  logger.info(f'new_timestamps = \n{new_timestamps}')
  for idx, ts in new_timestamps.iterrows():
    values_list = ts.tolist()
    len_values = len(values_list)
    len_cols = len(existing_df.columns)
    logger.info(f'len_values = {len_values}; len_cols = {len_cols}')
    logger.info(f'idx: {idx}; values_list = {values_list}')
    existing_df.loc[len(existing_df.index)] = values_list

    logger.info(f'idx  ={idx}; row = {values_list}')

  # Append only the new rows to the existing DataFrame
  # df_to_add = pd.concat([existing_df, new_timestamps]).drop_duplicates()
  # logger.info(f'df_to_add = \n{df_to_add}')
  logger.info(f'existing_df 3 = \n{existing_df}')
  return existing_df


def write_updated_mkt_breadth_to_xl(updated_df, greeks_xl_file_path):
  try:
    does_file_exist = os.path.exists(greeks_xl_file_path)
    writer = None
    if does_file_exist:
      writer = pd.ExcelWriter(greeks_xl_file_path, engine="openpyxl", mode='w')
    else:
      writer = pd.ExcelWriter(greeks_xl_file_path, engine="openpyxl", mode='w')

    # Set the number format for the entire column A to display dates as YYYY-MM-DD
    date_format = NamedStyle(name='date_format')
    date_format.number_format = 'YYYY-MM-DD'
    # Set the number format for the entire column B to display numbers with 2 decimals
    number_format = NamedStyle(name='number_format')
    number_format.number_format = '0.00'
    logger.info(
        f'greeks_xl_file_path= {greeks_xl_file_path}; updated_df=\n{updated_df}'
    )

    # Convert the dataframe to an openpyxl Excel object.
    updated_df.to_excel(writer, sheet_name="raw", header=True, index=False)

    # Get the openpyxl workbook and worksheet objects.
    workbook = writer.book
    worksheet = workbook["raw"]

    # Get the dimensions of the dataframe.
    (max_row, max_col) = updated_df.shape
    max_row += 3

    # Define the starting and ending column indices
    start_col_index = 3  # Column 'C'
    end_col_index = 55  # Column 'BC'

    # Get the column names from 'A' to 'BC'
    column_names = [
        get_column_letter(col_idx)
        for col_idx in range(start_col_index, end_col_index + 1)
    ]

    for cell in worksheet['A']:
      cell.style = date_format

    for ws_col in column_names:
      for cell in worksheet[ws_col]:
        cell.style = number_format

    for ws_col in column_names:
      if ws_col == 'AP':
        break
      cell_range = f'{ws_col}2:{ws_col}{max_row}'

      worksheet.conditional_formatting.add(
          cell_range,
          ColorScaleRule(start_type='min',
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
  except Exception as e:
    logger.error(f'Exception : {str(e)}')


def get_mkt_breadth_xl_path(dry_run, add_suffix: bool):
  file_prefix = "dry-" if dry_run else ""
  backup_suffix = ''
  if add_suffix:
    today = datetime.datetime.today()
    backup_suffix = f"_{today.strftime('%Y%m%d')}"
  greeks_xl_file = os.path.join(
      SAVE_FOLDER, f"{file_prefix}market-breadth-spx{backup_suffix}.xlsx")
  return greeks_xl_file


def save_local_tick_data(tickers_data):
  # Assuming tickers_data is your DataFrame and you want to split it into batches of 900 columns
  total_columns = len(tickers_data.columns)
  batch_size = 900
  num_batches = (total_columns //
                 batch_size) + 1  # Calculate the number of batches needed

  for i in range(num_batches):
    start_idx = i * batch_size
    end_idx = min(
        (i + 1) * batch_size,
        total_columns)  # Ensure the last batch might have fewer columns
    batch_data = tickers_data.iloc[:, start_idx:
                                   end_idx]  # Slice the DataFrame for the current batch
    batch_data.to_excel(
        f'data/spx/tickers_data_batch_{i + 1}.xlsx',
        index=False)  # Save each batch to a separate Excel file


def run_analysis(args):
  tickers = _get_list_of_companies()
  # df = get_equities_data(all_tickers, args.dry_run)
  # tickers = list(set(df['SYMBOL'].to_list()))

  # logger.info(f'df = {df}')
  # logger.info(f'df shape = {df.shape}')
  logger.info(f'tickers = {tickers}')
  logger.info(f'tickers len = {len(tickers)}')
  tickers_data = get_tickers_data(tickers, args.dry_run,
                                  args.read_parquet_file)
  logger.info(f'tickers_data = \n{tickers_data}')
  logger.info(f'tickers_data shape= {tickers_data.shape}')

  if False:
    save_local_tick_data(tickers_data[HLOCs.ADJ_CLOSE])

  tickers_calc = TickDataCalculations(tickers_data)
  df_raw = tickers_calc.check_stats(tickers)
  logger.info(f'finalized: \n{tickers_data}')

  vol_res = tickers_calc.calc_volumes(tickers, df_raw)
  tickers_calc.calc_highs_and_lows(tickers, df_raw, args.dry_run)
  logger.info(f'df_raw 0 = \n{df_raw}')
  logger.info(f'vol_res 0 = \n{vol_res}')
  excel_df = map_df_raw_to_excel(df_raw, tickers_data)

  logger.info(f'excel_df = \n{excel_df}')
  write_df_to_excel(excel_df, args.dry_run)


def _get_list_of_companies():
  df = pd.read_csv(COMPANIES_LIST_FILE)
  list_companies = df['Symbol'].to_list()
  logger.info(f'list_companies = {list_companies}')
  logger.info(f'len  = {len(list_companies)}')
  return list_companies


def main():
  args = process_args()
  logger.info(f'args = {args}')

  if args.date:
    date_obj = datetime.datetime.strptime(args.date, "%Y%m%d")
    # Validate the date format

  run_analysis(args)


if __name__ == '__main__':
  main()
