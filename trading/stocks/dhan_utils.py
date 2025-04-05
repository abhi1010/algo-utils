import warnings

# with warnings.catch_warnings():
#     warnings.simplefilter(action='ignore', category=FutureWarning)
# warnings.filterwarnings("ignore")

import yaml
import pandas as pd
from enum import Enum
import os, sys
import logging
from collections import namedtuple, OrderedDict
import multiprocessing
import math
import re
import traceback

import requests
import datetime
import time
import argparse
from trading.configs import txn_weekend
# warnings.filterwarnings("error")
from trading.stocks.dhan_common import *

import warnings

warnings.simplefilter(
    action='ignore'
)  # setting ignore as a parameter and further adding category

import yfinance as yf
import yahoo_fin.stock_info as si
import pandas_ta as ta
import numpy as np

from trading.common import utils

logger = utils.get_logger('dhan_utils')

from trading.tables import yf_download_rsi as yf_download
from trading.strategies import relative_strength
from trading.portfolio.tracker import *
from trading.stocks import dhan
from trading.services import telegram_runner

from dhanhq import dhanhq

NIFTY_CONSTITUENTS = 'data/ind_nifty500list.csv'
INITIAL_SL = 3  # %
TAG_IS_WEEKEND = 'IsWeekend'
date_yyyy_mm_dd_FMT = '%Y%m%d'


class DhanTickerInfo:

  def __init__(self, ticker, security_id, px, qty=0):
    self.ticker = ticker
    self.security_id = str(security_id)
    self.price = px
    self.qty = qty

  def __str__(self):
    return f'{self.ticker} - {self.security_id} - [[{self.qty} @ {self.price}]]'

  def __repr__(self):
    return str(self)

  def get_quantity(self, notional):
    # assert (self.price > 0, f"Price is expected to be non-0 for {self.ticker}")
    self.qty = math.floor(notional / self.price)
    return self.qty


class TxnHelper:

  def __init__(self):
    self.txn_mgr = txn_weekend.TxnManager()

  def get_all_trades_from_db(self):
    # provide a list for the last 2 days

    date_today = datetime.datetime.today().strftime('%Y-%m-%d')

    # df_trades = self.txn_mgr.get_all_txn_rows_as_df(
    #     date_today, DEFAULT_TAG, 'trades')
    # logger.info(
    #     f'get_all_trades_from_db: txn: df_trades 1 = \n{df_trades.to_string()}')

    date_today = datetime.datetime.today()
    df_trades = pd.DataFrame()  # Initialize an empty DataFrame to store trades

    for i in range(0, 5):  # Loop over the past 6 days
      date_past = (date_today - datetime.timedelta(days=i)).strftime('%Y-%m-%d')
      # Retrieve trades for the current day and merge them into df_trades
      df_daily_trades = self.txn_mgr.get_all_txn_rows_as_df(
          date_past, DEFAULT_TAG, 'tradebook')
      df_trades = pd.concat([df_trades, df_daily_trades], ignore_index=True)
    # row_trades = txn_mgr.get_all_txn_rows(date_today, DEFAULT_TAG, 'trades')
    # logger.info(f'df_trades 2 = \n{row_trades}')
    logger.info(
        f'get_all_trades_from_db : df_trades = \n{df_trades.to_string()}')
    return df_trades

  def get_matching_orders(self, df):
    df[TAG_IS_WEEKEND] = False
    for idx, row in df.iterrows():
      # logger.info(f'row = {row.to_dict()}')
      orderid = row['orderId']
      order_rows = self.txn_mgr.get_rows(
          DEFAULT_TAG, 'orders', f'"orderId": "{orderid}"')
      # "correlationId": "wknd
      if order_rows and len(order_rows):
        df.at[idx, TAG_IS_WEEKEND] = True
      logger.info(f'txn: row = {row.to_dict()}; \n\t order_rows = {order_rows}')

  def get_weekend_trades_from_txn(self):
    df = self.get_all_trades_from_db()
    self.get_matching_orders(df)
    df_weekend = df[df[TAG_IS_WEEKEND]]
    return df_weekend


class DhanTasks:

  def __init__(self):
    pass

  @staticmethod
  def task_run_gainers_list_creation(num_of_tickers, dry_run):
    tickers, tickers_data, returns = DhanYfinanceTickers.get_tickers_and_returns(
        num_of_tickers, dry_run)
    top_N_tickers = DhanYfinanceTickers.get_top_tickers(returns, num_of_tickers)
    df_ticker_with_closes = write_gainers_list(top_N_tickers, tickers_data)
    logger.info(f'df_ticker_with_closes = \n{df_ticker_with_closes}')

  @staticmethod
  def task_update_stop_loss_orders_for_holdings(
      stop_loss_pct=INITIAL_SL,
      place_orders=False,
      fixed_sl=False,
      close_price_col='High',
      product_type=dhanhq.CNC,
      after_market_order=False):

    dhan_ins = dhan.DhanTracker()
    p_filter = PortfolioFilters(dhan_ins=dhan_ins)
    curr_weekend_holdings = p_filter._get_current_weekend_holdings(
        fixed_sl, stop_loss_pct, close_price_col)

    if curr_weekend_holdings.empty:
      logger.info(f'No holdings for today')
      return

    all_orders = DhanActions._place_stop_loss_orders(
        curr_weekend_holdings,
        place_orders,
        dhan_ins,
        after_market_order=after_market_order,
        product_type=product_type,
        qty_col='qty_to_close')

    if place_orders:
      DhanDebugger.show_open_orders(False)
    if len(all_orders):
      add_results_into_db(all_orders, delay=15 if place_orders else 1)

  @staticmethod
  def task_check_and_cancel_pending_orders(place_orders):
    df_orders = DhanDebugger.show_open_orders(False)
    pending_orders = DhanActions.find_all_upper_circuit_pending_orders(
        df_orders)
    logger.info(f'orders = \n{df_orders.to_string()}')
    logger.info(f'pending_orders = \n{pending_orders.to_string()}')
    DhanActions.cancel_pending_orders(pending_orders, place_orders)

  @staticmethod
  def task_close_holdings(product_type, dry_run):
    close_column_name = 'qty_to_close'
    dhan_tracker = dhan.DhanTracker()

    holdings = dhan_tracker.get_holdings()
    logger.info(f'tch: holdings = {holdings};')
    logger.info(f'tch: About to show holdings from dhan API')
    dhan.show_debug_info(
        holdings, 'holdings', save_to_file=True, save_to_db=True)
    p_filter = PortfolioFilters(dhan_ins=dhan_tracker)
    try:
      curr_weekend_holdings = p_filter._get_current_weekend_holdings(
          fixed_sl=True, stop_loss_pct=INITIAL_SL, close_price_col='Close')
      if curr_weekend_holdings.empty:
        logger.info(f'st. g.c.w.h: curr_weekend_holdings is empty')

        # only replace holdings if the len of holdings is <=5
        if len(holdings) <= 5:
          holdings = pd.DataFrame(holdings)
          holdings[close_column_name] = holdings['availableQty']
          logger.info(f'st. g.c.w.h: updated holdings = {holdings}')
        else:
          logger.info(f'st. g.c.w.h: holdings length > 5, not updated holdings')
      else:
        # TODO: fix me later on, shouldnt have to call dhan_tracker.get_holdings()
        holdings = curr_weekend_holdings
        logger.info('st. g.c.w.h: Replaced Holdings with curr_weekend_holdings')

      logger.info(
          f'st. g.c.w.h: curr_weekend_holdings = \n{curr_weekend_holdings.to_string()}'
      )
    except Exception as e:
      logger.info(f'Exception in get_current_weekend_holdings: {e}')
      exc = traceback.format_exc()
      logger.info(f'exception = {exc}')
      logger.info(f'*' * 80)
      stacks = ''.join(traceback.format_stack())
      logger.info(f'stacks = {stacks}')
      logger.info('Exception END')

    logger.info(f'About to show holdings from tht we have inferred so far...')

    dhan.show_debug_info(
        holdings, 'holdings', save_to_file=False, save_to_db=False)

    logger.info(f'tch: About to cancel SLM orders')
    DhanActions.cancel_stoploss_market_orders(dhan_tracker, dry_run=dry_run)
    logger.info(
        f'tch: Cancelled SLM orders. Waiting for a while before closing.')
    time.sleep(1 if dry_run else 5)

    # print the line if instance of type df
    if isinstance(holdings, pd.DataFrame):
      logger.info(f'tch: holdings = \n{holdings.to_string()}')
    else:
      logger.info(f'tch: holdings = \n{holdings}')

    order_results = dhan_tracker.close_all_assets(
        holdings,
        'holdings',
        after_market_order=False,
        product_type=dhanhq.CNC,
        dry_run=dry_run,
        close_column_name=close_column_name)

    add_results_into_db(order_results, delay=1 if dry_run else 15)
    return order_results
    # full_update = dhan_tracker.close_all_positions_amo(
    #     after_market_order=False, product_type=dhanhq.CNC)

  @staticmethod
  def task_open_positions_and_orders(
      num_of_tickers, notional, product_type, dry_run, place_orders,
      df_ticker_with_closes):
    ticker_names = df_ticker_with_closes['ticker'].to_list()
    logger.info(f'ticker_names = \n{ticker_names}')
    logger.info(f'Read data: {df_ticker_with_closes}')

    ticker_close_dict = DhanYfinanceTickers.get_ticker_close_dict(
        df_ticker_with_closes)
    dhan_sec_ids = DhanYfinanceTickers.get_dhan_security_ids(ticker_names)
    DhanYfinanceTickers.set_closing_prices_and_quantities(
        dhan_sec_ids, ticker_close_dict, notional)
    order_results = DhanActions.open_positions(
        dhan_sec_ids,
        product_type=product_type,
        place_orders=place_orders,
        tag='wknd')

    time.sleep(10 if place_orders else 1)
    if place_orders:
      DhanDebugger.show_open_orders(False)
    add_results_into_db(order_results, delay=5 if place_orders else 1)
    return order_results

  @staticmethod
  def task_set_stop_loss_orders_for_positions(
      stop_loss_pct, place_orders=False, tag_prefix='wknd-sl'):
    after_market_order = False
    date_yyyy_mm_dd = datetime.datetime.today().strftime(date_yyyy_mm_dd_FMT)
    dhan_ins = dhan.DhanTracker()

    p_filter = PortfolioFilters(dhan_ins=dhan_ins)
    positions_for_today = p_filter.get_curr_weekend_trades_summarized()

    positions_for_today['price_sl'] = round(
        positions_for_today['tradedPrice'] * (100 - stop_loss_pct) / 100, 2)

    logger.info(f'positions_for_today = \n{positions_for_today.to_string()}')

    all_orders = []
    for idx, row in positions_for_today.iterrows():
      logger.info(f'row: {row}')
      ticker = row['tradingSymbol']
      qty = row['tradedQuantity']
      price_sl = int(row['price_sl'] * 10) / 10
      productType = row['productType']
      securityId = row['securityId']

      full_tag = f'{tag_prefix}_{date_yyyy_mm_dd}_{securityId}'
      if place_orders:
        logger.info(f'Placing SL order for {ticker}')
        order_res = dhan_ins.place_order(
            securityId,
            qty,
            after_market_order=after_market_order,
            product_type=productType,
            order_type=dhanhq.SLM,
            trigger_price=price_sl,
            transaction_type=dhanhq.SELL,
            tag=full_tag)
        all_orders.append(order_res)
        logger.info(f'order_res = {order_res}')
      else:
        logger.info(
            f'SL placement disabled. Skipping for: {ticker}; SL={price_sl}')

    if len(all_orders):
      add_results_into_db(all_orders, delay=15 if place_orders else 1)

  @staticmethod
  def task_set_stop_loss_orders_for_holdings(
      stop_loss_pct=INITIAL_SL, place_orders=False, tag_prefix='wknd-hl'):
    dhan_ins = dhan.DhanTracker()
    date_yyyy_mm_dd = datetime.datetime.today().strftime(date_yyyy_mm_dd_FMT)
    p_filter = PortfolioFilters(dhan_ins=dhan_ins)
    curr_weekend_holdings = p_filter._get_current_weekend_holdings(
        fixed_sl=True,
        stop_loss_pct=stop_loss_pct,
        close_price_col='Close',
        get_current=True)
    logger.info(
        f'curr_weekend_holdings = \n{curr_weekend_holdings.to_string()}')

    after_market_order = False
    holdings = dhan_ins.get_holdings()
    logger.info(f'holdings = \n{holdings}')
    if not holdings:
      logger.info(f'Empty holdings')
      return
    df_holdings = pd.DataFrame(holdings) if holdings else pd.DataFrame()
    productType = dhanhq.CNC
    logger.info(f'df_holdings = \n{df_holdings.to_string()}')

    holdings_for_today = curr_weekend_holdings[
        curr_weekend_holdings['outstanding_trade_qty'] > 0]
    holdings_for_today['price_sl'] = round(
        holdings_for_today['avgCostPrice'] * (100 - stop_loss_pct) / 100, 2)
    logger.info(f'holdings_for_today = {holdings_for_today}')

    all_orders = []
    for idx, row in holdings_for_today.iterrows():
      logger.info(f'row: {row}')
      ticker = row['tradingSymbol']
      qty = row['outstanding_trade_qty']
      price_sl = int(row['price_sl'] * 10) / 10
      securityId = row['securityId']
      tag_full = f'{tag_prefix}_{date_yyyy_mm_dd}_{securityId}'

      if place_orders:
        logger.info(f'Placing SL order for {ticker}; price_sl={price_sl}')
        order_res = dhan_ins.place_order(
            securityId,
            qty,
            after_market_order=after_market_order,
            product_type=productType,
            order_type=dhanhq.SLM,
            trigger_price=price_sl,
            transaction_type=dhanhq.SELL,
            tag=tag_full)
        all_orders.append(order_res)
        logger.info(f'order_res = {order_res}')
      else:
        logger.info(f'Not placing SL order for {ticker}; price_sl={price_sl}')

    if place_orders:
      DhanDebugger.show_open_orders(False)
    if len(all_orders):
      add_results_into_db(all_orders, delay=15 if place_orders else 1)

  @staticmethod
  def task_enforce_position_cap(
      num_of_tickers, notional, product_type, dry_run, place_orders):
    # TODO: How to find out which ones are weekend trades?
    valid_existing_trades = get_existing_valid_trades()
    if len(valid_existing_trades) == num_of_tickers:
      logger.info(
          f'No need to enforce position cap. Its already at {num_of_tickers}')
      return
    dhan_ins = dhan.DhanTracker()
    p_filter = PortfolioFilters(dhan_ins=dhan_ins)
    try:
      curr_weekend_holdings = p_filter._get_current_weekend_holdings(
          fixed_sl=True, stop_loss_pct=None, close_price_col=None)

      logger.info(
          f'curr_weekend_holdings = {curr_weekend_holdings.to_string()}')
    except Exception as e:
      logger.info(f'Exception in get_current_weekend_holdings: {e}')
      exc = traceback.format_exc()
      logger.info(f'exception = {exc}')
      logger.info(f'*' * 80)
      stacks = ''.join(traceback.format_stack())
      logger.info(f'stacks = {stacks}')
      logger.info('Exception END')

    # TODO: check if all the tickers are present in the curr_weekend_holdings.
    # only cancel for those
    # if not, there must be some pending orders. cancel them
    DhanTasks.task_check_and_cancel_pending_orders(place_orders)
    gainers_list = dhan.read_gainers_list(max_rank=num_of_tickers * 2)
    remaining_trades_len = num_of_tickers - len(valid_existing_trades)
    logger.info(f'gainers_list = \n{gainers_list}')

    remaining_gainers = gainers_list.iloc[num_of_tickers:num_of_tickers +
                                          remaining_trades_len]
    logger.info(f'remaining_gainers = \n{remaining_gainers}')

    order_results = DhanTasks.task_open_positions_and_orders(
        num_of_tickers, notional, product_type, dry_run, place_orders,
        remaining_gainers)

    add_results_into_db(order_results, delay=15 if place_orders else 1)
    return order_results

  @staticmethod
  def task_enforce_position_cap_improved(
      num_of_tickers, notional, product_type, dry_run, place_orders):
    # call the function to update Db for trades
    dhan_ins = dhan.DhanTracker()
    p_filter = PortfolioFilters(dhan_ins=dhan_ins)

    curr_weekend_holdings = p_filter.get_curr_weekend_trades_summarized()

    if not curr_weekend_holdings.empty:
      logger.info(
          f'ENF: curr_weekend_holdings = {curr_weekend_holdings.to_string()}')
      logger.info(
          f'ENF: curr_weekend_holdings len = {curr_weekend_holdings.shape[0]}')
    else:
      logger.info(f'ENF: No holdings for today')

    # TODO: How to find out which ones are weekend trades?
    # valid_existing_trades = get_existing_valid_trades()
    if curr_weekend_holdings.shape[0] == num_of_tickers:
      logger.info(
          f'No need to enforce position cap. Its already at {num_of_tickers}')
      return
    else:
      logger.info(
          f'ENF: curr_weekend_holdings len = {curr_weekend_holdings.shape[0]}')

    # TODO: check if all the tickers are present in the curr_weekend_holdings.
    # only cancel for those
    # if not, there must be some pending orders. cancel them
    DhanTasks.task_check_and_cancel_pending_orders(place_orders)
    gainers_list = dhan.read_gainers_list(max_rank=num_of_tickers * 2)
    remaining_trades_len = num_of_tickers - curr_weekend_holdings.shape[0]
    logger.info(f'ENF: gainers_list = \n{gainers_list}')

    remaining_gainers = gainers_list.iloc[num_of_tickers:num_of_tickers +
                                          remaining_trades_len]
    logger.info(f'ENF: remaining_gainers = \n{remaining_gainers}')

    order_results = DhanTasks.task_open_positions_and_orders(
        num_of_tickers, notional, product_type, dry_run, place_orders,
        remaining_gainers)

    add_results_into_db(order_results, delay=15 if place_orders else 1)
    return order_results


class DhanYfinanceTickers:

  def __init__(self):
    pass

  @staticmethod
  def get_top_tickers(returns, num_of_tickers):
    top_N_tickers = DhanYfinanceTickers.get_top_tickers_by_returns(
        returns, num_of_tickers)
    logger.info(f'top_N_tickers = \n{top_N_tickers.to_string()}')
    return top_N_tickers

  @staticmethod
  def get_ticker_close_dict(df_ticker_with_closes):
    return dict(
        zip(df_ticker_with_closes['ticker'], df_ticker_with_closes['close']))

  @staticmethod
  def get_tickers_and_returns(num_of_tickers, dry_run):
    tickers = DhanYfinanceTickers.get_all_tickers(
        dryrun=dry_run, number_of_tickers=num_of_tickers)
    tickers_data = DhanYfinanceTickers.get_tickers_data(tickers)
    logger.info(f'get_all_tickers = {tickers}')
    returns = DhanYfinanceTickers.get_returns(tickers_data)
    logger.info(f'returns = \n{returns.to_json()}')
    return tickers, tickers_data, returns

  @staticmethod
  def get_all_tickers(
      add_yahoo_suffix=True,
      dryrun=True,
      number_of_tickers=3,
      file_path=NIFTY_CONSTITUENTS,
      ticker_column_name='Symbol'):
    """
    Get all tickers based on the specified markets and pairs file.

    Returns:
        list: A list of tickers.

    Raises:
        FileNotFoundError: If the pairs file does not exist.

    """
    tickers = []
    df = pd.read_csv(file_path)
    if add_yahoo_suffix:
      tickers = [f'{x}.NS' for x in df[ticker_column_name].to_list()]
    else:
      tickers = df[ticker_column_name].to_list()
    logger.info(f'df = {df}')
    logger.info(f'tickers = {tickers}')
    # return the last 3 tickers
    # tickers = tickers[-3:]
    if dryrun:
      return tickers[:number_of_tickers]
    else:
      return tickers

  @staticmethod
  def get_tickers_data(tickers):
    data = yf.download(tickers, period='5d', progress=False)
    return data

  @staticmethod
  def get_returns(tickers_data):
    # data = yf_download.get_historical_data_multiple(
    #     tickers, period='5d', save_to_csv=False, reduce_columns=False)
    adj_close = tickers_data['Close']
    logger.info(f'adj_close str = \n{adj_close.to_string()}')
    adj_close_last_row = adj_close.iloc[-1]

    logger.info(f'adj_close_last_row json = \n{adj_close_last_row.to_json()}')
    returns = adj_close.pct_change()
    return returns

  @staticmethod
  def get_pnl_view_from_closed_positions_or_holdings():
    dhan_ins = dhan.DhanTracker()

    holdings = dhan_ins.get_holdings()
    df_holdings = pd.DataFrame(holdings)
    logger.info(f'df_holdings = \n{df_holdings.to_string()}')
    positions = dhan_ins.get_positions()
    df_positions = pd.DataFrame(positions)
    logger.info(f'df_positions = \n{df_positions.to_string()}')

    # Merge df_positions with df_holdings on 'tradingSymbol'
    merged_df = pd.merge(
        df_holdings,
        df_positions[['tradingSymbol', 'sellAvg']],
        on='tradingSymbol',
        how='left')

    # Filter rows where 'sellAvg' is not null
    filtered_rows = merged_df[(merged_df['sellAvg'].notnull())
                              & (merged_df['sellAvg'] != 0)]
    filtered_rows['pnl'] = filtered_rows['totalQty'] * (
        filtered_rows['sellAvg'] - filtered_rows['avgCostPrice'])
    filtered_rows = filtered_rows[[
        'tradingSymbol', 'totalQty', 'avgCostPrice', 'sellAvg', 'pnl'
    ]]
    return filtered_rows

  @staticmethod
  def get_top_tickers_by_returns(returns, number_of_tickers):

    # Sort the returns for the desired date in descending order and get the top N tickers
    last_row = returns.iloc[-1]
    top_N_tickers = last_row.sort_values(
        ascending=False).head(number_of_tickers)
    logger.info(f'top_N_tickers = \n{top_N_tickers.to_string()}')
    return top_N_tickers

  @staticmethod
  def get_latest_price_data_from_yf(tickers, close_price_col):

    data = yf.download(tickers, period='1d', progress=False)
    # positions_for_today.to_pickle(
    #     os.path.join('data/dhan-samples', 'feb-02-positions.pickle'))
    # data.to_pickle(
    #     os.path.join('data/dhan-samples', 'feb-02-tickers-data.pickle'))
    logger.info(
        f'tickers latest data = \n{data[f"{close_price_col}"].to_string()}')

    # Create a dictionary mapping ticker symbols to their corresponding close prices
    if len(tickers) > 1:
      close_prices = data[close_price_col].iloc[-1].to_dict()
    else:
      close_prices = {tickers[0]: data[close_price_col].iloc[-1]}
    return close_prices

  @staticmethod
  def get_current_price_from_yf(
      df_assets,
      stop_loss_pct,
      tickers_col='tradingSymbol',
      close_price_col='High',
      cost_price_col='costPrice',
      price_curr='price_curr'):

    # Modify column names dynamically
    tickers = [t + '.NS' for t in df_assets[tickers_col].tolist()]
    close_prices = DhanYfinanceTickers.get_latest_price_data_from_yf(
        tickers, close_price_col)

    logger.info(f'tickers={tickers}; close_prices = {close_prices}')

    # Assign the correct price_curr based on the tickerSymbol
    df_assets[price_curr] = df_assets[tickers_col].map(
        lambda x: close_prices.get(x + '.NS'))

    # Calculate stop-loss price (price_sl) dynamically based on conditions
    df_assets['price_sl'] = np.where(
        df_assets[price_curr] < df_assets[cost_price_col],
        round(df_assets[cost_price_col] * (100 - stop_loss_pct) / 100, 2),
        np.where(
            df_assets[price_curr] > df_assets[cost_price_col] * 1.1,
            round(df_assets[price_curr] * (100 - 5) / 100, 2),
            np.where(
                df_assets[price_curr] > df_assets[cost_price_col] * 1.05,
                round(df_assets[price_curr] * (100 - 4) / 100, 2),
                round(
                    df_assets[cost_price_col] * (100 - stop_loss_pct) / 100,
                    2))))

    logger.info(f'df_assets = \n{df_assets.to_string()}')
    return df_assets

  @staticmethod
  def get_dhan_security_ids(tickers):
    dhan_ids = dict()
    df = pd.read_csv(dhan.DHAN_SCRIP_FILE)
    for ticker in tickers:
      ticker_df = df[
          (df[DhanScripColumns.SEM_TRADING_SYMBOL] == ticker)
          & (df[DhanScripColumns.SEM_INSTRUMENT_NAME] == 'EQUITY') &
          (df[DhanScripColumns.SEM_EXM_EXCH_ID] == 'NSE') & (
              ~df[DhanScripColumns.SEM_CUSTOM_SYMBOL].fillna('').astype(str).str
              .contains(
                  r'%|SEC\sRE|\sNCD|\sWarrant', flags=re.IGNORECASE,
                  regex=True))]

      if ticker_df.shape[0] > 1:
        logger.info(f'shape is distorted for : {ticker}')
        logger.info(f'shape= {ticker_df.shape}')
        logger.info(f'ticker: {ticker}; ticker_df=\n{ticker_df.to_string()}')
      elif ticker_df.shape[0] == 0:
        logger.info(f'NO DATA: shape is distorted for : {ticker}')
        logger.info(f'NO DATA: shape= {ticker_df.shape}')
        logger.info(f'NO DATA: ticker: {ticker}; ')
      else:
        dhan_ids[ticker] = DhanTickerInfo(
            ticker, ticker_df[DhanScripColumns.SEM_SMST_SECURITY_ID].iloc[0], 0)
    return dhan_ids

  @staticmethod
  def set_dhan_ticker_info(
      dhan_sec_ids, adj_close, tick_yahoo_name, notional_amount):
    dhan_ticker_info = dhan_sec_ids[tick_yahoo_name]
    dhan_ticker_info.price = adj_close[tick_yahoo_name]
    dhan_ticker_info.qty = math.floor(notional_amount / dhan_ticker_info.price)
    return dhan_ticker_info

  @staticmethod
  def set_closing_prices_and_quantities(
      dhan_sec_ids, adj_close, notional_amount):
    logger.info(f'adj_close=\n{adj_close}; \nadj_close  = \n{adj_close}')
    for ticker in dhan_sec_ids:
      tick_yahoo_name = f'{ticker}.NS'
      # tick_yahoo_name = f'{ticker}.NS'
      if ticker in adj_close:
        logger.info(f'found {ticker} in adjc_close')
        DhanYfinanceTickers.set_dhan_ticker_info(
            dhan_sec_ids, adj_close, ticker, notional_amount)
      if tick_yahoo_name in adj_close:
        logger.info(f'found {tick_yahoo_name} in adjc_close')
        DhanYfinanceTickers.set_dhan_ticker_info(
            dhan_sec_ids, adj_close, tick_yahoo_name, notional_amount)


class DhanDebugger:
  pass

  @staticmethod
  def show_open_orders(send_telegram_msg=False):
    # wait for 10 seconds
    dhan_ins = dhan.DhanTracker()
    orders = dhan_ins.get_orders(only_valid_orders=False)
    df = pd.DataFrame(orders)
    orders_s = dhan.show_df_in_table_format('orders', df)
    logger.info(f'show_df_in_table_format = {orders_s}')
    msg_text = f"Updated Orders: \n```{orders_s}```"
    if send_telegram_msg:
      telegram_runner.send_text([msg_text])
    else:
      logger.info(f'Not sending Telegram message: {msg_text}')
    return df


# TODO: Weekend: Maybe change this to check against DB
def get_current_valid_orders(
    dhan_tracker,
    after_market_order,
    status_to_check=[
        dhan.OrderStatusVals.CANCELLED, dhan.OrderStatusVals.REJECTED
    ],
    only_weekend_orders=True):

  curr_orders = dhan_tracker.get_orders()
  df_curr_orders = pd.DataFrame(curr_orders)
  # df_curr_orders = df_curr_orders[
  #     (df_curr_orders['afterMarketOrder'] == True)
  #     & ~(df_curr_orders['orderStatus'].isin(['CANCELLED', 'REJECTED']))]

  if not df_curr_orders.empty:
    logger.info(f'df_curr_orders = \n{df_curr_orders.to_string()}')
    df_curr_orders = df_curr_orders[
        (df_curr_orders['afterMarketOrder'] == after_market_order)
        & ~(df_curr_orders[dhan.Fields.OrderStatus].isin(status_to_check))]

    if only_weekend_orders:
      df_curr_orders = df_curr_orders[
          df_curr_orders['correlationId'].str.startswith('wknd')]

    logger.info(f'df_curr_orders filtered = \n{df_curr_orders.to_string()}')
    logger.info(
        f"sec ids from curr orders: {df_curr_orders['securityId'].to_list()}")
  else:
    logger.info(f'Empty df_curr_orders')
  return df_curr_orders


def get_current_stoploss_market_orders(
    dhan_tracker,
    status_to_check=[
        dhan.OrderStatusVals.CANCELLED, dhan.OrderStatusVals.REJECTED,
        dhan.OrderStatusVals.TRADED
    ]):

  curr_orders = dhan_tracker.get_orders()
  df_curr_orders = pd.DataFrame(curr_orders)
  logger.info(f'df_curr_orders = \n{df_curr_orders}')

  if not df_curr_orders.empty:
    df_curr_orders = df_curr_orders[
        (df_curr_orders['orderType'].isin(['STOP_LOSS_MARKET']))
        & ~(df_curr_orders[dhan.Fields.OrderStatus].isin(status_to_check))]

    logger.info(f'df_curr_orders filtered = \n{df_curr_orders}')
    logger.info(
        f"sec ids from curr orders: {df_curr_orders['securityId'].to_list()}")
  return df_curr_orders


def get_current_valid_positions(dhan_tracker, after_market_order):
  curr_positions = dhan_tracker.get_positions()
  df_curr_positions = pd.DataFrame(curr_positions)
  logger.info(f'df_curr_positions = \n{df_curr_positions}')

  if not df_curr_positions.empty:
    df_curr_positions = df_curr_positions[
        (df_curr_positions['netQty'] > 0)
        & (df_curr_positions['positionType'] == 'LONG')
        & (df_curr_positions['productType'] == 'CNC')]

    logger.info(
        f'df_curr_positions filtered = \n{df_curr_positions.to_string()}')
    logger.info(
        f"sec ids from curr positions: {df_curr_positions['securityId'].to_list()}"
    )
  else:
    logger.info(f'Empty df_curr_positions')
  return df_curr_positions


# TODO: Weekend: version of this is going to be that reads orders from yest and today
def is_new_ticker_already_in_holdings(ticker, holdings, dhan_ticker_info):
  df_holdings = pd.DataFrame(holdings)
  # Assuming df_holdings is your pandas DataFrame
  df_holdings['securityId'] = df_holdings['securityId'].astype(str)
  ticker_has_no_holding = df_holdings.empty or df_holdings[
      df_holdings['securityId'] == dhan_ticker_info.security_id].empty
  if not ticker_has_no_holding:
    logger.info(f'Holding exists for : {dhan_ticker_info}. Ignoring')
    # we should print out all such positions too
    ticker_existing_holdings = df_holdings[df_holdings['securityId'] == str(
        dhan_ticker_info.security_id
    )] if not df_holdings.empty else pd.DataFrame()
    logger.info(f'Existing Positions: \n{ticker_existing_holdings.to_string()}')
    return True
  return False


def is_new_position_valid_on_dhan(ticker, df_curr_positions, dhan_ticker_info):

  logger.info(f'ticker = {ticker}; type={type(ticker)}')
  ticker_has_no_position = df_curr_positions.empty or df_curr_positions[
      df_curr_positions['securityId'] == str(
          dhan_ticker_info.security_id)].empty
  if not ticker_has_no_position:
    logger.info(f'Position exists for : {dhan_ticker_info}. Ignoring')
    # we should print out all such positions too
    ticker_existing_positions = df_curr_positions[
        df_curr_positions['securityId'] == str(
            dhan_ticker_info.security_id
        )] if not df_curr_positions.empty else pd.DataFrame()
    logger.info(
        f'Existing Positions: \n{ticker_existing_positions.to_string()}')
    return False

  if not dhan_ticker_info.qty:
    logger.info(f'Position Qty is 0 for : {dhan_ticker_info}. Ignoring')
    return False
  return True


def get_existing_valid_trades():
  dhan_ins = dhan.DhanTracker()

  holdings = dhan_ins.get_holdings()
  positions = dhan_ins.get_positions()

  logger.info(f'holdings = \n{holdings}')
  logger.info(f'positions = \n{positions}')
  df_holdings = pd.DataFrame(holdings)
  df_positions = pd.DataFrame(positions)
  df = pd.DataFrame()

  if not df_holdings.empty and not df_positions.empty:
    df = pd.merge(df_holdings, df_positions, on='tradingSymbol', how='outer')
  else:
    if not df_holdings.empty:
      df = df_holdings
    elif not df_positions.empty:
      df = df_positions
  if df.empty:
    logger.info(f'no positions or holding. returning')
    return df

  # if df contains column availableQty
  if 'availableQty' not in df.columns:
    df['availableQty'] = 0
  if 'buyQty' not in df.columns:
    df['buyQty'] = 0
  if 'positionType' not in df.columns:
    df['positionType'] = np.nan

  df['availableQty'] = df['availableQty'].fillna(0)
  df['buyQty'] = df['buyQty'].fillna(0)
  valid_existing_trades = df[(df['availableQty'] > 0) | (
      (df['positionType'] == 'LONG') & (df['buyQty'] > 0))]
  logger.info(f'valid_existing_trades = \n{valid_existing_trades.to_string()}')

  return valid_existing_trades


class DhanActions:
  pass

  @staticmethod
  def _set_stop_loss_values(
      fixed_sl, holdings_for_today, stop_loss_pct, close_price_col):
    if fixed_sl:
      holdings_for_today['price_sl'] = round(
          holdings_for_today['avgCostPrice'] * (100 - stop_loss_pct) / 100, 2)
    else:
      holdings_for_today = DhanYfinanceTickers.get_current_price_from_yf(
          holdings_for_today,
          stop_loss_pct,
          tickers_col='tradingSymbol',
          close_price_col=close_price_col,
          cost_price_col='avgCostPrice',
          price_curr='price_curr')
    return holdings_for_today

  @staticmethod
  def _place_stop_loss_orders(
      holdings_for_today,
      place_orders,
      dhan_ins,
      after_market_order,
      product_type,
      tag='wknd-sl',
      order_type=dhanhq.SLM,
      transaction_type=dhanhq.SELL,
      qty_col='availableQty'):
    qty_col_backup = 'availableQty'
    all_orders = []
    for idx, row in holdings_for_today.iterrows():
      logger.info(f'row: {row}')
      ticker = row['tradingSymbol']
      qty = row[qty_col] if qty_col in row else row[qty_col_backup]

      price_sl = int(row['price_sl'] * 10) / 10  # only 1 decimal
      securityId = row['securityId']

      if place_orders:
        logger.info(
            f'Placing SL order for {ticker}; price_sl={price_sl}; qty={qty}')
        order_res = dhan_ins.place_order(
            securityId,
            qty,
            after_market_order=after_market_order,
            product_type=product_type,
            order_type=order_type,
            trigger_price=price_sl,
            transaction_type=transaction_type,
            tag=tag)
        all_orders.append(order_res)
        logger.info(f'order_res = {order_res}')
      else:
        logger.info(f'Not placing SL order for {ticker}; price_sl={price_sl}')

    return all_orders

  @staticmethod
  def open_positions(
      dhan_sec_ids,
      product_type=dhanhq.INTRA,
      after_market_order=False,
      place_orders=False,
      tag=None,
      send_telegram_msg=False):
    order_results = []
    date_yyyy_mm_dd = datetime.datetime.today().strftime(date_yyyy_mm_dd_FMT)
    dhan_tracker = dhan.DhanTracker()
    df_curr_orders = get_current_valid_orders(
        dhan_tracker, after_market_order, only_weekend_orders=True)
    full_update = [
        f'*Opening Positions* \nproduct_type= `{product_type}` \nplace_orders= `{place_orders}` '
    ]
    holdings = dhan_tracker.get_holdings()
    p_filter = PortfolioFilters(dhan_ins=dhan_tracker)
    try:
      curr_weekend_holdings = p_filter._get_current_weekend_holdings(
          fixed_sl=True, stop_loss_pct=None, close_price_col=None)
      if not curr_weekend_holdings.empty:
        # TODO : fix me later on, shouldnt have to call dhan_tracker.get_holdings()
        holdings = curr_weekend_holdings
      logger.info(
          f'curr_weekend_holdings = {curr_weekend_holdings.to_string()}')
    except Exception as e:
      logger.info(f'Exception in get_current_weekend_holdings: {e}')
      exc = traceback.format_exc()
      logger.info(f'exception = {exc}')
      logger.info(f'*' * 80)
      stacks = ''.join(traceback.format_stack())
      logger.info(f'stacks = {stacks}')
      logger.info('Exception END')

    for ticker in dhan_sec_ids:
      dhan_ticker_info = dhan_sec_ids[ticker]
      valid_order = dhan.is_new_order_valid_on_dhan(
          ticker, df_curr_orders, dhan_ticker_info)
      try:
        logger.info(f'holdings = {holdings}')
        # place 2
        existing_holding = is_new_ticker_already_in_holdings(
            ticker, holdings, dhan_ticker_info)
        logger.info(f'ticker: {ticker}; existing_holding={existing_holding}')
        if existing_holding:
          logger.info(
              f'Skipping placing order for : {ticker} as holding exists')
          full_update.append(
              f'Skipping placing order for : {ticker} as holding exists')
          continue
      except Exception as e:
        logger.info(f'Exception Start during posn check: {e}')
        exc = traceback.format_exc()
        logger.info(f'exception = {exc}')
        logger.info(f'*' * 80)
        stacks = ''.join(traceback.format_stack())
        logger.info(f'stacks = {stacks}')
        logger.info('Exception END')

      if not valid_order:
        full_update.append(f'{ticker} - Not a valid order for {ticker}')
        logger.info(f'Not a valid order for {dhan_ticker_info}')
        continue

      if place_orders:
        tag_suffix = f'{date_yyyy_mm_dd}_{dhan_ticker_info.security_id}'
        full_tag = f'{tag}_{tag_suffix}' if tag else tag_suffix
        logger.info(f'Placing order for {dhan_ticker_info}')
        order_res = dhan_tracker.place_order(
            str(dhan_ticker_info.security_id),
            dhan_ticker_info.qty,
            after_market_order=after_market_order,
            product_type=product_type,
            tag=full_tag)
        order_results.append(order_res)
        dhan.show_debug_info(
            order_res, 'order-results', save_to_file=False, save_to_db=False)
        logger.info(f'Order placement results: {order_res}')
        full_update.append(f'{ticker} - Order Placed: {order_res}')
      else:
        full_update.append(
            f'{ticker} - Order placement is disabled so skipping {ticker}')
        logger.info(f'Order placement is disabled. Skipping {dhan_ticker_info}')
    if send_telegram_msg:
      telegram_runner.send_text(full_update)
    else:
      logger.info(f'Not sending telegram msg: {full_update}')
    return order_results

  @staticmethod
  def cancel_pending_orders(pending_orders, place_orders=False):
    dhan_ins = dhan.DhanTracker()
    for index, row in pending_orders.iterrows():
      if place_orders:
        logger.info(f'About to cancel: {row}')
        dhan_ins.cancel_order(row['orderId'])
      else:
        logger.info(
            f'Not about to cancel: {row}; as place_orders={place_orders}')

  @staticmethod
  # TODO: Weekend: Needs to check against DB. If the list matches, then cancel
  def cancel_stoploss_market_orders(dhan_ins=None, dry_run=True):
    try:
      if not dhan_ins:
        dhan_ins = dhan.DhanTracker()
      sl_orders = get_current_stoploss_market_orders(dhan_ins)

      logger.info(f'sl_orders = \n{sl_orders.to_string()}')
      if sl_orders.empty:
        logger.info(f'Empty list of orders for SLM. Skipping cancellation')
        return
      for index, row in sl_orders.iterrows():
        if dry_run:
          logger.info(f'dry_run={dry_run}; Not calling cancel for: {row}')
        else:
          logger.info(f'About to cancel: {row}')
          res = dhan_ins.cancel_order(row['orderId'])
          logger.info(f'res = {res}')
    except Exception as e:
      logger.error(f'Exception in cancel_stoploss_market_orders: {e}')
      exc = traceback.format_exc()
      logger.info(f'exception = {exc}')
      logger.info(f'*' * 80)
      stacks = ''.join(traceback.format_stack())
      logger.info(f'stacks = {stacks}')
      logger.info('Exception END')

  @staticmethod
  def find_all_upper_circuit_pending_orders(df_orders):
    # we need to find all orders that
    if df_orders.empty:
      return df_orders
    pending_orders = df_orders[
        (df_orders['omsErrorDescription'] == 'MKT to Limit Conversion')
        & (df_orders['orderStatus'] == 'PENDING')]
    return pending_orders


class PortfolioFilters:

  def __init__(self, txn_helper=None, dhan_ins=None):
    self.txn_helper = txn_helper if txn_helper else TxnHelper()
    self.dhan_ins = dhan_ins if dhan_ins else dhan.DhanTracker()

  # This misses today's trads. Call _get_current_weekend_trades() instead
  def get_weekend_trades_from_txn_sort_by_symbol(self):
    logger.info(f'About to get weekend trades from txn db')
    weekend_trades_from_txn_db = self.txn_helper.get_weekend_trades_from_txn()
    weekend_trades_from_txn_db = weekend_trades_from_txn_db.sort_values(
        by='tradingSymbol', ascending=True)
    logger.info(
        f'weekend_trades_from_txn_db (ALL) = \n{weekend_trades_from_txn_db.to_string()}'
    )
    # weekend_trades = weekend_trades[weekend_trades['transactionType'] ==
    #                                 dhanhq.BUY]
    # logger.info(f'weekend_trades (BUY) = \n{weekend_trades.to_string()}')
    return weekend_trades_from_txn_db

  def _get_holdings_for_today(self):
    holdings = self.dhan_ins.get_holdings()
    logger.info(f'holdings = \n{holdings}')
    if len(holdings) == 0:
      return pd.DataFrame()
    df_holdings = pd.DataFrame(holdings)
    logger.info(f'df_holdings = \n{df_holdings.to_string()}')
    holdings_for_today = df_holdings[df_holdings['availableQty'] > 0]
    return holdings_for_today

  # I expect a direct call to _get_current_weekend_trades_from_txn instead
  def _filter_valid_weekend_trades_from_txn(self, wknd_trades):
    if wknd_trades.empty:
      return wknd_trades
    logger.info(f'Inside _filter_valid_weekend_trades_from_txn')
    wknd_trades['outstanding_trade_qty'] = 0
    wknd_trades['qty_to_close'] = 0
    try:
      wknd_trades = wknd_trades.sort_values(by='exchangeTime')
      total_buy = total_sell = 0

      # Iterate through each holding
      for index, row in wknd_trades.iterrows():
        symbol = row['tradingSymbol']
        availableQty = row['tradedQuantity']
        total_buy = total_sell = 0

        # Filter weekend trades for the current symbol
        symbol_trades = wknd_trades[wknd_trades['tradingSymbol'] == symbol]
        logger.info(
            f'trades for symbol {symbol} = \n{symbol_trades.to_string()}')
        if not symbol_trades.empty:
          # TODO: if symbol_Trades is 0, then no trades for the symbol, assign outstanding as holding

          # Trim symbol_trades if first trade is SELL
          if symbol_trades.iloc[0]['transactionType'] == 'SELL':
            symbol_trades = symbol_trades.iloc[1:]
            logger.info(f'Going to filter off trades for {symbol}...')
            logger.info(
                f'symbol_trades <{symbol}> = \n{symbol_trades.to_string()}')
          # outstanding_qty is total buy  - total sell
          total_buy = symbol_trades[symbol_trades['transactionType'] ==
                                    'BUY']['tradedQuantity'].sum()
          total_sell = symbol_trades[symbol_trades['transactionType'] ==
                                     'SELL']['tradedQuantity'].sum()
          logger.info(f'<{symbol}>: total_buy = {total_buy}')
          logger.info(f'<{symbol}>: total_sell = {total_sell}')

        wknd_trades.at[index,
                       'outstanding_trade_qty'] = qty = total_buy - total_sell
        wknd_trades.at[index, 'qty_to_close'] = qty
      logger.info(
          f'tmp loop weeknd trades so far... \n{wknd_trades.to_string()}')
    except Exception as e:
      logger.info(f'Exception Start during filter_valid_weekend_trades: {e}')
      exc = traceback.format_exc()
      logger.info(f'exception = {exc}')
      logger.info(f'*' * 80)
      stacks = ''.join(traceback.format_stack())
      logger.info(f'stacks = {stacks}')
      logger.info('Exception END')
    return wknd_trades


# TODO: There's a problem with this - it iterates through holdings only, so if a trade has been newly added, it will miss it

  def filter_valid_weekend_holdings(self, holdings, weekend_trades):

    if holdings.empty:
      return holdings

    logger.info(f'Inside filter_valid_weekend_holdings')
    holdings['outstanding_trade_qty'] = 0
    holdings['qty_to_close'] = 0
    try:
      weekend_trades = weekend_trades.sort_values(by='createTime')
      total_buy = total_sell = 0

      # Iterate through each holding
      for index, row in holdings.iterrows():
        symbol = row['tradingSymbol']
        availableQty = row['availableQty']
        total_buy = total_sell = 0

        # Filter weekend trades for the current symbol
        symbol_trades = weekend_trades[weekend_trades['tradingSymbol'] ==
                                       symbol]
        logger.info(
            f'trades for symbol {symbol} = \n{symbol_trades.to_string()}')
        if not symbol_trades.empty:
          # TODO: if symbol_Trades is 0, then no trades for the symbol, assign outstanding as holding

          # Trim symbol_trades if first trade is SELL
          if symbol_trades.iloc[0]['transactionType'] == 'SELL':
            symbol_trades = symbol_trades.iloc[1:]

          # outstanding_qty is total buy  - total sell
          total_buy = symbol_trades[symbol_trades['transactionType'] ==
                                    'BUY']['tradedQuantity'].sum()
          total_sell = symbol_trades[symbol_trades['transactionType'] ==
                                     'SELL']['tradedQuantity'].sum()
          logger.info(f'<{symbol}>: total_buy = {total_buy}')
          logger.info(f'<{symbol}>: total_sell = {total_sell}')

        holdings.at[index,
                    'outstanding_trade_qty'] = qty = total_buy - total_sell
        holdings.at[index, 'qty_to_close'] = min(qty, availableQty)
        logger.info(f'holdings so far... \n{holdings.to_string()}')
    except Exception as e:
      logger.info(f'Exception Start during filter_valid_weekend_holdings: {e}')
      exc = traceback.format_exc()
      logger.info(f'exception = {exc}')
      logger.info(f'*' * 80)
      stacks = ''.join(traceback.format_stack())
      logger.info(f'stacks = {stacks}')
      logger.info('Exception END')

    logger.info(
        f'filter_valid_weekend_holdings: holdings = \n{holdings.to_string()}')

    # ONly need to return the holdings where qty_to_close is more than 0
    return holdings[holdings['qty_to_close'] > 0].reset_index(drop=True)
    # return holdings

  def _get_current_weekend_trades_from_txn(self):
    logger.info(f'About to get current weekend trades from txn db')
    weekend_trades_from_txn = self.txn_helper.get_weekend_trades_from_txn()
    weekend_trades_from_txn = weekend_trades_from_txn.sort_values(
        by='tradingSymbol', ascending=True)
    logger.info(
        f'weekend_trades_from_txn = \n{weekend_trades_from_txn.to_string()}')

    curr_weekend_trades_from_txn = self._filter_valid_weekend_trades_from_txn(
        weekend_trades_from_txn)
    # sort curr_weekend_trades by tradingSymbol
    curr_weekend_trades_from_txn = curr_weekend_trades_from_txn.sort_values(
        by='tradingSymbol')

    logger.info(
        f'curr_weekend_trades_from_txn = \n{curr_weekend_trades_from_txn.to_string()}'
    )
    return curr_weekend_trades_from_txn

  def get_curr_weekend_trades_summarized(self):
    trade_book = self.dhan_ins.get_trade_book()
    dhan.show_debug_info(
        trade_book,
        'trades',
        save_to_file=False,
        save_to_db=True,
        tag=DEFAULT_TAG)

    df = self._get_current_weekend_trades_from_txn()
    logger.info(f'df=\n{df.to_string()}')
    df = df[df['transactionType'] == 'BUY']
    data_normalizer = dhan.NormalizeDataFrame(df)
    normalized_trades = data_normalizer.normalize_trades_df_by_ticker()
    normalized_trades = normalized_trades[
        normalized_trades['outstanding_trade_qty'] > 0]
    logger.info(f'normalized_trades = \n{normalized_trades.to_string()}')
    return normalized_trades

  def _get_current_weekend_holdings(
      self, fixed_sl, stop_loss_pct, close_price_col, get_current=False):
    curr_weekend_holdings = pd.DataFrame()
    try:
      if get_current:
        logger.info(f'Getting Current trades from txn')
        weekend_trades = self._get_current_weekend_trades_from_txn()
      else:
        logger.info(f'Getting trades from _get_weekend_trades')
        weekend_trades = self.get_weekend_trades_from_txn_sort_by_symbol()

      logger.info(f'weekend_trades (holdings) = \n{weekend_trades.to_string()}')
      logger.info(f'get_current = {get_current}')
      holdings_for_today = self._get_holdings_for_today()
      logger.info(f'holdings_for_today = \n{holdings_for_today.to_string()}')

      curr_weekend_holdings = self.filter_valid_weekend_holdings(
          holdings_for_today, weekend_trades)
      if curr_weekend_holdings.empty:
        logger.info(f'Empty curr_weekend_holdings')
      else:
        if stop_loss_pct:
          curr_weekend_holdings = DhanActions._set_stop_loss_values(
              fixed_sl, curr_weekend_holdings, stop_loss_pct, close_price_col)
          logger.info(
              f'curr_weekend_holdings = \n{curr_weekend_holdings.to_string()}')
        else:
          logger.info(
              f'curr_weekend_holdings (w.o stop_loss_pct)= \n{curr_weekend_holdings.to_string()}'
          )
    except Exception as e:
      logger.info(f'Exception Start during _get_current_weekend_holdings: {e}')
      exc = traceback.format_exc()
      logger.info(f'exception = {exc}')
      logger.info(f'*' * 80)
      stacks = ''.join(traceback.format_stack())
      logger.info(f'stacks = {stacks}')
      logger.info('Exception END')
    return curr_weekend_holdings


def write_gainers_list(top_N_tickers, tickers_data, send_telegram_msg=False):
  df = pd.DataFrame(columns=['ticker', 'close'])
  ticker_names = top_N_tickers.index.to_list()
  logger.info(f'ticker names to write = {ticker_names}')
  df['ticker'] = [t[:-3] for t in ticker_names]
  close_prices = ticker_names.copy()
  # df['close'] = ticker_names
  for ticker in top_N_tickers.index:

    close_price = tickers_data['Close'][ticker].iloc[-1]
    logger.info(f'FINDING: {ticker}; close_price = {close_price}')
    # find the index of ticker in close_prices and then assign close price in that index
    close_prices[top_N_tickers.index.get_loc(ticker)] = close_price

  df['close'] = close_prices

  filepath = dhan.gainers_expected_file_name()
  os.makedirs(dhan.GAINERS_DIR, exist_ok=True)
  df.to_csv(filepath, index=False)
  list_gainers = df.to_string(index=False).replace('.', r'\.')
  full_update = [f'Gainers List for today - \n`{list_gainers}`']

  if send_telegram_msg:
    telegram_runner.send_text(full_update)
  else:
    logger.info(f'Not sending Telegram message: {full_update}')
  return df


def add_results_into_db(order_results, delay=10):
  try:
    if delay:
      logger.info(f'Sleeping for {delay} seconds')
      time.sleep(delay)

    # {"orderId": "552312018866", "orderStatus": "TRANSIT"}
    orders = []
    dhan_ins = dhan.DhanTracker()
    for results in order_results:
      logger.info(f'results = {results}')
      order = dhan_ins.get_order_by_id(results['orderId'])
      if order:
        orders.append(order)
        logger.info(f'get_order_by_id = {order}')
    date_today = datetime.datetime.today().strftime('%Y-%m-%d')
    txn_mgr = txn_weekend.TxnManager()
    logger.info(f'Len of orders = {len(orders)}')

    for order in orders:
      logger.info(f'Writing into DB for {order}')
      txn_mgr.write_transaction(
          DEFAULT_TAG, DHAN_TXN_TYPE.ORDERS,
          order[DHAN_INFO_COLS[DHAN_TXN_TYPE.ORDERS][0]],
          order[DHAN_INFO_COLS[DHAN_TXN_TYPE.ORDERS][1]] or date_today, order)
  except Exception as e:
    logger.info(f'Exception in adding results into db: {e}')
    exc = traceback.format_exc()
    logger.info(f'exception = {exc}')
    logger.info(f'*' * 80)
    stacks = ''.join(traceback.format_stack())
    logger.info(f'stacks = {stacks}')
    logger.info('Exception END')


if __name__ == '__main__':
  p_filter = PortfolioFilters()
  normalized_trades = p_filter.get_curr_weekend_trades_summarized()
  logger.info(f'normalized_trades = \n{normalized_trades.to_string()}')

  # main()
