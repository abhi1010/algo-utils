# -*- coding: utf-8 -*-
import yfinance as yf
import argparse
import sys
from datetime import timedelta, date
import math
import re
import pickle
import traceback

from trading.common import utils

logger = utils.get_logger('tasty-place-orders-manual-buy', use_rich=True)

from trading.brokers import tasty as tt
from trading.common.util_prints import *
from trading.common.util_dates import *
from trading.common import telegram_helper
from trading.portfolio.tracker import *
from trading.configs import txn_weekend
# from trading.services import telegram_runner
from tastytrade import order as TT_Order, InstrumentType
'''
Usage:

  python trading/stocks/tasty_place_orders_stop_loss.py \
    --tickers ABBV,ANRO

  python trading/stocks/tasty_place_orders_stop_loss.py --tickers ABB,ALAB
  python trading/stocks/tasty_place_orders_stop_loss.py --tickers data/tasty-tickers-open.txt  --place-orders

'''

CURR_USD = '$'


def read_tickers_from_file(filename):
  with open(filename, 'r') as file:
    return [line.strip() for line in file if line.strip()]


def create_args():
  parser = argparse.ArgumentParser(
      description="Process stock tickers with given parameters.")
  # add an argument for place-orders, which is disabled by default
  parser.add_argument(
      '--place-orders', action='store_true', help='Should we place orders')

  # add arg called stoploss_threshold
  parser.add_argument(
      '--stoploss-threshold',
      type=float,
      default=0.5,
      help='Stoploss threshold',
      required=False)
  return parser.parse_args()


def _does_ticker_already_have_open_stoploss_order(pending_orders_df, ticker):

  matching_orders = pending_orders_df[
      (pending_orders_df['Symbol'] == ticker)
      & (
          (pending_orders_df['Type'] == TT_Order.OrderType.STOP)
          | (pending_orders_df['Type'] == TT_Order.OrderType.STOP_LIMIT))]
  logger.info(f'ticker = {ticker}; matching_orders=\n{matching_orders}')
  return not matching_orders.empty


def get_open_pending_orders_df(tasty_utils):
  orders = tasty_utils.get_pending_open_orders_only_equities()
  logger.info(f'orders = {orders}')
  df = tasty_utils.create_orders_dataframe(orders)
  logger.info(f'orders in df = \n{df.to_string()}')
  return df


def _get_potential_stoploss_price(position, stop_loss_threshold):
  ticker = position.symbol

  # df_stock = yf.download(f'{ticker}', period='1d', progress=False)
  # last_closing_price = df_stock['Close'].iloc[-1]
  last_closing_price = float(position.close_price)
  position_open_price = float(position.average_open_price)
  higher_price = max(last_closing_price, position_open_price)
  sl_price = higher_price * (1 - stop_loss_threshold)
  logger.info(f'ticker = {ticker}; sl_price = {sl_price}')
  return sl_price
  # price to be higher of last closing price and open price


def main():
  args = create_args()
  logger.info(f'args = {args}')

  tasty_utils = tt.TastyUtilities()

  pending_orders_df = get_open_pending_orders_df(tasty_utils)
  logger.info(f'pending_orders_df = \n{pending_orders_df}')

  # tasty_utils.print_positions()
  positions_without_sl = tasty_utils.get_equity_positions_without_sl()

  for position in positions_without_sl:
    ticker = position.symbol
    has_existing_order = _does_ticker_already_have_open_stoploss_order(
        pending_orders_df, ticker)
    if has_existing_order:
      logger.info(f'{ticker} --> ABHI: Found a matching order. IGNORED')
      continue

    sl_price = _get_potential_stoploss_price(position, args.stoploss_threshold)
    logger.info(f'OK to create SL for {position}; sl_price = {sl_price}')

    tasty_utils.set_stop_loss_for_position(
        position, sl_price, dry_run=not args.place_orders)


if __name__ == "__main__":
  main()
