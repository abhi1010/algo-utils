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
from tastytrade import order as TT_Order
'''
Usage:

  python trading/stocks/tasty_place_orders_manual_buy.py \
    --tickers ABBV,ANRO \
    --max-price 100 \
    --amount 40

  python trading/stocks/tasty_place_orders_manual_buy.py --tickers ABBV,ALAB --max-price 100 --amount 40

  python trading/stocks/tasty_place_orders_manual_buy.py  --tickers data/tasty-tickers-open.txt --max-price 100 --amount 40 --place-orders
  python trading/stocks/tasty_place_orders_manual_buy.py --tickers data/tasty-tickers-open.txt --max-price 120 --amount 80 --place-orders --place-market-orders

'''

CURR_USD = '$'


def process_tickers(tickers, amount, max_price):
  results = []
  for ticker_input in tickers:
    try:
      # Split ticker and quantity if present
      if ',' in ticker_input:
        ticker, quantity = ticker_input.split(',')
        quantity = float(quantity)
        manual_quantity = True
      else:
        ticker = ticker_input
        manual_quantity = False

      # multi_level_index=False
      df_stock = yf.download(f'{ticker}',
                             period='1d',
                             progress=False,
                             multi_level_index=False)
      logger.info(f'df_stock = \n{df_stock.to_string()}')
      current_price = df_stock['Close'].iloc[-1]

      if not manual_quantity:
        if current_price <= max_price:
          if current_price <= amount:
            quantity = int(amount // current_price)
          else:
            quantity = 1
        else:
          quantity = round(amount / current_price, 2)

      results.append({
          'ticker': ticker,
          'price': current_price,
          'quantity': quantity,
          'total_cost': quantity * current_price
      })
    except Exception as e:
      logger.error(f"Error processing {ticker_input}: {str(e)}")
  logger.info(f'processed tickers: {results}')
  return results


def read_tickers_from_file(filename):
  with open(filename, 'r') as file:
    return [line.strip() for line in file if line.strip()]


def create_args():
  parser = argparse.ArgumentParser(
      description="Process stock tickers with given parameters.")
  parser.add_argument(
      "--tickers",
      help=
      "Comma-separated list of tickers or path to a file containing tickers")
  parser.add_argument("--amount",
                      type=float,
                      help="Amount willing to spend per stock")
  parser.add_argument("--max-price",
                      type=float,
                      help="Maximum price per share")

  # add an argument for place-orders, which is disabled by default
  parser.add_argument('--place-orders',
                      '--place-order',
                      '-po',
                      action='store_true',
                      help='Should we place orders')

  parser.add_argument(
      '--place-market-orders',
      '--place-market-order',
      '-pmo',
      action='store_true',
      help='Should we place market orders for fractional shares')

  # add an arg for after market order, default being true, which means AMO is true, and false means AMO is false
  parser.add_argument('--no-amo', action='store_true', help='AMO')

  return parser.parse_args()


def get_open_pending_orders(tasty_utils):
  orders = tasty_utils.get_pending_open_orders_only_equities()
  logger.info(f'orders = {orders}')
  df = tasty_utils.create_orders_dataframe(orders)
  logger.info(f'orders in df = \n{df.to_string()}')
  return df


def main():
  args = create_args()
  logger.info(f'args = {args}')

  tasty_utils = tt.TastyUtilities()
  tickers = get_tickers_from_cli(args)
  logger.info(f'tickers= {tickers}')

  results = process_tickers(tickers, args.amount, args.max_price)
  logger.info(f'processed tickers: {results}')

  open_positions = tasty_utils.get_positions_as_df()
  logger.info(f'open_positions = \n{open_positions}')

  pending_orders = get_open_pending_orders(tasty_utils)
  logger.info(f'pending_orders = \n{pending_orders}')

  responses = create_buy_orders(tasty_utils,
                                results,
                                pending_orders=pending_orders,
                                open_positions=open_positions,
                                place_orders=args.place_orders,
                                place_market_orders=args.place_market_orders)

  _rename_tasty_tickers_file(args)

  show_results(responses)


def get_today_date():
  today = date.today()
  return today.strftime("%y%m%d")  # Format date as YYMMDD


def _rename_tasty_tickers_file(args):
  # we are goign to rename the tasty tickers cli file,
  # if it is being used and place_market_orders is enabled
  if (args.tickers.endswith('.txt') or args.tickers.endswith('.csv')):
    if not args.place_market_orders:
      logger.info(
          f'Not renaming file {args.tickers} as place_market_orders is not enabled'
      )
      return
    # new file name should include today's date so that i can find it easily
    new_file_name = f'{args.tickers.split(".")[0]}-{get_today_date()}.txt'
    new_file_name = new_file_name.replace('data/',
                                          'data/tasty-tickers-orders/')
    logger.info(f'Renaming file {args.tickers} to {new_file_name}')
    os.rename(args.tickers, new_file_name)


def _does_ticker_already_have_open_order(pending_orders_df, ticker):

  # Symbol should be same as ticker and Type should be Limit Or Market
  if pending_orders_df.empty:
    return False
  matching_orders = pending_orders_df[
      (pending_orders_df['Symbol'] == ticker)
      & ((pending_orders_df['Type'] == TT_Order.OrderType.MARKET)
         | (pending_orders_df['Type'] == TT_Order.OrderType.LIMIT))]
  logger.info(f'ticker = {ticker}; matching_orders=\n{matching_orders}')
  return not matching_orders.empty


def _does_ticker_already_have_open_position(open_positions, ticker):
  # Symbol should be same as ticker and Type should be Limit Or Market

  if open_positions.empty:
    return False
  matching_positions = open_positions[
      (open_positions['Symbol'] == ticker)
      & (open_positions['InstrumentType'] == TT_Order.InstrumentType.EQUITY)]
  logger.info(f'ticker = {ticker}; matching_positions=\n{matching_positions}')
  return not matching_positions.empty


def _check_existing_ticker_status(ticker, pending_orders, open_positions):
  """
    Check if a ticker has existing orders or positions.
    Returns a tuple of (has_order, has_position)
    """
  has_order = _does_ticker_already_have_open_order(pending_orders, ticker)
  has_position = _does_ticker_already_have_open_position(
      open_positions, ticker)

  if has_order:
    logger.info(f'{ticker} --> Found a matching order. IGNORED ')
  if has_position:
    logger.info(f'{ticker} --> Found a matching position. IGNORED ')

  return has_order, has_position


def _create_market_order(tasty_utils, ticker, qty, place_market_orders):
  """
    Create a market order for fractional quantities
    """
  if not place_market_orders:
    logger.info(f'Skipping market order for {ticker}')
    return None

  return tasty_utils.place_order(ticker,
                                 None,
                                 qty,
                                 action=TT_Order.OrderAction.BUY_TO_OPEN,
                                 time_in_force=TT_Order.OrderTimeInForce.DAY,
                                 order_type=TT_Order.OrderType.MARKET,
                                 price_effect=TT_Order.PriceEffect.DEBIT,
                                 dry_run=not place_market_orders)


def _create_limit_order(tasty_utils, ticker, price, qty, place_orders):
  """
    Create a limit order for whole number quantities
    """
  if not place_orders:
    logger.info(f'Skipping limit order for {ticker}')
    return None

  return tasty_utils.place_order(
      ticker,
      price,  # used to be -price
      qty,
      action=TT_Order.OrderAction.BUY_TO_OPEN,
      time_in_force=TT_Order.OrderTimeInForce.GTC,
      order_type=TT_Order.OrderType.LIMIT,
      price_effect=TT_Order.PriceEffect.DEBIT,
      dry_run=not place_orders)


def _place_order(tasty_utils, ticker, price, qty, place_orders,
                 place_market_orders):
  """
    Place either a market or limit order based on quantity type
    """
  is_fraction_quantity = qty % 1 != 0

  logger.info(
      f'  Debug Info <{ticker}> --> qty: {qty}; place_orders={place_orders}; '
      f'place_market_orders={place_market_orders}')

  if is_fraction_quantity:
    return _create_market_order(tasty_utils, ticker, qty, place_market_orders)
  else:
    return _create_limit_order(tasty_utils, ticker, price, qty, place_orders)


def create_buy_orders(tasty_utils,
                      tickers,
                      pending_orders,
                      open_positions,
                      place_orders=False,
                      place_market_orders=False):
  """
    Create buy orders for a list of tickers based on various conditions
    """
  responses = {}
  today = date.today()
  date_str = today.strftime("%y%m%d")  # Format date as YYMMDD

  for result in tickers:
    ticker = result['ticker']
    price = -result['price']
    qty = result['quantity']

    has_order, has_position = _check_existing_ticker_status(
        ticker, pending_orders, open_positions)

    if has_order:
      logger.info(f'{ticker} --> Found a matching order. IGNORED ')
      continue
    if has_position:
      continue

    order_res = _place_order(tasty_utils, ticker, price, qty, place_orders,
                             place_market_orders)
    responses[ticker] = order_res
    logger.info(f'order_res = {order_res}')

  return responses


def show_results(results):
  for ticker, response in results.items():
    logger.info(f"Ticker: {ticker}; Response: {response}")
    logger.info("---")


def get_tickers_from_cli(args):
  # Check if tickers is a file or a list
  if ',' in args.tickers:
    tickers = [ticker.strip() for ticker in args.tickers.split(',')]
  elif args.tickers.endswith('.txt') or args.tickers.endswith('.csv'):
    tickers = read_tickers_from_file(args.tickers)
  else:
    print("Error: Tickers must be a comma-separated list or a .txt file")
    sys.exit(1)
  return tickers


if __name__ == "__main__":
  main()
