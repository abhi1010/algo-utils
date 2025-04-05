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

logger = utils.get_logger('tasty-place-orders-manual-sell', use_rich=True)

from trading.brokers import tasty as tt
from trading.common.util_prints import *
from trading.common.util_dates import *
from trading.common import telegram_helper
from trading.portfolio.tracker import *
from trading.configs import txn_weekend

from tastytrade import order as TT_Order
# from trading.services import telegram_runner
'''
Usage:

  python trading/stocks/tasty_place_orders_manual_sell.py \
    --tickers ABBV,ANRO \
    --max-price 100 \
    --amount 40

  python trading/stocks/tasty_place_orders_manual_sell.py --tickers ABBV,ALAB --max-price 100 --amount 40

  python trading/stocks/tasty_place_orders_manual_sell.py  --tickers data/tasty-tickers-close.txt --place-orders
  python trading/stocks/tasty_place_orders_manual_sell.py --tickers data/tasty-tickers-close.txt

'''

CURR_USD = '$'


def remove_tickers_with_no_position(tickers):

  results = []

  tasty_utils = tt.TastyUtilities()
  # positions_without_sl = tasty_utils.get_equity_positions_without_sl()
  positions = tasty_utils.tasty_bite.get_positions()

  def get_position(ticker):
    for position in positions:
      if position.symbol == ticker and position.instrument_type == TT_Order.InstrumentType.EQUITY:
        return position
    return None

  for ticker in tickers:
    try:
      ticker_position = get_position(ticker)
      if not ticker_position:
        logger.warning(f'ticker: {ticker}; No position for {ticker}')
        continue
      logger.info(f'ticker: {ticker}; position = {ticker_position}')
      df_stock = yf.download(f'{ticker}',
                             period='1d',
                             progress=False,
                             multi_level_index=False)
      logger.info(f'ticker: {ticker}; df_stock = \n{df_stock.to_string()}')
      last_close_price = df_stock['Close'].iloc[-1][ticker]
      quantity = ticker_position.quantity
      # if quantity is integer, then it is a share
      # can_close = quantity == math.floor(quantity)
      logger.info(f'ticker: {ticker} -- > price=$$\n{last_close_price};$$'
                  f'qty = {quantity};')

      # if can_close:
      results.append({
          'ticker': ticker,
          'price': last_close_price,
          'quantity': quantity
      })
      # stock = yf.Ticker(ticker)
      # logger.info(f'stock.info = \n{stock.info}')
      # current_price = stock.info['regularMarketPrice']
    except Exception as e:
      logger.error(f"ticker: {ticker}; Error processing {ticker}: {str(e)}")

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


def main():
  args = create_args()
  logger.info(f'args = {args}')

  tickers = get_tickers_from_cli(args)
  logger.info(f'tickers= {tickers}')

  results = remove_tickers_with_no_position(tickers)
  logger.info(f'processed tickers: {results}')

  responses = create_sell_orders(results,
                                 place_orders=args.place_orders,
                                 place_market_orders=args.place_market_orders)
  show_results(responses)


def _does_ticker_already_have_open_order(pending_orders_df, ticker):

  # Symbol should be same as ticker and Type should be Limit Or Market

  matching_orders = pending_orders_df[
      (pending_orders_df['Symbol'] == ticker)
      & ((pending_orders_df['Type'] == TT_Order.OrderType.MARKET)
         | (pending_orders_df['Type'] == TT_Order.OrderType.LIMIT))]
  logger.info(f'ticker = {ticker}; matching_orders=\n{matching_orders}')
  return not matching_orders.empty


def get_open_pending_orders(tasty_utils):
  orders = tasty_utils.get_pending_open_orders_only_equities()
  logger.info(f'orders = {orders}')
  df = tasty_utils.create_orders_dataframe(orders)
  logger.info(f'orders in df = \n{df.to_string()}')
  return df


def create_sell_orders(ticker_results,
                       place_orders=False,
                       place_market_orders=False):
  responses = dict()

  tasty_utils = tt.TastyUtilities()

  pending_orders = get_open_pending_orders(tasty_utils)
  logger.info(f'pending_orders = \n{pending_orders}')

  today = date.today()

  # strftime to format date as YYMMDD
  date_str = today.strftime("%y%m%d")

  for result in ticker_results:
    ticker = result['ticker']
    price = result['price']
    qty = result['quantity']

    has_existing_order = _does_ticker_already_have_open_order(
        pending_orders, ticker)
    # if has_existing_order:
    #   logger.info(f'{ticker} --> Found a matching order. IGNORED')
    #   continue

    is_fraction_quantity = qty % 1 != 0
    if is_fraction_quantity:
      if not place_market_orders:
        logger.warning(
            f'{ticker} --> Not placing market sell for fractional qty'
            f'= {qty}')
        # dry-run in actual call will handle this.
      order_res = tasty_utils.place_order(
          ticker,
          None,
          qty,
          action=TT_Order.OrderAction.SELL_TO_CLOSE,
          time_in_force=TT_Order.OrderTimeInForce.DAY,
          order_type=TT_Order.OrderType.MARKET,
          price_effect=TT_Order.PriceEffect.CREDIT,
          dry_run=not place_market_orders)
    else:
      if not place_orders:
        logger.warning(f'{ticker} --> Not placing market sell for any qty'
                       f'= {qty}')
        # dry-run in actual call will handle this.

      order_res = tasty_utils.place_order(
          ticker,
          price,
          qty,
          action=TT_Order.OrderAction.SELL_TO_CLOSE,
          time_in_force=TT_Order.OrderTimeInForce.GTC,
          order_type=TT_Order.OrderType.LIMIT,
          price_effect=TT_Order.PriceEffect.CREDIT,
          dry_run=not place_orders)

    logger.info(f'Debug Info <{ticker}> --> qty: {qty}; '
                f'Fractional: {is_fraction_quantity}')

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
