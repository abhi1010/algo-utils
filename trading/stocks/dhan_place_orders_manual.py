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

logger = utils.get_logger('dhan-place-orders-manual', use_rich=True)

from trading.common.util_prints import *
from trading.common.util_dates import *
from trading.common import telegram_helper
from trading.portfolio.tracker import *
from trading.configs import txn_weekend
# from trading.services import telegram_runner
from trading.stocks.dhan_common import *
from trading.stocks import dhan
'''
Usage:

  python trading/stocks/dhan_place_orders_manual.py \
    --tickers ICICIBANK.NS,SBIN.NS \
    --max-price 6000 \
    --amount 4000

  python trading/stocks/dhan_place_orders_manual.py  --tickers "LLOY,HEMISPHERE PROPERTIES" --max-price 7000  --amount 2500 --place-orders
  python trading/stocks/dhan_place_orders_manual.py  --tickers "LLOY,HEMISPHERE PROPERTIES" --max-price 7000  --amount 2500 --place-orders

  python trading/stocks/dhan_place_orders_manual.py  --tickers data/tickers-to-buy.txt --max-price 7000  --amount 2500 --place-orders
'''

CURR_INR = 'Rs.'


def get_dhan_id_from_ticker(ticker, dhan_info, dhan_scrips):
  if ticker in dhan_scrips:
    return ticker, dhan_scrips[ticker]

  matching_dhan_ticker = list()
  for info_dict in dhan_info:
    if ticker.lower() in info_dict['description'].lower():
      matching_dhan_ticker.append(info_dict)

  if len(matching_dhan_ticker) == 1:
    return matching_dhan_ticker[0]['symbol'], matching_dhan_ticker[0]['id']
  elif len(matching_dhan_ticker) > 1:
    logger.error(
        f'Found more than 1 matching dhan ticker: {matching_dhan_ticker}'
        f'; ticker to search: {ticker}')
    sys.exit(1)
  return None, None


def process_tickers(tickers, dhan_info, dhan_scrips, amount, max_price):
  results = []
  for ticker in tickers:
    try:
      ticker_short_name, dhan_id = get_dhan_id_from_ticker(
          ticker, dhan_info, dhan_scrips)
      logger.info(
          f'ticker: {ticker}; ticker_short_name={ticker_short_name}; dhan_id = {dhan_id}'
      )
      df_stock = yf.download(f'{ticker_short_name}.NS',
                             period='1d',
                             progress=False,
                             multi_level_index=False)
      #multi_level_index=False)
      logger.info(f'df_stock = \n{df_stock.to_string()}')
      current_price = df_stock['Close'].iloc[-1]
      # stock = yf.Ticker(ticker)
      # logger.info(f'stock.info = \n{stock.info}')
      # current_price = stock.info['regularMarketPrice']

      if current_price <= max_price:
        if current_price <= amount:
          quantity = int(amount // current_price)
        else:
          quantity = 1

        results.append({
            'dhan_id': dhan_id,
            'ticker': ticker_short_name,
            'price': current_price,
            'quantity': quantity,
            'total_cost': quantity * current_price
        })
      else:
        logger.warning(
            f'Skipping {ticker_short_name}: Price ({CURR_INR} {current_price}) '
            f'exceeds max price ({CURR_INR} {max_price})')
    except Exception as e:
      logger.error(f"Error processing {ticker}: {str(e)}")

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
                      action='store_true',
                      help='Should we place orders')

  # add an argument for exchange, default being NSE
  parser.add_argument('--exchange',
                      type=str,
                      default=dhanhq.NSE,
                      help='Exchange',
                      choices=[dhanhq.NSE, dhanhq.BSE])
  # add an arg for after market order, default being true, which means AMO is true, and false means AMO is false
  parser.add_argument('--no-amo', action='store_true', help='AMO')

  return parser.parse_args()


'''
This code snippet defines a main function that serves as the entry point of the program.
It first calls the create_args function to parse command-line arguments and store them in the args variable.
It then logs the value of args using the logger.info function.

Next, it calls the get_tickers_from_cli function to retrieve tickers from the command-line arguments or a file.
The retrieved tickers are stored in the tickers variable.

Then, it calls the get_dhan_scrips_as_dict_sym_as_key function from the dhan module to get a dictionary of Dhan scrips,
replacing any _EQ in the exchange_to_use argument. The resulting dictionary is stored in the dhan_scrips variable.

After that, it calls the get_dhan_scrips_as_list_with_info function from the dhan module
to get a list of Dhan scrips with additional information, also replacing any _EQ in the exchange_to_use argument.
The resulting list is stored in the dhan_info variable.

The code then logs the value of dhan_scrips using the logger.info function.

Next, it calls the process_tickers function with the tickers, dhan_info, dhan_scrips, args.amount,
and args.max_price arguments, and stores the results in the results variable.

Then, it calls the create_amo_orders function with the results, args.place_orders, args.exchange,
and not args.no_amo arguments.

Finally, it calls the show_results function with the results argument.

Overall, this code snippet retrieves and processes tickers,
gets Dhan scrips and information, processes the tickers based on the retrieved information,
creates AMO orders, and displays the results.
'''


def main():
  args = create_args()
  logger.info(f'args = {args}')

  tickers = get_tickers_from_cli(args)
  dhan_scrips = dhan.get_dhan_scrips_as_dict_sym_as_key(
      exchange_to_use=args.exchange.replace('_EQ', ''))
  dhan_info = dhan.get_dhan_scrips_as_list_with_info(
      exchange_to_use=args.exchange.replace('_EQ', ''))
  logger.info(f'dhan_scrips = {dhan_scrips}')
  logger.info(f'dhan_info = {dhan_info}')
  results = process_tickers(tickers, dhan_info, dhan_scrips, args.amount,
                            args.max_price)
  create_amo_orders(results,
                    place_orders=args.place_orders,
                    exchange_segment=args.exchange,
                    after_market_order=not args.no_amo)
  show_results(results)


def create_amo_orders(results,
                      place_orders=False,
                      exchange_segment=dhanhq.NSE,
                      transaction_type=dhanhq.BUY,
                      order_type=dhanhq.MARKET,
                      product_type=dhanhq.CNC,
                      after_market_order=True):
  dhan_tracker = dhan.DhanTracker()
  today = date.today()

  # strftime to format date as YYMMDD
  date_str = today.strftime("%y%m%d")
  tag_prefix = f'TA-{date_str}'

  for result in results:
    dhan_id = result['dhan_id']
    if not dhan_id or not place_orders:
      if not dhan_id:
        logger.warning(f'Cannot create order for ticker: {result["ticker"]}')
      if not place_orders:
        logger.info(f'Not placing order for: {result["ticker"]}; ')

      logger.info(f'  Debug Info: qty: {result["quantity"]}; '
                  f'dhan_id: {dhan_id}; exch: {exchange_segment}; '
                  f'txn type: {transaction_type}; '
                  f'order type: {order_type}; '
                  f'AMO: {after_market_order}')
      continue

    order_res = dhan_tracker.place_order(dhan_id,
                                         qty=result['quantity'],
                                         exchange_segment=exchange_segment,
                                         transaction_type=transaction_type,
                                         order_type=order_type,
                                         product_type=product_type,
                                         after_market_order=after_market_order,
                                         tag=f'{tag_prefix}-{dhan_id}')
    logger.info(f'order_res = {order_res}')


def show_results(results):
  for result in results:
    logger.info(f"Ticker: {result['ticker']}; ID: {result['dhan_id']}")
    logger.info(f"Price: ${result['price']:.2f}")
    logger.info(f"Quantity: {result['quantity']}")
    logger.info(f"Total Cost: {CURR_INR} {result['total_cost']:.2f}")
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
