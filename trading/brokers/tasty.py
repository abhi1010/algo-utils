import os
import io
import re
import sys
import requests
import warnings
from datetime import datetime, timedelta
import decimal

warnings.filterwarnings('ignore')

from trading.common import utils

logger = utils.get_logger("tasty")

import pandas as pd
import numpy as np

from tastytrade import Session
from tastytrade import Account
from tastytrade.instruments import Equity
from tastytrade import Account, AlertStreamer
from tastytrade.streamer import AlertType
from tastytrade import Watchlist
from tastytrade.metrics import get_risk_free_rate
from tastytrade import order as TT_Order

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.style import Style
from enum import Enum
from zoneinfo import ZoneInfo

# load key and token from env variables
UESRNAME = os.getenv('TASTY_USERNAME')
KEY = os.getenv('TASTY_KEY')
TOKEN = os.getenv('TASTY_TOKEN')
ACCOUNT_UNIQUE_ID = os.getenv('TASTY_ACCOUNT_UNIQUE_ID')

TOKEN_PATH = 'data/tt.token'


def save_token(token, path):
  with open(path, 'w') as file:
    file.write(token)


def load_token(path):
  with open(path, 'r') as file:
    return file.read()


def token_is_recent(path, hours=24):
  if os.path.exists(path):
    mod_time = os.path.getmtime(path)
    return (datetime.now() -
            datetime.fromtimestamp(mod_time)) < timedelta(hours=hours)
  return False


def get_session(username, password):
  if token_is_recent(TOKEN_PATH):
    remember_token = load_token(TOKEN_PATH)
    logger.info(f'Loading the existing token: {remember_token}')
    # delete the token as well after one time use
    os.remove(TOKEN_PATH)
    return Session(username, remember_token=remember_token)
  else:
    session = Session(username, password, remember_me=True)
    logger.info(
        f'Saving the new token into: {TOKEN_PATH}. val={session.remember_token}'
    )
    save_token(session.remember_token, TOKEN_PATH)
    return session


class TastyBite:

  def __init__(self, account_unique_id=ACCOUNT_UNIQUE_ID):
    value = utils.decrypt_string(KEY, TOKEN)
    self.session = get_session(USERNAME, value)

    self.accounts = None
    self.account_unique_id = account_unique_id

  def get_risk_free_rate(self):
    return get_risk_free_rate(self.session)

  def get_accounts(self):
    if not self.accounts:
      self.accounts = Account.get_accounts(self.session)
    return self.accounts

  @property
  def default_acc(self):
    return self.get_accounts()[0]

  def get_positions(self, index=0):
    account_idx = self.get_accounts()[index]
    positions = account_idx.get_positions(self.session)
    return positions

  def get_balances(self):
    balance = self.default_acc.get_balances(self.session)
    return balance

  def historical_txn(self, date_begin):
    history = self.default_acc.get_history(self.session,
                                           start_date=date(2024, 1, 1))
    return history

  def get_private_watchlist(self, name):
    watchlist = Watchlist.get_private_watchlist(self.session, name)
    return watchlist

  def get_public_watchlists(self):
    watchlists = Watchlist.get_public_watchlists(self.session)
    return watchlists

  def get_public_watchlist(self, name):
    watchlist = Watchlist.get_public_watchlist(self.session, name)
    return watchlist

  def get_private_watchlists(self):
    watchlists = Watchlist.get_private_watchlists(self.session)
    return watchlists

  def get_private_watchlists(self):
    watchlists = Watchlist.get_private_watchlists(self.session)
    return watchlists

  def get_private_watchlist(self, name):
    watchlist = Watchlist.get_private_watchlist(self.session, name)
    return watchlist

  def get_orders(self):
    orders = self.default_acc.get_live_orders(self.session)
    return orders


def parse_position(position_str):
  pattern = r"(\w+)=([^,\)]+)"
  matches = re.findall(pattern, position_str)
  position = {}
  for key, value in matches:
    if value.startswith('Decimal'):
      position[key] = Decimal(value.split("'")[1])
    elif value.startswith('datetime.datetime'):
      dt_str = re.search(r'\((.*?)\)', value).group(1)
      dt_parts = [int(p) for p in dt_str.split(',') if p.strip().isdigit()]
      position[key] = datetime(*dt_parts, tzinfo=ZoneInfo("UTC"))
    elif value.startswith('datetime.date'):
      date_str = re.search(r'\((.*?)\)', value).group(1)
      date_parts = [int(p) for p in date_str.split(',')]
      position[key] = datetime(*date_parts).date()
    elif value.startswith('<InstrumentType.'):
      position[key] = TT_Order.InstrumentType(value.split("'")[1])
    elif value.startswith('<PriceEffect.'):
      position[key] = PriceEffect(value.split("'")[1])
    elif value == 'None':
      position[key] = None
    elif value.startswith("'") and value.endswith("'"):
      position[key] = value.strip("'")
    else:
      position[key] = value
  return position


def get_table_string(positions):
  table = create_rich_table(positions)

  # Create a StringIO object to capture the output
  string_io = io.StringIO()

  # Create a Console object that writes to the StringIO
  console = Console(file=string_io, force_terminal=True)

  # Print the table to the console (which writes to the StringIO)
  console.logger.info(table)

  # Get the string value and return it
  return string_io.getvalue()


def positions_to_dataframe(positions):
  """
    Convert positions data to a pandas DataFrame with the same structure as the rich table.

    Args:
        positions: List of position objects with attributes matching the table columns

    Returns:
        pandas.DataFrame: DataFrame containing formatted position data
    """

  # Initialize lists to store data
  data = []
  get_val = lambda position, name: getattr(position, name)

  for position in positions:
    # Format expires_at
    expires_val = get_val(position, 'expires_at')
    if expires_val:
      expires_val = expires_val.strftime("%Y-%m-%d %H:%M:%S")
    else:
      expires_val = 'N/A'

    # Create row data
    row = {
        'Symbol': get_val(position, 'symbol'),
        'InstrumentType': get_val(position, 'instrument_type').value,
        'Quantity': get_val(position, 'quantity'),
        'Direction': get_val(position, 'quantity_direction'),
        'Close Price': f"${get_val(position, 'close_price'):.2f}",
        'Open Price': f"${get_val(position, 'average_open_price'):.2f}",
        'Realized Day Gain': f"${get_val(position, 'realized_day_gain'):.2f}",
        'Expires At': expires_val
    }
    data.append(row)

  # Create DataFrame
  df = pd.DataFrame(data)

  return df


def create_rich_table(positions):
  table = Table(title="Equity Positions")

  columns = [
      "Symbol", "Type", "Quantity", "Direction", "Close Price", "Open Price",
      "Realized Day Gain", "Expires At"
  ]

  # for column in columns:
  #   table.add_column(column, style="cyan", justify="right")

  for column in columns:
    # Set the "Type" column to green
    # style = "green" if column == "Type" else "cyan"
    table.add_column(column, justify="right")

  get_val = lambda name: getattr(position, name)

  for position in positions:
    logger.info(f'position = {position}')
    expires_val = get_val('expires_at')
    if expires_val:
      expires_val = expires_val.strftime("%Y-%m-%d %H:%M:%S")
    else:
      expires_val = 'N/A'

    row_style = Style(color="yellow") if get_val(
        'instrument_type') == TT_Order.InstrumentType.EQUITY_OPTION else Style(
            color="green")

    table.add_row(
        get_val('symbol'),
        Text(get_val('instrument_type').value),
        str(get_val('quantity')),  # qty
        get_val('quantity_direction'),  # direction
        f"${get_val('close_price'):.2f}",  # close price
        f"${get_val('average_open_price'):.2f}",
        f"${get_val('realized_day_gain'):.2f}",
        expires_val,  # expires
        style=row_style)

  return table


class TastyUtilities:

  def __init__(self):
    self.tasty_bite = TastyBite()
    self.session = self.tasty_bite.session

  def create_orders_dataframe(self, orders):
    data = []
    for order in orders:
      row = {
          'Symbol':
          order.underlying_symbol,
          'Type':
          order.order_type.value,
          'Status':
          order.status.value,
          'Time in Force':
          order.time_in_force.value,
          'Price':
          float(order.price),
          'Size':
          float(order.size),
          'Instrument Type':
          order.underlying_instrument_type.value,
          'Legs':
          ', '.join([
              f"{leg.action.value} {leg.quantity} {leg.symbol}"
              for leg in order.legs
          ]),
          'Received At':
          order.received_at,
          'Last Updated':
          order.updated_at
      }
      data.append(row)

    df = pd.DataFrame(data)
    return df

  def print_orders(self, orders):
    logger.info("\n=== Orders ===\n")
    for index, order in enumerate(orders, 1):
      logger.info(f"Order {index}:")
      logger.info(f"  Symbol: {order.underlying_symbol}")
      logger.info(f"  Type: {order.order_type.value}")
      logger.info(f"  Status: {order.status.value}")
      logger.info(f"  Time in Force: {order.time_in_force.value}")
      logger.info(f"  Price: ${order.price}")
      logger.info(f"  Size: {order.size}")
      logger.info(
          f"  Instrument Type: {order.underlying_instrument_type.value}")

      logger.info("  Legs:")
      for leg in order.legs:
        logger.info(f"    - {leg.action.value} {leg.quantity} {leg.symbol}")

      logger.info(
          f"  Received at: {order.received_at.strftime('%Y-%m-%d %H:%M:%S')}")
      logger.info(
          f"  Last Updated: {order.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")

  def get_filtered_orders(self, instrument_types=None, order_types=None):
    open_orders = self.tasty_bite.get_orders()
    filtered_orders = []

    for order in open_orders:
      include_order = order.status not in [
          TT_Order.OrderStatus.REJECTED, TT_Order.OrderStatus.CANCELLED
      ]

      if instrument_types:
        include_order = include_order and order.underlying_instrument_type in instrument_types

      if order_types:
        include_order = include_order and order.order_type in order_types

      if include_order:
        filtered_orders.append(order)

    return filtered_orders

  def get_stop_loss_orders(
      self,
      instrument_types=[TT_Order.InstrumentType.EQUITY],
      order_types=[TT_Order.OrderType.STOP, TT_Order.OrderType.STOP_LIMIT]):
    return self.get_filtered_orders(instrument_types, order_types)

  def get_pending_open_orders(
      self,
      instrument_types=[TT_Order.InstrumentType.EQUITY],
      order_types=[TT_Order.OrderType.LIMIT]):
    return self.get_filtered_orders(instrument_types, order_types)

  def get_pending_open_orders_only_equities(
      self,
      instrument_types=[TT_Order.InstrumentType.EQUITY],
      order_types=[TT_Order.OrderType.LIMIT]):
    orders = self.get_filtered_orders(instrument_types, order_types)
    return [order for order in orders if len(order.legs) == 1]

  def get_equity_positions_without_sl(self):
    open_orders = self.tasty_bite.get_orders()
    for order in open_orders:
      logger.info(f'open order = {order}')

    positions = self.tasty_bite.get_positions()
    logger.info(f'positions  = {positions}')
    for posn in positions:
      logger.info(f'posn = {posn}')

    # Create a set of equities with SL orders
    equities_with_sl = set()
    for order in open_orders:
      if order.underlying_instrument_type == TT_Order.InstrumentType.EQUITY and order.order_type in [
          TT_Order.OrderType.STOP, TT_Order.OrderType.STOP_LIMIT
      ]:
        equities_with_sl.add(order.underlying_symbol)

    # Check which equities do not have SL orders
    positions_without_sl = []
    for position in positions:
      if position.instrument_type == TT_Order.InstrumentType.EQUITY and position.underlying_symbol not in equities_with_sl:
        logger.info(f'appending: {position}')
        positions_without_sl.append(position)

    return positions_without_sl

  def get_positions_as_df(self):
    positions = self.tasty_bite.get_positions()
    logger.info(f'positions  = {positions}')
    df_positions = positions_to_dataframe(positions)
    return df_positions

  def print_positions(self):

    positions = self.tasty_bite.get_positions()
    logger.info(f'positions  = {positions}')

    table_str = get_table_string(positions)

    logger.info(f'pretty print of positions = \n{table_str}')

  def place_order(self,
                  ticker,
                  price,
                  qty,
                  action=TT_Order.OrderAction.BUY_TO_OPEN,
                  time_in_force=TT_Order.OrderTimeInForce.GTC,
                  order_type=TT_Order.OrderType.LIMIT,
                  price_effect=TT_Order.PriceEffect.DEBIT,
                  dry_run=True):
    logger.info(f'ticker: {ticker}; price = {price}')
    price = round(price, 2) if price else None
    equity = Equity.get_equity(self.session, ticker)
    leg = equity.build_leg(qty, action)
    # round off price to 2 decimals

    logger.info(f"Placing NEW order for {ticker} at {price}; "
                f'; qty={qty}; dry_run={dry_run}; '
                f'price_effect={price_effect}')

    new_order = TT_Order.NewOrder(
        time_in_force=time_in_force,
        order_type=order_type,
        price=price,
        legs=[leg],  # you can have multiple legs in an order
        # price=stop_loss_price,  # limit price, $10/share for a total value of $50
        price_effect=price_effect)
    logger.info(f'About to place order: {new_order}')

    try:
      response = self.tasty_bite.default_acc.place_order(
          self.tasty_bite.session, new_order, dry_run=dry_run)
      logger.info(f"Placed NEW order for {ticker} at {price}; "
                  f'; qty={qty}; response = {response}')
      return response
    except Exception as e:
      logger.error(f'Exception during order placement : {str(e)}')

  def set_stop_loss_for_equities(self,
                                 positions,
                                 stop_loss_threshold=0.08,
                                 dry_run=True):
    for position in positions:
      symbol = position.underlying_symbol

      # Calculate stop-loss price
      stop_loss_price = position.average_open_price * decimal.Decimal(
          1 - stop_loss_threshold)
      stop_loss_price = round(stop_loss_price, 2)
      logger.info(
          f'Putting SL for position: <{symbol}> = {position}; SL = {stop_loss_price}'
      )

      # https://tastyworks-api.readthedocs.io/en/v7.6/orders.html
      equity = Equity.get_equity(self.session, symbol)
      leg = equity.build_leg(position.quantity,
                             TT_Order.OrderAction.SELL_TO_CLOSE)

      stop_loss_order = TT_Order.NewOrder(
          time_in_force=TT_Order.OrderTimeInForce.GTC,
          order_type=TT_Order.OrderType.STOP,
          stop_trigger=stop_loss_price,
          legs=[leg],  # you can have multiple legs in an order
          # price=stop_loss_price,  # limit price, $10/share for a total value of $50
          price_effect=TT_Order.PriceEffect.CREDIT)
      logger.info(f'SL order = {stop_loss_order}')

      # Place the stop-loss order
      response = self.tasty_bite.default_acc.place_order(
          self.tasty_bite.session, stop_loss_order, dry_run=dry_run)
      logger.info(f"Placed SL order for {symbol} at {stop_loss_price}; "
                  f'response = {response}')

  def set_stop_loss_for_position(self,
                                 position,
                                 stop_loss_price,
                                 dry_run=True):
    symbol = position.underlying_symbol

    stop_loss_price = round(stop_loss_price, 2)
    logger.info(
        f'Putting SL for position: <{symbol}> = {position}; SL = {stop_loss_price}'
    )

    # https://tastyworks-api.readthedocs.io/en/v7.6/orders.html
    equity = Equity.get_equity(self.session, symbol)
    leg = equity.build_leg(position.quantity,
                           TT_Order.OrderAction.SELL_TO_CLOSE)

    stop_loss_order = TT_Order.NewOrder(
        time_in_force=TT_Order.OrderTimeInForce.GTC,
        order_type=TT_Order.OrderType.STOP,
        stop_trigger=stop_loss_price,
        legs=[leg],  # you can have multiple legs in an order
        # price=stop_loss_price,  # limit price, $10/share for a total value of $50
        price_effect=TT_Order.PriceEffect.CREDIT)
    logger.info(f'SL order = {stop_loss_order}')

    # Place the stop-loss order
    response = self.tasty_bite.default_acc.place_order(self.tasty_bite.session,
                                                       stop_loss_order,
                                                       dry_run=dry_run)
    logger.info(f"Placed SL order for {symbol} at {stop_loss_price}; "
                f'response = {response}')
    return response


def main():
  tasty_utils = TastyUtilities()
  # tasty_utils.print_positions()
  positions_without_sl = tasty_utils.get_equity_positions_without_sl()

  tasty_utils.set_stop_loss_for_equities(positions_without_sl[:1])

  # df_positions = pd.DataFrame(positions)
  # logger.info(f'df positions =\n{df_positions.to_string()}')

  # wl = tasty_bite.get_public_watchlists()
  # logger.info(f'watchlists = {wl}')


if __name__ == '__main__':
  main()
'''
METRICS look like this:

+---------------------------------------------+---------------------------------------------------+
| Metric                                      | Value                                             |
+---------------------------------------------+---------------------------------------------------+
| Symbol                                      | JNPR                                              |
| Implied Volatility Index                    | 0.205448027                                       |
| 5-Day Change in Implied Volatility Index    | -0.091668907                                      |
| Implied Volatility Index Rank               | 0.498737829                                       |
| TOS Implied Volatility Index Rank           | 0.498737829                                       |
| TW Implied Volatility Index Rank            | 0.115622979                                       |
| TOS IV Rank Updated At                      | 2024-07-26 19:52:35 UTC                           |
| IV Rank Source                              | tos                                               |
| Implied Volatility Percentile               | 0.300892607                                       |
| Implied Volatility Updated At               | 2024-07-26 19:52:35 UTC                           |
| Liquidity Rating                            | 3                                                 |
| Updated At                                  | 2024-07-29 11:40:36 UTC                           |
| Beta                                        | 0.945563583                                       |
| 3-Month Correlation with SPY                | 0.07                                              |
| Market Cap                                  | 12,261,339,555                                    |
| Earnings                                    | Actual EPS: 0.42, Consensus Estimate: 0.24,       |
|                                             | Expected Report Date: 2024-07-25, Quarter End:    |
|                                             | 2024-06-01, Time: AMC, Updated At: 2024-07-25     |
|                                             | 10:01:33 UTC                                      |
| Price-Earnings Ratio                        | 51.89179                                          |
| Earnings Per Share                          | 0.71784                                           |
| Dividend Rate Per Share                     | 0.22                                              |
| Implied Volatility (30 Day)                 | 20.54                                             |
| Historical Volatility (30 Day)              | 11.14                                             |
| Historical Volatility (60 Day)              | 9.73                                              |
| Historical Volatility (90 Day)              | 10.19                                             |
| IV/HV (30 Day) Difference                   | 9.4                                               |
| Beta Updated At                             | 2024-07-28 17:00:32 UTC                           |
| Dividend Ex-Date                            | 2024-08-30                                        |
| Dividend Next Date                          | 2022-08-31                                        |
| Dividend Pay Date                           | 2024-09-23                                        |
| Dividend Updated At                         | 2024-07-29 00:15:46 UTC                           |
| Liquidity Value                             | 1506.106442577                                    |
| Liquidity Rank                              | 0.00656502                                        |
| Liquidity Running State                     | Sum: 0.0, Count: 0, Started At: 2024-07-27 10:00: |
|                                             | 11 UTC, Updated At: None                          |
| Dividend Yield                              | 0.023624161                                       |
| Listed Market                               | XNYS                                              |
| Lendability                                 | Easy To Borrow                                    |
| Borrow Rate                                 | 0.0                                               |
+---------------------------------------------+---------------------------------------------------+
| Option Expiration Implied Volatilities:     |                                                   |
| - 2024-08-16 (PM, Standard)                 | 0.168805716                                       |
| - 2024-09-20 (PM, Standard)                 | 0.238910372                                       |
| - 2024-10-18 (PM, Standard)                 | None                                              |
| - 2024-11-15 (PM, Standard)                 | None                                              |
| - 2024-12-20 (PM, Standard)                 | 0.281232983                                       |
| - 2025-01-17 (PM, Standard)                 | 0.184046621                                       |
| - 2025-06-20 (PM, Standard)                 | 0.14519231                                        |
| - 2026-01-16 (PM, Standard)                 | 0.117304553                                       |
+---------------------------------------------+---------------------------------------------------+

'''
