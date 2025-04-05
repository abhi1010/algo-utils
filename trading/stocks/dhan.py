from dhanhq import dhanhq
from enum import Enum
import pandas as pd
import datetime
from datetime import timedelta
import math
import re
import argparse
import pickle
import traceback

from trading.common import utils

logger = utils.get_logger('dhan-runner')

from trading.common.util_prints import *
from trading.common.util_dates import *
from trading.common import telegram_helper
from trading.portfolio.tracker import *
from trading.configs import txn_weekend
# from trading.services import telegram_runner
from trading.stocks.dhan_common import *

TICKERS_TO_AVOID_CLOSING = ['DEEPAKNTR', 'JUBLFOOD', 'AUBANK', 'KANORICHEM']
TICKERS_TO_AVOID_CLOSING = []


class NormalizeDataFrame:

  def __init__(self, df):
    self.df = df

  def normalize_trades_df(self):
    agg_dict = {
        'tradedQuantity':
            'sum',
        'tradedPrice':
            lambda x: sum(x * self.df.loc[x.index, 'tradedQuantity']) / sum(
                self.df.loc[x.index, 'tradedQuantity']),
        'dhanClientId':
            'first',
        'exchangeOrderId':
            'first',
        'exchangeTradeId':
            'first',
        'transactionType':
            'first',
        'orderType':
            'first',
        'tradingSymbol':
            'first',
        'securityId':
            'first',
        'createTime':
            'first',
        'updateTime':
            'first',
        'exchangeTime':
            'first',
        'exchangeSegment':
            'first',
        'productType':
            'first',
        'customSymbol':
            'first',
        'drvExpiryDate':
            'first',
        'drvOptionType':
            'first',
        'drvStrikePrice':
            'first'
    }

    summary_df = self.df.groupby('orderId').agg(agg_dict).reset_index()

    return summary_df

  def normalize_trades_df_by_ticker(self):
    agg_dict = {
        'tradedQuantity':
            'sum',
        'tradedPrice':
            lambda x: sum(x * self.df.loc[x.index, 'tradedQuantity']) / sum(
                self.df.loc[x.index, 'tradedQuantity']),
        'dhanClientId':
            'first',
        'exchangeOrderId':
            'first',
        'exchangeTradeId':
            'first',
        'transactionType':
            'first',
        'orderType':
            'first',
        'orderId':
            'first',
        'securityId':
            'first',
        'createTime':
            'first',
        'updateTime':
            'first',
        'exchangeTime':
            'first',
        'exchangeSegment':
            'first',
        'productType':
            'first',
        'customSymbol':
            'first',
        'drvExpiryDate':
            'first',
        'drvOptionType':
            'first',
        'drvStrikePrice':
            'first',
        'outstanding_trade_qty':
            'first',
        'qty_to_close':
            'first',
    }

    summary_df = self.df.groupby('tradingSymbol').agg(agg_dict).reset_index()

    return summary_df


def get_all_dhan_scripts(exchange_to_use='NSE'):
  dhan_ids = dict()
  df = pd.read_csv(DHAN_SCRIP_FILE)
  df = df[(df[DhanScripColumns.SEM_INSTRUMENT_NAME] == 'EQUITY') & (
      df[DhanScripColumns.SEM_EXM_EXCH_ID] == exchange_to_use) & (
          ~df[DhanScripColumns.SEM_CUSTOM_SYMBOL].fillna('').astype(str).str
          .contains(
              r'%|SEC\sRE|\sNCD|\sWarrant', flags=re.IGNORECASE, regex=True))]

  return df


def get_dhan_scrips_as_dict_id_as_key(exchange_to_use='NSE'):
  df = get_all_dhan_scripts(exchange_to_use)
  sem_mapping = dict()
  for index, row in df.iterrows():
    sem_security_id = row['SEM_SMST_SECURITY_ID']
    sem_trading_symbol = row['SEM_TRADING_SYMBOL']
    sem_mapping[sem_security_id] = sem_trading_symbol
  return sem_mapping


def get_dhan_scrips_as_dict_sym_as_key(exchange_to_use='NSE'):
  df = get_all_dhan_scripts(exchange_to_use)
  sem_mapping = dict()
  for index, row in df.iterrows():
    sem_security_id = row['SEM_SMST_SECURITY_ID']
    sem_trading_symbol = row['SEM_TRADING_SYMBOL']
    sem_mapping[sem_trading_symbol] = sem_security_id
  return sem_mapping


def get_dhan_scrips_as_list_with_info(exchange_to_use='NSE'):
  df = get_all_dhan_scripts(exchange_to_use)
  sem_mapping = list()
  for index, row in df.iterrows():
    sem_security_id = row['SEM_SMST_SECURITY_ID']
    sem_trading_symbol = row['SEM_TRADING_SYMBOL']
    sem_sym_desc = row['SEM_CUSTOM_SYMBOL']
    sem_mapping.append(
        dict(
            id=sem_security_id,
            symbol=sem_trading_symbol,
            description=sem_sym_desc))
  return sem_mapping


def gainers_expected_file_name():
  fmt = "gainers-list__%Y-%m-%d.csv"
  now_s = datetime.datetime.now().strftime(fmt)
  return os.path.join(GAINERS_DIR, now_s)


def read_gainers_list(max_rank=5):
  # file format is csv with each line as ticker,close_price
  filepath = gainers_expected_file_name()
  if os.path.exists(filepath):
    gainers = pd.read_csv(filepath, nrows=max_rank)
    return gainers
  else:
    return pd.DataFrame()


class DhanTracker:

  def __init__(self):

    self.dhan = dhanhq("client_id", TOKEN)

  def process_results(self, results):
    if results:
      assert (
          results['status'] == StatusDhan.SUCCESS
      ), f"expected status OK for results for remarks: {results['remarks']}"
    return results['data']

  def get_orders(self, only_valid_orders=False):
    # Ref: data/dhan-samples/order-update.json
    orders = self.dhan.get_order_list()
    if orders:
      assert (
          orders['status'] == StatusDhan.SUCCESS
      ), f"expected status OK for Orders for remarks: {orders['remarks']}"
      # logger.info(f'orders = {orders}')
      if not only_valid_orders:
        return orders['data']
      return [
          o for o in orders['data'] if o[Fields.OrderStatus] not in [
              OrderStatusVals.CANCELLED, OrderStatusVals.REJECTED,
              OrderStatusVals.TRADED
          ]
      ]

  def get_positions(self):
    positions = self.dhan.get_positions()
    if positions:
      # assert (
      #     positions['status'] == StatusDhan.SUCCESS
      # ), f"expected status OK for Positions for remarks: {positions['remarks']}"
      # logger.info(f'positions = {positions}')
      return positions['data']
    else:
      return []

  def get_holdings(self):
    holdings = self.dhan.get_holdings()
    if holdings:
      # logger.info(f'holdings = {holdings}')
      # assert (
      #     holdings['status'] == StatusDhan.SUCCESS
      # ), f"expected status OK for holdings for remarks: {holdings['remarks']}"

      return holdings['data']
    else:
      return []

  def get_trade_book(self):
    # Get trade book
    trade_book = self.dhan.get_trade_book()
    if trade_book:
      logger.info(f'trade_book = {trade_book}')
      assert (
          trade_book['status'] == StatusDhan.SUCCESS
      ), f"expected status OK for trade_book for remarks: {trade_book['remarks']}"

      return trade_book['data']

  def get_trade_history(self, from_date, to_date):

    # Get trade history
    trade_hist = self.dhan.get_trade_history(from_date, to_date, page_number=0)

    if trade_hist:
      logger.info(f'trade_hist = {trade_hist}')
      assert (
          trade_hist['status'] == StatusDhan.SUCCESS
      ), f"expected status OK for trade_hist for remarks: {trade_hist['remarks']}"

      return trade_hist['data']

  def get_trades(self):
    near_date = utils.get_business_days_offset(-2)
    near_s = date_to_string_YYMM_MM_DD(near_date)
    today_s = utils.get_business_days_offset(1)
    logger.info(f'near_s = {near_s}, today_s = {today_s}')
    return self.get_trade_history(near_s, today_s)

  def place_order(
      self,
      security_id,
      qty,
      exchange_segment=dhanhq.NSE,
      transaction_type=dhanhq.BUY,
      quantity=1,
      order_type=dhanhq.MARKET,
      product_type=dhanhq.CNC,
      after_market_order=False,
      price=0,
      trigger_price=0,
      tag=None):
    #  [CNC, CO, BO, MARGIN, MTF, INTRADAY]
    # Place an order for Equity Cash
    # tag can be max 25 characters
    logger.info(f'Placing order: {locals()}')
    ord_result = self.dhan.place_order(
        security_id,
        exchange_segment=exchange_segment,
        transaction_type=transaction_type,
        quantity=qty,
        order_type=order_type,
        product_type=product_type,
        after_market_order=after_market_order,
        price=price,
        trigger_price=trigger_price,
        tag=tag)

    if ord_result:
      logger.info(f'ord_result = {ord_result}')
      if ord_result['status'] != StatusDhan.SUCCESS:
        logger.info(
            f"expected status OK for ord_result for remarks: {ord_result['remarks']}"
        )
      return ord_result['data']

  def forever_order(
      self,
      security_id,
      qty,
      exchange_segment=dhanhq.NSE,
      transaction_type=dhanhq.BUY,
      quantity=1,
      order_type=dhanhq.MARKET,
      product_type=dhanhq.CNC,
      price=0,
      trigger_price=0,
      tag=None):
    #  [CNC, CO, BO, MARGIN, MTF, INTRADAY]
    # Place an order for Equity Cash
    logger.info(f'Placing order: {locals()}')

    ord_result = self.dhan.forever_order(
        tag=tag,
        order_flag='SINGLE',
        transaction_type=transaction_type,
        exchange_segment=exchange_segment,
        product_type=product_type,
        order_type=order_type,
        validity=dhan.DAY,
        security_id=security_id,
        quantity=qty,
        price=price,
        trigger_price=trigger_price)

    if ord_result:
      logger.info(f'ord_result = {ord_result}')
      if ord_result['status'] != StatusDhan.SUCCESS:
        logger.info(
            f"expected status OK for ord_result for remarks: {ord_result['remarks']}"
        )
      return ord_result['data']

  def modify_order(
      self,
      order_id,
      quantity,
      trigger_price,
      price=0,
      order_type=dhanhq.MARKET,
      leg_name='ENTRY_LEG',
      disclosed_quantity=1,
      validity=dhanhq.DAY):
    ord_result = self.dhan.modify_order(
        order_id, order_type, leg_name, quantity, price, trigger_price,
        disclosed_quantity, validity)
    if ord_result:
      logger.info(f'ord_result = {ord_result}')
      if ord_result['status'] != StatusDhan.SUCCESS:
        logger.info(
            f"expected status OK for ord_result for remarks: {ord_result['remarks']}"
        )
      return ord_result['data']

  def cancel_all_orders(self):
    orders = self.get_orders(only_valid_orders=True)
    for order in orders:
      logger.info(f'About to cancel order: {order}')
      self.cancel_order(order['orderId'])

  def get_order_by_id(self, order_id):
    ord_result = self.dhan.get_order_by_id(order_id)
    if ord_result:
      logger.info(f'ord_result = {ord_result}')
      assert (
          ord_result['status'] == StatusDhan.SUCCESS
      ), f"expected status OK for ord_result for remarks: {ord_result['remarks']}"
      return ord_result['data']

  def cancel_order(self, order_id):
    # Cancel order
    ord_result = self.dhan.cancel_order(order_id)
    if ord_result:
      logger.info(f'ord_result = {ord_result}')
      if ord_result['status'] != StatusDhan.SUCCESS:
        logger.info(
            f"expected status OK for ord_result for remarks: {ord_result['remarks']}"
        )
      return ord_result['data']

  def close_all_positions_amo(
      self, after_market_order=True, product_type=dhanhq.INTRA):

    positions = self.get_positions()
    show_debug_info(positions, 'positions', save_to_file=True, save_to_db=True)
    existing_orders = self.get_orders()

    for position in positions:
      security_id = str(position['securityId'])
      qty = abs(position['netQty'])
      positionType = position['positionType'].upper()
      posn_txn_type = position['positionType']
      order_txn_type = dhanhq.SELL
      ticker = position['tradingSymbol']

      if positionType.upper() == 'SHORT':
        order_txn_type = dhanhq.BUY
      # logger.info(f'cancel fr {security_id} for qty: {qty}')
      if positionType != 'CLOSED':
        logger.info(
            f'Closing for: {ticker}: SecID: {security_id}; qty: {qty}; '
            f'txn type: {order_txn_type}; AMO: {after_market_order}; '
            f'product type: {product_type}; positionType = {positionType}')
        does_exist = self.check_order_doesnt_exist(
            existing_orders, security_id, qty, positionType, product_type,
            order_txn_type, 'DAY', after_market_order)
        logger.info(f'does_exist = {does_exist}')
        # res = dhan.place_order(
        #     security_id,
        #     qty,
        #     transaction_type=order_txn_type,
        #     after_market_order=after_market_order,
        #     product_type=product_type)
        # logger.info(f'res of sell: {res}')

  def close_all_assets(
      self,
      assets,
      asset_tag,
      after_market_order=True,
      product_type=dhanhq.INTRA,
      dry_run=True,
      close_column_name='availableQty'):
    order_results = []
    full_update = [
        f'*Closing All Assets* \nOrder type is `{product_type}`  \nDry Run `{dry_run}`'
    ]
    show_debug_info(assets, asset_tag, save_to_file=True, save_to_db=True)

    df_gainers = read_gainers_list()
    if df_gainers.empty:
      logger.warning(f'df_gainers is empty')
    gainer_tickers = df_gainers['ticker'].to_list(
    ) if not df_gainers.empty else []
    # close_prices = df_gainers['close'].to_list()
    logger.info(f'df_gainers = \n{gainer_tickers}')
    # logger.info(f'close_prices = \n{close_prices}')
    logger.info(f'assets = \n{assets.to_string()}. type = {type(assets)}')

    existing_orders = self.get_orders()
    for index, asset in assets.iterrows():
      logger.info(f'asset = \n{asset.to_string()}')
      security_id = str(asset['securityId'])
      qty = abs(asset[close_column_name])
      positionType = order_txn_type = dhanhq.SELL
      ticker = asset['tradingSymbol']
      if ticker in gainer_tickers:
        logger.info(f'Not selling {ticker} as it is a Gainer')
        full_update.append(f'{ticker} - Not selling as it is a Gainer')
        continue

      if not qty:
        logger.info(f'Not selling {ticker} as qty = {qty} unavailable')
        full_update.append(f'{ticker} - Not selling as qty: {qty} unavailable')
        continue

      if ticker in TICKERS_TO_AVOID_CLOSING:
        logger.info(
            f'Not selling {ticker} as it is in TICKERS_TO_AVOID_CLOSING'
            f'; TICKERS_TO_AVOID_CLOSING = {TICKERS_TO_AVOID_CLOSING}')
        full_update.append(
            f'{ticker} - Not selling as it is in TICKERS_TO_AVOID_CLOSING'
            f'; TICKERS_TO_AVOID_CLOSING = {TICKERS_TO_AVOID_CLOSING}')
        continue

      logger.info(
          f'Closing for: {ticker}: SecID: {security_id}; qty: {qty}; '
          f'txn type: {order_txn_type}; AMO: {after_market_order}; '
          f'product type: {product_type}; ')

      does_exist = self.check_order_doesnt_exist(
          existing_orders, security_id, qty, positionType, product_type,
          order_txn_type, dhanhq.DAY, after_market_order)

      logger.info(f'does_exist = {does_exist}')

      if not dry_run:
        res = self.place_order(
            security_id,
            qty,
            transaction_type=order_txn_type,
            after_market_order=after_market_order,
            product_type=product_type)
        order_results.append(res)
        logger.info(f'res of sell: {res}')
        full_update.append(f'{ticker} - {res}')
      else:
        full_update.append(
            f'{ticker} - Not placing order as dry_run = {dry_run}')
        logger.info(f'Not placing order as dry_run = {dry_run}')

    # telegram_runner.send_text(full_update)

    # add_results_into_db(order_results, delay=15)
    return order_results

  # TODO: Weekend: should check against DB if needed
  def check_order_doesnt_exist(
      self, orders_list, security_id, qty, txn_type, product_type, order_type,
      validity, after_market_order):
    for order in orders_list:
      if (order['securityId'] == str(security_id) and
          order['transactionType'] == txn_type and
          order['productType'] == product_type and
          order['orderType'] == order_type and order['validity'] == validity and
          order['quantity'] == qty and
          order['afterMarketOrder'] == after_market_order):
        return True
    return False

  def edis_enquiry(self, isin='ALL'):
    return self.process_results(self.dhan.edis_inquiry(isin))

  def generate_tpin(self):
    return self.dhan.generate_tpin()

  def get_fund_limits(self):
    return self.process_results(self.dhan.get_fund_limits())


## ----------------


def read_dhan_scrip(exchange='NSE', columns_to_drop=[]):
  df = pd.read_csv(DHAN_SCRIP_FILE)
  df = df[(df[DhanScripColumns.SEM_INSTRUMENT_NAME] == 'EQUITY') & (
      df[DhanScripColumns.SEM_EXM_EXCH_ID] == 'NSE') & (
          ~df[DhanScripColumns.SEM_CUSTOM_SYMBOL].fillna('').astype(str).str
          .contains(
              r'%|SEC\sRE|\sNCD|\sWarrant', flags=re.IGNORECASE, regex=True))]
  if columns_to_drop:
    df = df.drop(columns_to_drop, axis=1)
  return df


def is_new_order_valid_on_dhan(ticker, df_curr_orders, dhan_ticker_info):
  logger.info(f'ticker = {ticker}; type={type(ticker)}')
  ticker_has_no_order = df_curr_orders.empty or df_curr_orders[
      df_curr_orders['securityId'] == str(dhan_ticker_info.security_id)].empty
  if not ticker_has_no_order:
    logger.info(f'Order exists for : {dhan_ticker_info}. Ignoring')
    # we should print out all such orders too
    ticker_existing_orders = df_curr_orders[df_curr_orders['securityId'] == str(
        dhan_ticker_info.security_id
    )] if not df_curr_orders.empty else pd.DataFrame()
    logger.info(f'Existing Orders: \n{ticker_existing_orders.to_string()}')
    return False

  if not dhan_ticker_info.qty:
    logger.info(f'Order Qty is 0 for : {dhan_ticker_info}. Ignoring')
    return False
  return True


def open_amo_positions(
    dhan_sec_ids,
    product_type=dhanhq.INTRA,
    after_market_order=False,
    place_orders=False):
  dhan = DhanTracker()

  curr_orders = dhan.get_orders()
  df_curr_orders = pd.DataFrame(curr_orders)
  logger.info(f'df_curr_orders = \n{df_curr_orders}')
  # df_curr_orders = df_curr_orders[
  #     (df_curr_orders['afterMarketOrder'] == True)
  #     & ~(df_curr_orders['orderStatus'].isin(['CANCELLED', 'REJECTED']))]
  if not df_curr_orders.empty:
    df_curr_orders = df_curr_orders[
        (df_curr_orders['afterMarketOrder'] == True)
        & ~(
            df_curr_orders[Fields.OrderStatus].isin(
                [OrderStatusVals.CANCELLED, OrderStatusVals.REJECTED]))]

    logger.info(f'df_curr_orders = \n{df_curr_orders.to_string()}')
    logger.info(
        f"sec ids from curr orders: {df_curr_orders['securityId'].to_list()}")
  else:
    logger.info(f'Empty df_curr_orders')

  for ticker in dhan_sec_ids:
    dhan_ticker_info = dhan_sec_ids[ticker]
    valid_order = is_new_order_valid_on_dhan(
        ticker, df_curr_orders, dhan_ticker_info)
    if not valid_order:
      continue

    logger.info(f'Placing order for {dhan_ticker_info}')
    order_res = dhan.place_order(
        str(dhan_ticker_info.security_id),
        dhan_ticker_info.qty,
        after_market_order=after_market_order,
        product_type=product_type)

    show_debug_info(
        order_res, 'order-results', save_to_file=False, save_to_db=True)
    logger.info(f'Order placement results: {order_res}')


def show_df_in_table_format(text, df):
  if isinstance(df, list) and df == []:
    return ''
  simplified_df = pd.DataFrame().to_string(index=False)
  if text.lower() in ['orders', 'positions', 'trades', 'tradebook',
                      'trades_history', 'holdings', 'funds', 'fund_limits']:

    if text == 'orders':
      simplified_df = df.drop(
          OrderColumnsToRemove, axis=1).to_string(
              index=False) if not df.empty else simplified_df
      logger.info(f'{text}:: df = \n{simplified_df}')

    elif text == ['funds', 'fund_limits']:
      logger.info(f'{text}:: df = \n{df}')
      logger.info(f'{text}:: df cols = \n{df.columns.to_list()}')
      logger.info(f'FundColumnsToRemove = {FundColumnsToRemove}')

      simplified_df = df.drop(
          FundColumnsToRemove, axis=1).to_string(
              index=False) if not df.empty else simplified_df
      logger.info(f'{text}:: df = \n{simplified_df}')

    elif text == 'holdings':
      logger.info(f'{text}:: df = \n{df.to_string()}')
      logger.info(f'{text}:: df cols = \n{df.columns.to_list()}')
      simplified_df = df.drop(
          HoldingsColumnsToRemove, axis=1).to_string(
              index=False) if not df.empty else simplified_df
      # logger.info(f'{text}:: df = \n{df.to_string()}')
      logger.info(f'{text}:: df = \n{simplified_df}')

    elif text == 'positions':
      logger.info(f'{text}:: df  = \n{df}')
      logger.info(f'{text}:: df cols = \n{df.columns.to_list()}')
      simplified_df = df.drop(
          PositionColumnsToRemove, axis=1).to_string(
              index=False) if not df.empty else simplified_df
      # logger.info(f'{text}:: df = \n{df.to_string()}')
      logger.info(f'{text}:: df = \n{simplified_df}')

    elif text in ['trades_history', 'trades', 'tradebook']:
      logger.info(f'{text}:: df cols = \n{df.columns.to_list()}')
      simplified_df = df.drop(
          TradesHistoryColumnsToRemove, axis=1).to_string(
              index=False) if not df.empty else simplified_df
      # logger.info(f'{text}:: df = \n{df.to_string()}')
      logger.info(f'{text}:: df = \n{simplified_df}')

  else:
    logger.info(f'{text}:: df (none) = \n{df.to_string(index=False)}')

  res = df.to_string(index=False)
  if simplified_df:
    res = simplified_df
    logger.info(f'{text}; simplified..... ')
  else:
    logger.info(f'{text}; res ..... ')

  return simplified_df if simplified_df else df.to_string(index=False)


def write_info_to_csv(info, text, filename_csv, save_to_file):
  try:
    df = pd.DataFrame(info)
    show_df_in_table_format(text, df)

    if not df.empty:
      if save_to_file:
        df.to_csv(filename_csv, index=False)
    else:
      logger.info(f'{text}:: df is empty, not saving to csv')
  except Exception as e:
    logger.exception(
        f'Exception in writing to CSV: {str(e)}; info={info}; text={text}')
    exc = traceback.format_exc()
    logger.info(f'exception = {exc}')
    logger.info(f'*' * 80)
    stacks = ''.join(traceback.format_stack())
    logger.info(f'stacks = {stacks}')
    logger.info('Exception END')


def write_info_to_json(info, txn_type, filename_json, save_to_file):
  info_s = get_pretty_print(info)
  logger.info(f'{txn_type}:: info = \n{info_s}')

  if save_to_file:
    with open(filename_json, 'wt') as f:
      f.write(str(info_s).replace("'", '"').replace('": None', '": null'))


def write_info_to_db(info, txn_type, tag):
  # tag = txn_weekend.DEFAULT_TAG
  try:
    df = pd.DataFrame(info)

    if txn_type in ['trades', 'tradebook']:
      data_normalizer = NormalizeDataFrame(df)
      df = data_normalizer.normalize_trades_df()
      logger.info(f'normalized trades data = \n{df.to_string()}')

    txn_mgr = txn_weekend.TxnManager()
    # trim s from end of the text
    text_removed_s = txn_type[:-1]
    col_ticker, col_date = DHAN_INFO_COLS[
        txn_type] if txn_type in DHAN_INFO_COLS else DHAN_INFO_COLS[
            text_removed_s] if text_removed_s in DHAN_INFO_COLS else ['', '']
    if not col_date:
      date_val = datetime.datetime.now().date()
    # iterate rows
    for index, row in df.iterrows():
      # Perform operations on each row
      # ...
      # convert row to json
      # row_s = json.dumps(row.to_dict())
      row_s = row.to_dict()
      logger.info(f'row_s = {row_s}')
      date_col_val = row[col_date] if col_date else date_val
      logger.info(f'  other values  = tag={tag}; txn_type={txn_type}; ')
      logger.info(f'  col_ticker = {col_ticker}; ')
      logger.info(f'  other val2: row[col_ticker]={row[col_ticker]};')
      logger.info(f'  other val3: date_col_val={date_col_val}')
      txn_mgr.write_transaction(
          tag, txn_type, row[col_ticker], date_col_val, row_s)
    logger.info(f'Writing to DB finished for {tag} & {txn_type}')
  except Exception as e:
    logger.info(
        f'Exception in writing to DB: {str(e)}; info={info}; txn_type={txn_type}'
    )
    exc = traceback.format_exc()
    logger.info(f'exception = {exc}')
    logger.info(f'*' * 80)
    stacks = ''.join(traceback.format_stack())
    logger.info(f'stacks = {stacks}')
    logger.info('Exception END')


def show_debug_info(
    info, txn_type, save_to_file=False, save_to_db=False, tag='trial'):
  now = datetime.datetime.now()
  os.makedirs(DHAN_DIR, exist_ok=True)

  date_s = now.strftime('%Y-%m-%d')
  time_s = now.strftime('%H-%M-%S')
  filename_csv = os.path.join(DHAN_DIR, f'{date_s}-{txn_type}-{time_s}.csv')
  filename_json = os.path.join(DHAN_DIR, f'{date_s}-{txn_type}-{time_s}.json')

  if save_to_file:
    write_info_to_csv(info, txn_type, filename_csv, save_to_file)
    write_info_to_json(info, txn_type, filename_json, save_to_file)
  else:
    try:
      df = pd.DataFrame(info)
      show_df_in_table_format(txn_type, df)
    except Exception as e:
      logger.info(f'exception: {str(e)}')
      logger.info(f'info = {info}')

  if save_to_db:
    write_info_to_db(info, txn_type, tag)


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument(
      '--view',
      '-v',
      choices=[
          'orders', 'positions', 'holdings', 'trades', 'edis', 'tradebook',
          'test', 'scrips'
      ],
      default='orders',
      required=True,
      help='What info do you want to view')

  # add arg for save to db
  parser.add_argument('--save-to-db', action='store_true', help='Save to DB')

  # add arg for save to file
  parser.add_argument(
      '--save-to-file', action='store_true', help='Save to file')

  args = parser.parse_args()
  logger.info(f'args = {args}')
  return args


def view_trades(save_to_file, save_to_db, dhan):
  today_s = date_to_string_YYMM_MM_DD()
  today_s_minus_1 = date_to_string_YYMM_MM_DD(offset=-5)
  logger.info(f'today_s_minus_1 = {today_s_minus_1}; today_s = {today_s}')
  trades_hist = dhan.get_trade_history(today_s_minus_1, today_s)
  dhan_ids = get_dhan_scrips_as_dict_id_as_key()
  for trd in trades_hist:
    if not trd['tradingSymbol']:
      if int(trd['securityId']) in dhan_ids:
        t = trd['tradingSymbol'] = dhan_ids[int(trd['securityId'])]
        logger.info(f'Updated t = {t}')
      else:
        logger.info(f'Mising securityID in dhan ids: {trd["securityId"]}')
    else:
      t = trd['tradingSymbol']
      logger.info(f'Found TrdSym t = {t}')
    logger.info(f'trd = {trd}')
  logger.info(f'trades_hist = {trades_hist}')
  show_debug_info(trades_hist, 'trades', save_to_file, save_to_db, DEFAULT_TAG)


if __name__ == '__main__':

  # # dhan.close_all_positions_amo()

  # res = dhan.generate_tpin()
  # logger.info(f'res = {res}')

  args = get_args()
  dhan = DhanTracker()

  # Run Scrips
  if args.view == 'scrips':
    scrips = get_all_dhan_scripts()
    # columns to remove
    cols_to_remove = [
        'SEM_EXM_EXCH_ID', ' SEM_SEGMENT', 'SEM_EXPIRY_CODE', 'SEM_EXPIRY_DATE',
        'SEM_INSTRUMENT_NAME', 'SEM_STRIKE_PRICE', 'SEM_OPTION_TYPE',
        'SEM_EXPIRY_FLAG'
    ]

    scrips.drop(cols_to_remove, axis=1, inplace=True)
    # logger.info(f'scrips = \n{scrips}')
    # logger.info(f'scrip cols = {scrips.columns}')
    dhan_ids = get_dhan_scrips_as_dict_id_as_key()
    logger.info(f'dhan ids = {dhan_ids}')
    dhan_syms = get_dhan_scrips_as_dict_sym_as_key()
    logger.info(f'dhan syms = {dhan_syms}')
    logger.info(f'id lens = {len(dhan_ids)}, sym lens = {len(dhan_syms)}')

  # Run EDIS
  if args.view == 'edis':
    # edis enquiry
    res = dhan.edis_enquiry()
    logger.info(f'res = {res}')

  # Run CANCEL
  if args.view == 'cancel':
    dhan.cancel_all_orders()

  # Run ORDERS
  if args.view == 'orders':
    # Fetch all orders
    orders = dhan.get_orders(only_valid_orders=False)
    logger.info(f'orders = {orders}')

    show_debug_info(orders, 'orders', args.save_to_file, args.save_to_db)

    valid_orders = [
        o for o in orders if o[Fields.OrderStatus] not in [
            OrderStatusVals.CANCELLED, OrderStatusVals.TRADED,
            OrderStatusVals.REJECTED
        ]
    ]
    show_debug_info(valid_orders, 'orders', args.save_to_file, args.save_to_db)

  # Run POSITIONS
  if args.view == 'positions':
    positions = dhan.get_positions()
    show_debug_info(positions, 'positions', args.save_to_file, args.save_to_db)

  # Run HOLDINGS
  if args.view == 'holdings':
    holdings = dhan.get_holdings()
    # Createa filename by timestamp
    filename = 'data/holdings_' + datetime.datetime.now().strftime(
        '%Y%m%d_%H%M%S') + '.pickle'
    with open(filename, 'wb') as f:
      pickle.dump(holdings, f)
    logger.info(f'holdings = {holdings}; type={type(holdings)}')
    show_debug_info(holdings, 'holdings', args.save_to_file, args.save_to_db)

  # Run TRADES
  if args.view == 'trades':
    view_trades(args.save_to_file, args.save_to_db, dhan)
    # logger.info(trades_hist)

  # Run TRADEBOOK
  if args.view == 'tradebook':
    trade_book = dhan.get_trade_book()
    show_debug_info(
        trade_book, 'tradebook', args.save_to_file, args.save_to_db,
        DEFAULT_TAG)
    logger.info(trade_book)

  # date_today = datetime.datetime.today().strftime('%Y-%m-%d')
  # txn_mgr = txn_weekend.TxnManager()
  # if args.view == 'test':
  #   holdings = dhan.get_holdings()
  #   for holding in holdings:
  #     logger.info(f'holding = {holding}; type={type(holding)}')
  #     txn_mgr.write_transaction(
  #         DEFAULT_TAG, DHAN_TXN_TYPE.HOLDINGS,
  #         holding[DHAN_INFO_COLS[DHAN_TXN_TYPE.HOLDINGS][0]], date_today,
  #         holding)

  # place_order_wilmar = dhan.place_order(
  #     '8110', 14, after_market_order=True)  # adani wilmar
  # {'status': 'success', 'remarks': '', 'data': {'orderId': '552311297609', 'orderStatus': 'TRANSIT'}}

  # [CNC, CO, BO, MARGIN, MTF, INTRADAY]

  # df = read_dhan_scrip(
  #     columns_to_drop=[
  #         DhanScripColumns.SEM_EXPIRY_DATE, DhanScripColumns.SEM_STRIKE_PRICE,
  #         DhanScripColumns.SEM_OPTION_TYPE, DhanScripColumns.SEM_TICK_SIZE,
  #         DhanScripColumns.SEM_EXPIRY_FLAG, DhanScripColumns.SEM_EXPIRY_CODE
  #     ])
  # df.to_csv('data/dhan-samples/dhan-scrip.csv', index=False)
  # logger.info(f'df = \n{df}')
