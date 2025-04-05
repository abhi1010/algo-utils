import os
import logging
import pickle
import sys

from trading.common import utils

logger = utils.get_logger("tasty_ic")

from tastytrade import ProductionSession, Account
from tastytrade.metrics import get_market_metrics
from tastytrade.instruments import get_option_chain, get_future_option_chain, OptionType
from tastytrade.utils import get_tasty_monthly
from tastytrade.order import InstrumentType

from decimal import Decimal
from tabulate import tabulate
from datetime import date

from devtools import pprint
from tastytrade.dxfeed import EventType

from trading.common import utils
from trading.brokers.tasty import KEY, TOKEN
'''
  Tasty Iron Condors Screener. Not in use.

  Runs but is incomplete as there's more work to be done
  for finding out IC puts and calls, and the deltas, BPE etc.

'''


def get_position_tickers(session, account) -> list[str]:

  positions = account.get_positions(session)

  tickers = set()

  for position in positions:
    logger.info(position)
    if position.instrument_type == InstrumentType.EQUITY:
      tickers.add(position.symbol)
    elif position.instrument_type == InstrumentType.EQUITY_OPTION:
      tickers.add(position.underlying_symbol)

  return list(tickers)


def initialize_session(username: str, password: str):
  return ProductionSession(username, password)


def get_first_account(session):
  accounts = Account.get_accounts(session)
  if accounts:
    return accounts[0]
  else:
    raise ValueError("No accounts found")


def log_metrics(metrics, label):
  logger.info(f'{label} metrics: {metrics}')
  for metric in metrics:
    logger.info(f'{label}:: metric: {str(metric)}')


def get_report(watchlist_metrics, positions_metrics) -> str:
  # sort watchlist metrics by IV rank in decending order
  sorted_watchlist_metrics = sorted(
      watchlist_metrics,
      key=lambda x: x.implied_volatility_index_rank or Decimal("0"),
      reverse=True,
  )

  # Combine top 10, bottom 10 from watchlist, and all position metrics
  combined_metrics = {
      item.symbol: item
      for item in positions_metrics + sorted_watchlist_metrics[:10] +
      sorted_watchlist_metrics[-10:]
  }.values()

  # Remove duplicates and sort again by IV rank
  unique_metrics = sorted(
      {metric.symbol: metric
       for metric in combined_metrics}.values(),
      key=lambda x: x.implied_volatility_index_rank or Decimal("0"),
      reverse=True,
  )

  # Construct table data
  table_data = []
  for metric in unique_metrics:
    symbol = f"{metric.symbol}*" if metric in positions_metrics else metric.symbol
    iv_rank = f"{Decimal(metric.implied_volatility_index_rank) * 100:.1f}%"
    liquidity = (f"{Decimal(metric.liquidity_rank) * 100:.1f}%"
                 if metric.liquidity_rank else "")

    days_to_earnings = (
        f"{(metric.earnings.expected_report_date - date.today()).days}"
        if metric.earnings and metric.earnings.expected_report_date
        and metric.earnings.expected_report_date >= date.today() else "")
    table_data.append([symbol, iv_rank, liquidity, days_to_earnings])

  headers = [
      "Symbol",
      "IV Rank",
      "Liquidity",
      "DTE",
  ]

  return tabulate(table_data, headers=headers, tablefmt="plain")


# Placeholder function for getting option delta
# Replace with the actual method to get delta values
def get_option_delta(option):
  # Placeholder: Replace with actual API call or data retrieval
  return 0.20  # Example delta


def find_20_delta_options(options):
  # Find options closest to 20 delta for calls and -20 delta for puts
  calls = [option for option in options if option.option_type == 'C']
  puts = [option for option in options if option.option_type == 'P']

  call_20_delta = min(calls, key=lambda x: abs(get_option_delta(x) - 0.20))
  put_20_delta = min(puts, key=lambda x: abs(get_option_delta(x) + 0.20))

  return call_20_delta, put_20_delta


def calculate_iron_condor(session, ticker):
  # Fetch option chain
  chain, exp = get_option_chains_list(session, ticker)
  # chain = await tt.options.get_option_chain(session, ticker)

  # Get the specific expiration you want to work with
  # exp = '2024-09-20'  # Example expiration date

  if exp not in chain:
    raise ValueError(f"Expiration {exp} not found in the option chain.")

  options = chain[exp]
  import pickle

  with open('data/options-bac.pickle', 'wb') as f:
    pickle.dump(options, f)

  sys.exit(1)

  # Find the 20 delta options
  short_call, short_put = find_20_delta_options(options)

  # Define the strikes for the long positions (wider spread)
  long_call_strike = short_call.strike_price  # do we need to adjust?
  long_put_strike = short_put.strike_price  # do we need to adjust?

  # Find the long call and put options
  long_call = next(option for option in options
                   if option.strike_price == long_call_strike
                   and option.option_type == OptionType.CALL)
  long_put = next(option for option in options
                  if option.strike_price == long_put_strike
                  and option.option_type == OptionType.PUT)

  logger.info(f'Short Call: {short_call}')
  logger.info(f'Short Put: {short_put}')

  # Fetch the premiums
  short_call_price = fetch_option_price(session, short_call)
  short_put_price = fetch_option_price(session, short_put)
  long_call_price = fetch_option_price(session, long_call)
  long_put_price = fetch_option_price(session, long_put)

  # Calculate the premiums
  credit_received = short_call_price + short_put_price
  debit_paid = long_call_price + long_put_price
  net_credit = credit_received - debit_paid

  # Calculate the buying power effect
  spread_width = max(abs(short_call.strike_price - long_call.strike_price),
                     abs(short_put.strike_price - long_put.strike_price))
  bpe = spread_width - net_credit

  return net_credit, bpe


def calculate_fake_ic():
  options = None
  with open('data/options-bac.pickle', 'rb') as f:
    options = pickle.load(f)
    logger.info(f'Loaded: {options}')

  for opt_item in options:
    logger.info(f'option item: {opt_item}')

  # Find the 20 delta options
  short_call, short_put = find_20_delta_options(options)
  logger.info(f'Short Call: {pprint(short_call)}')
  logger.info(f'Short Put: {pprint(short_put)}')

  # Define the strikes for the long positions (wider spread)
  long_call_strike = short_call.strike_price + Decimal(2)
  long_put_strike = short_put.strike_price

  # Find the long call and put options
  long_call = next(option for option in options
                   if option.strike_price == long_call_strike
                   and option.option_type == OptionType.CALL)
  long_put = next(option for option in options
                  if option.strike_price == long_put_strike
                  and option.option_type == OptionType.PUT)

  logger.info(f'Long Call: {pprint(long_call)}')
  logger.info(f'Long Put: {pprint(long_put)}')


def create_iron_condor(session):
  net_credit, bpe = calculate_iron_condor(session, 'BAC')
  print(f"Net Credit Received: ${net_credit:.2f}")
  print(f"Buying Power Effect: ${bpe:.2f}")


def get_option_chains_list(session, ticker):
  chain = get_option_chain(session, ticker)
  logger.info(f'chain_type = {type(chain)}')
  tasty_monthly_expiry = get_tasty_monthly()  # 45 DTE expiration!

  logger.info(f'exp = {tasty_monthly_expiry}')

  for opt_exp in chain:
    logger.info(f'chain item = {opt_exp}')
    for option in chain[opt_exp]:
      logger.info(
          f'Expiry: {opt_exp}; type = {type(option)}; option = {option}')

  return chain, tasty_monthly_expiry


def run_session(session):
  account = get_first_account(session)
  logger.info(f'Account: {account}')

  position_tickers = get_position_tickers(session, account)
  logger.info(f'Position tickers: {position_tickers}')

  watchlist_metrics = get_market_metrics(session, ['SPY', 'AAPL'])
  log_metrics(watchlist_metrics, 'Watchlist')

  position_metrics = get_market_metrics(session, position_tickers)
  log_metrics(position_metrics, 'Position')  # Uncomment if needed

  report = get_report(watchlist_metrics, position_metrics)
  logging.info(f'report = \n{report}')


def main(post_to_tg: bool = True):
  value = utils.decrypt_string(KEY, TOKEN)
  session = initialize_session(os.getenv('TASTY_USERNAME'), value)
  run_session(session)


if __name__ == '__main__':
  main()
