import asyncio
import os
import sys
from datetime import date
import datetime
import signal
import pickle
import argparse
import re
from dataclasses import dataclass
from collections import defaultdict

from trading.common import utils

logger = utils.get_logger("tasty-greeks", True)

from dataclasses import dataclass, field
from typing import List, Dict

from tastytrade import DXLinkStreamer
from tastytrade.instruments import get_option_chain, OptionType
from tastytrade.dxfeed import Greeks, Quote
from tastytrade.utils import today_in_new_york
from tastytrade.session import Session

from tastytrade.instruments import get_option_chain, Option
from tastytrade.utils import (get_tasty_monthly, get_future_fx_monthly,
                              get_future_grain_monthly,
                              get_future_index_monthly,
                              get_future_metal_monthly, get_future_oil_monthly,
                              get_future_treasury_monthly)
from tastytrade.metrics import get_market_metrics
from tastytrade.streamer import EventType

from tastytrade import DXLinkStreamer, Session
from tabulate import tabulate

from trading.common import utils
from trading.brokers.tasty import KEY, TOKEN, UESRNAME as TASTY_USERNAME

from trading.options import option_utils
from trading.screener import yf_option_ic


@dataclass
class LivePrices:
  quotes: Dict[str, Quote] = field(default_factory=dict)
  greeks: Dict[str, Greeks] = field(default_factory=dict)
  streamer: DXLinkStreamer = None
  puts: Dict[str, List[Option]] = field(default_factory=dict)
  calls: Dict[str, List[Option]] = field(default_factory=dict)
  tickers: List[str] = field(default_factory=list)
  pickle_file: str = "data/greeks_data.pickle"
  _update_task: asyncio.Task = None
  subscription_items: List[str] = field(default_factory=list)

  @classmethod
  async def create(cls,
                   session: Session,
                   symbols: List[str],
                   exp_tasty_monthly: date = today_in_new_york(),
                   use_pickle: bool = False,
                   event_greeks_or_quotes: EventType = EventType.GREEKS,
                   all_streamer_symbols=[]):
    self = cls(tickers=symbols)
    if use_pickle and os.path.exists(self.pickle_file):
      logger.info(f'Loading from pickle file: {self.pickle_file}')
      self._load_from_pickle()
    else:
      logger.info(
          f'Initializing streamer for {symbols or all_streamer_symbols}; exp_tasty_monthly = {exp_tasty_monthly}'
      )
      await self._initialize_streamer(session)
      is_valid = await self._process_symbols(
          session,
          symbols,
          exp_tasty_monthly,
          event_greeks_or_quotes,
          all_streamer_symbols=all_streamer_symbols)

      if is_valid:
        if event_greeks_or_quotes == EventType.GREEKS:
          await self._wait_for_greeks()
        else:
          await self._wait_for_quotes()

    return self

  def _load_from_pickle(self):
    with open(self.pickle_file, 'rb') as f:
      self.greeks = pickle.load(f)
    logger.info(f"Loaded greeks data from {self.pickle_file}")

  def _save_to_pickle(self):
    with open(self.pickle_file, 'wb') as f:
      pickle.dump(self.greeks, f)
    logger.info(f"Saved greeks data to {self.pickle_file}")

  async def _update_greeks(self):
    async for e in self.streamer.listen(EventType.GREEKS):
      logger.info(f'Received greek 0: {e};')
      self.greeks[e.eventSymbol] = e
      # self._save_to_pickle()  # Save to pickle file after each update

  async def _update_quotes(self):
    async for e in self.streamer.listen(EventType.QUOTE):
      logger.info(f'Received quote 0: {e};')
      self.quotes[e.eventSymbol] = e
      # self._save_to_pickle()  # Save to pickle file after each update

  def _start_listening_for_greeks(self):
    self._update_task = asyncio.create_task(self._update_greeks())

  def _start_listening_for_quotes(self):
    self._update_task = asyncio.create_task(self._update_quotes())

  async def cleanup(self):
    if self._update_task:
      self._update_task.cancel()
      try:
        await self._update_task
      except asyncio.CancelledError:
        pass
    if self.streamer:
      await self.streamer.close()

  async def _initialize_streamer(self, session: Session):
    self.streamer = await DXLinkStreamer.create(session)

  async def _process_symbols(self, session: Session, symbols: List[str],
                             exp_tasty_monthly: date,
                             event_greeks_or_quotes: EventType,
                             all_streamer_symbols: List[str]):
    if symbols and len(symbols):
      all_streamer_symbols = []
    logger.info(f'[{event_greeks_or_quotes}] -> incoming symbols = {symbols}')
    logger.info(
        f'[{event_greeks_or_quotes}] -> incoming all_streamer_symbols = {all_streamer_symbols}'
    )

    full_list_streamer_symbols = all_streamer_symbols

    for symbol in symbols:
      logger.info(f'Symbol: {symbol}')
      chain = self._get_option_chain(session, symbol)
      if not chain:
        logger.info(f'Skipping {symbol} with no option chain')
        return False

      logger.info(f'Symbol: {symbol}; chains = {len(chain)}')
      if len(chain):
        for item in chain:
          logger.info(
              f'Symbol: {symbol}; Chain item = {item}; type = {type(chain[item])}'
          )
      options = self._filter_options_by_expiration(symbol, chain,
                                                   exp_tasty_monthly)
      streamer_symbols = self._get_streamer_symbols(options)
      logger.info(f'Symbol: {symbol}; streamer_symbols={streamer_symbols}')
      full_list_streamer_symbols.extend(streamer_symbols)
      self._categorize_options(symbol, options)

    await self._subscribe_to_event(full_list_streamer_symbols,
                                   event_greeks_or_quotes)

    if event_greeks_or_quotes == EventType.GREEKS:
      self._start_listening_for_greeks()
    else:
      self._start_listening_for_quotes()
    return True

  def _get_option_chain(self, session: Session, symbol: str):
    chains = None
    try:
      chains = get_option_chain(session, symbol)
    except Exception as e:
      logger.error(f'Exception : {str(e)}')
    return chains

  def _filter_options_by_expiration(self, symbol, chain, expiration: date):
    closest_expiry = get_closest_expiry(symbol, chain, expiration)
    logger.info(f'[{symbol}]:: closest_expiry = {closest_expiry}'
                f' all chains = {chain}')
    return [o for o in chain[closest_expiry]]

  def _get_streamer_symbols(self, options: List[Option]):
    return [o.streamer_symbol for o in options]

  def _categorize_options(self, symbol: str, options: List[Option]):
    self.puts[symbol] = [o for o in options if o.option_type == OptionType.PUT]
    self.calls[symbol] = [
        o for o in options if o.option_type == OptionType.CALL
    ]

  async def _subscribe_to_event(self, streamer_symbols: List[str],
                                event_greeks_or_quotes: EventType):
    logger.info(
        f'About to subscribe {event_greeks_or_quotes} to: {streamer_symbols} for greeks'
    )
    self.subscription_items.append(streamer_symbols)
    await self.streamer.subscribe(event_greeks_or_quotes, streamer_symbols)
    logger.info(
        f'Finished subscribing for {event_greeks_or_quotes} to: {streamer_symbols}'
    )

  def unsubscribe_to_event(self, streamer_symbols: List[str],
                           event_greeks_or_quotes: EventType):
    logger.info(
        f'About to unsubscribe {event_greeks_or_quotes} to: {streamer_symbols} '
    )
    if not len(self.subscription_items):
      logger.info(
          f'Nothing to unsubsribe to for {event_greeks_or_quotes} to: {streamer_symbols}'
      )
      return
    last_subscribed_items = self.subscription_items.pop()
    self.streamer.unsubscribe(event_greeks_or_quotes, last_subscribed_items)
    logger.info(f'Finished unsubscribing to: {last_subscribed_items}')

  async def _wait_for_quotes(self):
    while len(self.quotes) < 4:
      logger.info(f'waiting for quotes...')
      logger.info(f'len(quotes) = {len(self.quotes)} ')
      await asyncio.sleep(3)

  async def _wait_for_greeks(self):
    total_options = sum(
        len(self.puts[s]) + len(self.calls[s]) for s in self.tickers)

    try:
      await asyncio.wait_for(self._wait_until_greeks_ready(total_options),
                             timeout=15)
    except asyncio.TimeoutError:
      logger.warning(
          'Timeout reached: waiting for greeks took longer than 30 seconds.')

  async def _wait_until_greeks_ready(self, total_options):
    while len(self.greeks) < 0.99 * total_options:
      logger.info('waiting for greeks...')
      logger.info(f'len(greeks) = {len(self.greeks)} / {total_options}')
      await asyncio.sleep(3)

  def get_greeks_for_ticker(self, ticker: str):
    ticker = ticker.replace('/', '.')
    return [g for s, g in self.greeks.items() if s.startswith(f'.{ticker}')]

  def get_quotes_for_ticker(self, ticker: str):
    ticker = ticker.replace('/', '.')
    return [g for s, g in self.quotes.items() if s.startswith(f'.{ticker}')]


def convert_greeks_to_optionholders(greeks_list, tasty_monthly):
  option_holders = []

  # Regex pattern to match the eventSymbol format
  pattern = r"\.(\w+)(\d{6})([CP])(\d+(?:\.\d+)?)"

  for greek in greeks_list:
    option_holder = option_utils.OptionHolder()

    # Extract ticker, strike, expiry, and option type from the eventSymbol

    tasty_sym = greek.eventSymbol

    # Parse the eventSymbol using regex
    match = re.match(pattern, greek.eventSymbol)
    if match:
      option_holder.Ticker = match.group(1)
      expiry_str = match.group(2)
      option_type = match.group(3)
      strike_str = match.group(4)

      logger.info(f'greek.eventSymbol = {greek.eventSymbol}; '
                  f'ticker={option_holder.Ticker}; '
                  f'expiry={expiry_str}; option_type={option_type}; '
                  f'strike={strike_str}')

      # Convert expiry string to datetime object
      option_holder.Expiry = tasty_monthly

      # Set strike price
      option_holder.Strike = float(strike_str)

      # Set option type
      option_holder.IsCall = option_holder.IsCallCE_Or_PE = True if option_type in [
          'C', 'CE'
      ] else False

    else:
      logger.info(f'greek.eventSymbol = {greek.eventSymbol}')
      raise ValueError(f'Failed to parse eventSymbol: {greek.eventSymbol}')

    # Set other attributes from the Greeks object
    option_holder.IV = float(greek.volatility)
    option_holder.Theta = float(greek.theta)
    option_holder.Delta = float(greek.delta)
    option_holder.Gamma = float(greek.gamma)
    option_holder.eventSymbol = greek.eventSymbol

    # Set the price as both BidPrice and AskPrice for now
    option_holder.BidPrice = float(greek.price)
    option_holder.AskPrice = float(greek.price)

    # Calculate CurrentSpread
    option_holder.CurrentSpread = option_holder.AskPrice - option_holder.BidPrice

    # Set other attributes that we can't derive from Greeks
    option_holder.Valid = True
    option_holder.YahooString = f"{option_holder.Ticker}{option_holder.Expiry.strftime('%y%m%d')}{'C' if option_holder.IsCall else 'P'}{option_holder.Strike:08.2f}"

    option_holders.append(option_holder)

  return option_holders


async def update_options_with_bid_ask_from_tasty(session, tasty_monthly_expiry,
                                                 ticker, option_holders_list,
                                                 use_pickle):

  call_to_sell, call_to_buy, put_to_sell, put_to_buy = yf_option_ic.get_calls_and_puts_for_ic(
      option_holders_list, ticker, delta_for_search=0.2)

  if call_to_sell is None or call_to_buy is None or put_to_sell is None or put_to_buy is None:
    logger.info(f'[{ticker}]:: Missing call or put with delta of 50 for IC')
    return False

  event_symbols_list = [
      call_to_sell.eventSymbol, call_to_buy.eventSymbol,
      put_to_sell.eventSymbol, put_to_buy.eventSymbol
  ]

  live_quotes = await LivePrices.create(
      session, [],
      tasty_monthly_expiry,
      event_greeks_or_quotes=EventType.QUOTE,
      use_pickle=use_pickle,
      all_streamer_symbols=event_symbols_list)

  specific_quotes = live_quotes.get_quotes_for_ticker(ticker)

  for quote in specific_quotes:
    # find a matching option holder
    logger.info(f'[{ticker}]:: quote = {quote}')
    for oh in option_holders_list:
      if oh.eventSymbol == quote.eventSymbol:
        logger.info(f' Found matching option holder: {oh}')
        oh.BidPrice = float(quote.bidPrice)
        oh.AskPrice = float(quote.askPrice)
        logger.info(f' Updated matching option holder: {oh}')

  live_quotes.unsubscribe_to_event(event_symbols_list, EventType.QUOTE)
  return True


def get_first_day_of_next_month() -> datetime:
  today = datetime.datetime.today()
  first_day_next_month = datetime.datetime(
      today.year, today.month +
      1, 1) if today.month < 12 else datetime.datetime(today.year + 1, 1, 1)
  last_day_current_month = first_day_next_month - datetime.timedelta(days=1)
  return first_day_next_month


def get_expiries_by_symbol(tasty_metrics):
  expiries = defaultdict(list)
  for metric in tasty_metrics:
    implied_ivs = metric.option_expiration_implied_volatilities
    for iv in implied_ivs:
      expiries[metric.symbol].append(iv.expiration_date)
    logger.info(
        f'Added expiries [{metric.symbol}]:: {expiries[metric.symbol]}')

  return expiries


# def get_all_closest_expiries(expiries, input_date):
#   closest_expiries = dict()
#   for symbol, sym_expiries in expiries.items():
#     closest_expiries[symbol] = get_closest_expiry(
#         symbol, sym_expiries, input_date)
#   return closest_expiries


def get_closest_expiry(ticker, expiries, input_date):
  closest_expiry = min(expiries, key=lambda date: abs(date - input_date))
  logger.info(f'[{ticker}]:: closest_expiry = {closest_expiry}')
  return closest_expiry


# def get_all_tasty_metrics(session, tickers):
#   tickers_metrics = []
#   # break tickers into a subgroup of 100 items
#   tickers = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
#   for lst in tickers:
#     tickers_metrics.extend(get_market_metrics(session, lst))

#   return tickers_metrics


async def main(session, use_pickle, sub_group_size=1):
  liq_ticker_metrics = yf_option_ic.get_stock_list()
  tickers = sorted([x['symbol'] for x in liq_ticker_metrics])
  # tickers = tickers[:10]
  logger.info(f'tickers= {tickers}')
  # get the last day of this month
  first_day_next_mnth = get_first_day_of_next_month()

  exp_tasty_monthly = get_tasty_monthly()
  exp_oil_monthly = get_future_oil_monthly(first_day_next_mnth)
  exp_future_metal_mnthly = get_future_metal_monthly(first_day_next_mnth)
  exp_future_index_mnthly = get_future_index_monthly(first_day_next_mnth)
  exp_future_grain_monthly = get_future_grain_monthly(first_day_next_mnth)
  exp_future_fx_monthly = get_future_fx_monthly(first_day_next_mnth)
  exp_trsy_monthly = get_future_treasury_monthly(first_day_next_mnth)

  logger.info(f'EXPIRIES: \n'
              f' --> future_fx_monthly = {exp_future_fx_monthly} \n'
              f' --> future_grain_monthly = {exp_future_grain_monthly}; \n'
              f' --> future_index_mnthly = {exp_future_index_mnthly}; \n'
              f' --> future_metal_mnthly = {exp_future_metal_mnthly}; \n'
              f' --> future_oil_monthly = {exp_oil_monthly}; \n'
              f' --> future_treasury_monthly = {exp_trsy_monthly} \n'
              f' --> tasty_monthly = {exp_tasty_monthly}; \n')

  logger.info(
      f'exp_trsy_monthly = {exp_trsy_monthly} ; type={type(exp_trsy_monthly)}')
  # tasty_metrics = get_all_tasty_metrics(session, tickers)
  # expiries = get_expiries_by_symbol(tasty_metrics)
  # symbol_expiries = get_all_closest_expiries(expiries, exp_tasty_monthly)

  # for sym in symbol_expiries:
  #   logger.info(f'[{sym}] Closest Expiry == {symbol_expiries[sym]}')

  logger.info(
      f'tasty_monthly = {exp_tasty_monthly}; type={type(exp_tasty_monthly)}')
  all_ticker_options_dict = dict()

  # break tickers into groups of 3
  ticker_groups = [
      tickers[i:i + sub_group_size]
      for i in range(0, len(tickers), sub_group_size)
  ]
  logger.info(f'len(ticker_groups) = {len(ticker_groups)}')

  for i, tickers_grp in enumerate(ticker_groups):
    ticker = tickers_grp[0]

    logger.info(f'{i} : tickers to go through {tickers_grp}; ')

    live_prices = await LivePrices.create(
        session,
        tickers_grp,
        exp_tasty_monthly,
        event_greeks_or_quotes=EventType.GREEKS,
        use_pickle=use_pickle,
    )

    for ticker in tickers_grp:
      ticker_greeks = live_prices.get_greeks_for_ticker(ticker)
      option_holders_list = convert_greeks_to_optionholders(
          ticker_greeks, exp_tasty_monthly)

      options_updated_successfully = await update_options_with_bid_ask_from_tasty(
          session, exp_tasty_monthly, ticker, option_holders_list, use_pickle)

      if options_updated_successfully:
        yf_option_ic.get_iron_condor_credit(option_holders_list,
                                            ticker,
                                            delta_for_search=0.2)
        all_ticker_options_dict[ticker] = option_holders_list
      else:
        logger.info(f'Some issues with {ticker}. Skipping IC calculation')

    live_prices.unsubscribe_to_event(tickers_grp, EventType.GREEKS)

  logger.info(f'----' * 80)

  iron_condor_data = yf_option_ic.generate_iron_condor_stats(
      all_ticker_options_dict)

  yf_option_ic.tabulate_ic_data(iron_condor_data, liq_ticker_metrics)

  # Keep the program running
  # if not use_pickle:
  #   while True:
  #     await asyncio.sleep(1)


def handle_sigint():
  for task in asyncio.all_tasks():
    task.cancel()


async def run_main():

  parser = argparse.ArgumentParser(description="LivePrices with pickle option")
  parser.add_argument('-up',
                      "--use-pickle",
                      action="store_true",
                      help="Use pickled data if available")
  args = parser.parse_args()

  loop = asyncio.get_running_loop()
  loop.add_signal_handler(signal.SIGINT, handle_sigint)

  value = utils.decrypt_string(KEY, TOKEN)
  my_username = TASTY_USERNAME
  session = Session(my_username, value, remember_token=True)
  logger.info(f'logged in...')

  try:
    await main(session, args.use_pickle)
  except asyncio.CancelledError:
    logger.info("Main task cancelled")
  finally:
    logger.info("Cleanup complete")


if __name__ == "__main__":
  asyncio.run(run_main())
