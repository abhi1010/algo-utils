import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import ccxt

from trading.common import utils

logger = utils.get_logger('screener_2_reds', use_rich=True)

import argparse

from trading.common.utils import HLOC, Markets
from trading.screener import bollinger_band_width_percentile as bbwp
'''Usage

python trading/screener/screener_2_reds.py -t "AAPL,MSFT,GOOGL" --market spx
'''


def get_crypto_data(symbol, start_date):
  """Get cryptocurrency data using ccxt"""
  try:
    # Initialize Binance exchange (you can change this to other exchanges)
    exchange = ccxt.kucoin({
        'enableRateLimit': True,
    })

    # Convert start_date to timestamp in milliseconds
    since = int(start_date.timestamp() * 1000)

    # Fetch OHLCV data (1d timeframe)
    ohlcv = exchange.fetch_ohlcv(
        symbol,
        timeframe='1d',
        since=since,
        limit=5  # Get last 5 days
    )

    # Convert to DataFrame
    df = pd.DataFrame(
        ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # Add Adj Close column to match yfinance format
    df['Adj Close'] = df['Close']

    return df

  except Exception as e:
    raise Exception(f"Error fetching crypto data: {str(e)}")


def get_stock_data(symbol, start_date, end_date):
  """Get stock data using yfinance"""
  try:
    stock = yf.Ticker(symbol)
    df = stock.history(start=start_date, end=end_date)
    return df
  except Exception as e:
    raise Exception(f"Error fetching stock data: {str(e)}")


def scan_pattern(tickers, market_type=Markets.SPX):
  """
    Scans for tickers with pattern: green candle followed by 2 red candles,
    where first red is lower than previous open and second red is lower than first red's close
    """
  # Get today's date
  end_date = datetime.today()
  # Get data for last 5 days to ensure we have enough data
  start_date = end_date - timedelta(days=5)

  matching_tickers = []

  for ticker in tickers:
    try:
      # Get data based on market type
      if market_type.lower() != Markets.CRYPTO:
        df = get_stock_data(ticker, start_date, end_date)
      elif market_type.lower() == Markets.CRYPTO:
        # For crypto, we need to append USDT or appropriate trading pair
        symbol = f"{ticker}/USDT" if not '/' in ticker else ticker
        df = get_crypto_data(symbol, start_date)
      else:
        raise ValueError(f"Unsupported market type: {market_type}")

      # Skip if we don't have enough data
      if len(df) < 3:
        continue

      # Get the last 3 days of data
      last_three_days = df.tail(3)
      logger.info(f'ticker={ticker}; last_three_days=\n{last_three_days}')

      if len(last_three_days) < 3:
        continue

      # Check pattern conditions
      day1 = last_three_days.iloc[0]  # First day (should be green)
      day2 = last_three_days.iloc[1]  # Second day (first red)
      day3 = last_three_days.iloc[2]  # Third day (second red)

      # Check if day1 is green (close > open)
      condition1 = day1[HLOC.Col_CLOSE] > day1[HLOC.Col_Open]

      # Check if day2 is red (close < open) and lower than day1's open
      condition2 = (day2[HLOC.Col_CLOSE] < day2[HLOC.Col_Open]
                    and day2[HLOC.Col_CLOSE] < day1[HLOC.Col_Open])

      # Check if day3 is red (close < open) and lower than day2's close
      condition3 = (day3[HLOC.Col_CLOSE] < day3[HLOC.Col_Open]
                    and day3[HLOC.Col_CLOSE] < day2[HLOC.Col_CLOSE])

      # Check if day3 is today
      condition4 = day3.name.date() == end_date.date()

      if all([condition1, condition2, condition3]):
        matching_tickers.append(ticker)
        logger.info(f"Pattern found in {ticker}")
        logger.info(
            f"Day 1 (Green): Open={day1[HLOC.Col_Open]:.2f}, Close={day1[HLOC.Col_CLOSE]:.2f}"
        )
        logger.info(
            f"Day 2 (Red): Open={day2[HLOC.Col_Open]:.2f}, Close={day2[HLOC.Col_CLOSE]:.2f}"
        )
        logger.info(
            f"Day 3 (Red): Open={day3[HLOC.Col_Open]:.2f}, Close={day3[HLOC.Col_CLOSE]:.2f}"
        )
        logger.info("-" * 50)

    except Exception as e:
      logger.info(f"Error processing {ticker}: {str(e)}")
      continue

  return matching_tickers


def read_tickers_from_file(filename):
  """Read tickers from a file, one per line or comma-separated"""
  try:
    with open(filename, 'r') as f:
      content = f.read().strip()
      # Check if content contains commas
      if ',' in content:
        # Split by comma and clean each ticker
        tickers = [t.strip() for t in content.split(',')]
      else:
        # Split by lines and clean each ticker
        tickers = [t.strip() for t in content.splitlines()]
    return [t for t in tickers if t]  # Remove empty strings
  except FileNotFoundError:
    logger.info(f"Error: File '{filename}' not found.")
    sys.exit(1)
  except Exception as e:
    logger.info(f"Error reading file: {str(e)}")
    sys.exit(1)


def main():

  parser = argparse.ArgumentParser(description='Market pattern scanner')
  group = parser.add_mutually_exclusive_group(required=True)

  # add arg for tag
  parser.add_argument('--tag', default='Crypto 2reds', type=str)

  # arg called output_dir, default to be data/rsi
  parser.add_argument('--output-dir',
                      '-o',
                      default='data/crypto-2reds',
                      type=str)

  group.add_argument(
      '-f',
      '--file',
      help='File containing tickers (one per line or comma-separated)')
  group.add_argument('-t', '--tickers', help='Comma-separated list of tickers')
  parser.add_argument(
      '-m',
      '--market',
      choices=[Markets.CRYPTO, Markets.SPX],
      default=Markets.SPX,
      type=Markets,
      required=False,
      help='Market type: us (US stocks) or crypto (cryptocurrencies)')

  args = parser.parse_args()
  logger.info(f'args: {args}')

  if args.file:
    tickers_to_check = read_tickers_from_file(args.file)
  else:
    tickers_to_check = [ticker.strip() for ticker in args.tickers.split(',')]

  logger.info(f"Scanning {len(tickers_to_check)} tickers for pattern...")
  matching = scan_pattern(tickers_to_check, args.market)

  if not matching:
    logger.error("No tickers found matching the pattern.")
    return

  logger.info(f"Summary of matching tickers: {matching}")

  if args.market == Markets.CRYPTO:
    date_today = datetime.today()
    largest_date = date_today
    tickers_for_tradingview = matching

    bbwp.save_names_to_txt(tickers_for_tradingview, args.output_dir,
                           args.market, '1d', largest_date)
    new_tickers = bbwp.get_new_tickers_compared_to_older_file(
        tickers_for_tradingview, args.output_dir, args.market, '1d',
        largest_date)

    logger.info(f'new_tickers = {new_tickers}')
    add_prefix = lambda syms: [f'KUCOIN:{ticker}USDT' for ticker in syms]

    bbwp.send_telegram_msg(args, add_prefix(tickers_for_tradingview),
                           add_prefix(new_tickers), args.market)


# Example usage
if __name__ == "__main__":
  main()
