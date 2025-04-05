from trading.common import utils

logger = utils.get_logger('quallamaggie_charts', use_rich=True)

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import argparse

from trading.common.utils import HLOC, Markets
from trading.screener import bollinger_band_width_percentile as bbwp
'''Usage

python trading/stocks/quallamaggie_charts.py  \
    --dir data/nse-bbwp-screener \
    --output-dir data/quallamaggie-charts-nse \
    --tag "Quallamaggie NIFTY" \
    --market nifty
'''


def get_data(ticker, start_date, end_date):
  stock = yf.Ticker(ticker)
  data = stock.history(start=start_date, end=end_date)
  data.index = data.index.tz_localize(None)

  df_csv = pd.read_csv(f'data/spx-bbwp-screener/{ticker}.csv')
  df_csv.index = pd.to_datetime(df_csv['Date'])

  logger.info(f'df_csv = \n{df_csv}')
  logger.info(f'data = \n{data}')
  return df_csv


def get_data_from_csv(ticker):
  df_csv = pd.read_csv(f'data/spx-bbwp-screener/{ticker}.csv')
  return df_csv


def find_episodic_pivots(ticker,
                         data,
                         start_date,
                         end_date,
                         future_60d_return_threshold=0.10):

  if data.empty:
    logger.info(f"No data available for {ticker} in the specified date range.")
    return pd.DataFrame(), None

  # Calculate daily returns
  data['Return'] = data['Close'].pct_change()

  # The return 60 days later, compared to the current close
  data['Return60'] = data['Close'].shift(-60) / data['Close'] - 1

  # Calculate average volume for comparison
  data['SMA_Volume'] = data['Volume'].rolling(window=50).mean()

  # Identify potential EPs
  eps = []
  for i in range(1, len(data)):
    # Check for gap up of 10% or more
    if data['Open'][i] >= data['Close'][i - 1] * 1.10:
      # Check if volume is massive near the open
      current_volume = data['Volume'][i]
      if current_volume >= data['SMA_Volume'][
          i] * 2:  # Volume at least twice the 50-day average

        # The return 60 days later
        return60 = data['Return60'][i]
        dt = data.index[i].date()
        if return60 < future_60d_return_threshold:
          logger.info(f'{ticker} failed; dt={dt}; return60={return60}')
        else:
          logger.info(f'{ticker} passed; dt={dt}; return60={return60}')

        if return60 >= future_60d_return_threshold:
          eps.append({
              'Date': data.index[i].date(),
              'Gap': data['Open'][i] / data['Close'][i - 1] - 1,
              'Volume': current_volume,
              'SMA_Volume': data['SMA_Volume'][i],
              'Return60': return60
          })

  return pd.DataFrame(eps), data


def create_pivot_chart(data, pivot_date, ticker, output_dir):
  # Convert pivot_date to datetime if it's not already

  if isinstance(pivot_date, str):
    pivot_date = datetime.strptime(pivot_date, '%Y-%m-%d')

  filename = f"{output_dir}/{ticker}_pivot_{pivot_date.strftime('%Y%m%d')}.png"
  if os.path.exists(filename):
    logger.info(f"Chart already exists: {filename}")
    return

  # Define the date range for the chart
  start_date = pivot_date - timedelta(days=60)
  end_date = pivot_date + timedelta(days=60)

  # Filter data for the chart period
  mask = (data.index >= pd.to_datetime(start_date)) & (
      data.index <= pd.to_datetime(end_date))
  chart_data = data[mask]

  if len(chart_data) == 0:
    logger.info(f"No data available for chart around {pivot_date}")
    return

  # Create the chart
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[2, 1])
  fig.suptitle(f'{ticker} Pivot Analysis - {pivot_date}', fontsize=16)

  # Candlestick chart
  width = 0.6
  width2 = 0.05

  up = chart_data[chart_data.Close >= chart_data.Open]
  down = chart_data[chart_data.Close < chart_data.Open]

  # Up candlesticks
  ax1.bar(up.index,
          up.Close - up.Open,
          width,
          bottom=up.Open,
          color='green',
          alpha=0.6)
  ax1.bar(up.index, up.High - up.Close, width2, bottom=up.Close, color='green')
  ax1.bar(up.index, up.Low - up.Open, width2, bottom=up.Open, color='green')

  # Down candlesticks
  ax1.bar(down.index,
          down.Close - down.Open,
          width,
          bottom=down.Open,
          color='red',
          alpha=0.6)
  ax1.bar(down.index,
          down.High - down.Open,
          width2,
          bottom=down.Open,
          color='red')
  ax1.bar(down.index,
          down.Low - down.Close,
          width2,
          bottom=down.Close,
          color='red')

  # Add pivot line
  ax1.axvline(x=pivot_date, color='purple', linestyle='--', label='Pivot Date')
  ax1.set_ylabel('Price')
  ax1.legend()
  ax1.grid(True)

  # Volume chart
  ax2.bar(chart_data.index,
          chart_data['Volume'],
          color=[
              'green' if close >= open else 'red'
              for close, open in zip(chart_data.Close, chart_data.Open)
          ],
          alpha=0.6)
  ax2.plot(chart_data.index,
           chart_data['SMA_Volume'],
           color='orange',
           label='50-day Volume SMA')
  ax2.axvline(x=pivot_date, color='purple', linestyle='--')
  ax2.set_ylabel('Volume')
  ax2.legend()
  ax2.grid(True)

  # Rotate x-axis labels for better readability
  plt.xticks(rotation=45)
  plt.tight_layout()

  # Create directory if it doesn't exist
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Save the chart

  plt.savefig(filename)
  plt.close()

  logger.info(f"Chart saved: {filename}")


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--dir",
                      default='data/spx-bbwp-screener',
                      help="Which directory to save to",
                      type=str)

  # arg called output_dir, default to be data/bbwp
  parser.add_argument('--output-dir',
                      '-o',
                      default='data/quallamaggie_charts',
                      type=str)
  parser.add_argument('--timeframe',
                      '-t',
                      default='1d',
                      type=str,
                      help='What timeframe/interval to use')

  # add tag as ttm squeeze
  parser.add_argument('--tag', default='quallamaggie charts', type=str)

  parser.add_argument('--market',
                      type=Markets,
                      default=Markets.SPX,
                      help='What market')

  parser.add_argument('--future-60d-return-threshold',
                      type=float,
                      default=0.10,
                      help='60-day return threshold')

  args = parser.parse_args()
  return args


def main():
  # Example usage
  end_date = datetime.now(tz=None)
  start_date = end_date - timedelta(days=365 * 5)  # how many years back?
  start_date_dt, end_date_dt = start_date.strftime(
      '%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

  args = get_args()
  logger.info(f'args: {args}')

  # Create output directory for charts
  output_dir = args.output_dir
  os.makedirs(output_dir, exist_ok=True)

  # files_df = bbwp.process_files(args.dir)
  files_df = bbwp.process_files(args.dir, limit=10)

  dfs, largest_date = bbwp.filter_away_stale_data(files_df, args.market,
                                                  args.timeframe)

  for ticker, data in dfs.items():
    data.index = pd.to_datetime(data['Date'])
    pivots, historical_data = find_episodic_pivots(
        ticker, data, start_date_dt, end_date_dt,
        args.future_60d_return_threshold)

    if not pivots.empty and historical_data is not None:
      logger.info(f"Episodic Pivots for {ticker}: \n{pivots}")
      # logger.info(pivots)
      # logger.info(f"\nHistorical data for {ticker}: \n{historical_data}")

      # Create charts for each pivot
      for pivot_date in pivots['Date']:
        create_pivot_chart(historical_data, pivot_date, ticker, output_dir)

      logger.info(
          f"\nAll charts have been saved to the '{output_dir}' directory")
    else:
      logger.info(f"No Episodic Pivots found for {ticker}.")


if __name__ == "__main__":
  main()
