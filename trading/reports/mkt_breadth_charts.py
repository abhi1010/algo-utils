from trading.common import utils

logger = utils.get_logger('market-breadth-chart', use_rich=True)

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from enum import Enum
from datetime import datetime
import os

from trading.services import telegram_runner
from trading.screener.screener_common import Markets

import pandas as pd
import matplotlib.pyplot as plt
import argparse
from enum import Enum
from datetime import datetime
import os
import yfinance as yf


class Markets(Enum):
  SPX = "SPX"


import pandas as pd
import matplotlib.pyplot as plt
import argparse
from enum import Enum
from datetime import datetime
import os


class Markets(Enum):
  SPX = "SPX"


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--file',
                      default='data/spx/market-breadth-spx.xlsx',
                      type=str)

  parser.add_argument('--market',
                      type=Markets,
                      default=Markets.SPX,
                      help='What market')

  parser.add_argument('--window', '-w', default=200, type=int, help='Window')

  parser.add_argument('--output',
                      '-o',
                      default='data/tmp/spx-market_movements.png',
                      type=str,
                      help='Output file path')

  # New arguments for columns and colors
  parser.add_argument('--up_column',
                      default='25% up in 20 days',
                      type=str,
                      help='Column name for upward movement')

  parser.add_argument('--down_column',
                      default='25% down in 20 days',
                      type=str,
                      help='Column name for downward movement')

  parser.add_argument('--up_color',
                      default='#2ecc71',
                      type=str,
                      help='Color for upward line (hex code or named color)')

  parser.add_argument('--down_color',
                      default='#e74c3c',
                      type=str,
                      help='Color for downward line (hex code or named color)')

  parser.add_argument('--date_column',
                      default='Date/Time',
                      type=str,
                      help='Column name for dates')

  args = parser.parse_args()
  return args


def process_data(file_path, window, date_col, up_col, down_col):
  # Read Excel file
  df = pd.read_excel(file_path)

  # Select only needed columns
  cols = [date_col, up_col, down_col]
  df = df[cols].copy()

  # Take last n rows based on window
  df = df.tail(window)

  # Convert dates to datetime if they aren't already
  df[date_col] = pd.to_datetime(df[date_col])

  # download spy data
  spy_df = yf.download('SPY',
                       period='max',
                       progress=False,
                       multi_level_index=False)
  logger.info(f'spy_df = \n{spy_df}')

  spy_df = spy_df[['Close']]
  spy_df.columns = ['SPY']
  spy_df[date_col] = pd.to_datetime(spy_df.index)

  df = pd.merge(df, spy_df, on=date_col, how='left')
  logger.info(f'df = \n{df}')

  return df


def create_plot(
    df,
    market,
    window,
    output_path,
    date_col,
    up_col,
    down_col,
    up_color,
    down_color,
    spy_col='SPY',  # New parameter for SPY data column
    spy_color='purple'  # Default color for SPY line
):
  # Set the style
  plt.style.use('bmh')

  # Create figure and axis with specified size
  fig, ax = plt.subplots(figsize=(15, 8))

  # Plot all three lines
  ax.plot(df[date_col], df[up_col], color=up_color, label=up_col, linewidth=2)
  ax.plot(df[date_col],
          df[down_col],
          color=down_color,
          label=down_col,
          linewidth=2)
  ax.plot(df[date_col],
          df[spy_col],
          color=spy_color,
          label=spy_col,
          linewidth=2)

  # Customize the plot
  ax.set_title(
      f'{market} Market Movements - Last {window} days\n{datetime.now().strftime("%Y-%m-%d")}',
      pad=20,
      fontsize=14)
  ax.set_xlabel('Date')
  ax.set_ylabel('Number of Stocks')
  ax.grid(True, linestyle='--', alpha=0.7)
  ax.legend(loc='upper left')

  # Rotate x-axis labels for better readability
  plt.xticks(rotation=45)

  # Adjust layout to prevent label cutoff
  plt.tight_layout()

  # Save the plot
  plt.savefig(output_path, dpi=300, bbox_inches='tight')
  plt.close()


# Example usage with your screenshot function:
def process_and_send_market_data(output_file, index_name):
  """
    Process market data and send to Telegram
    """
  try:
    telegram_runner.send_image_with_text(image_path=output_file,
                                         index_name=index_name)

  except Exception as e:
    logger.exception(f'Error processing and sending market data: {str(e)}')


def main():
  # Get command line arguments
  args = get_args()

  # Process the data
  df = process_data(args.file, args.window, args.date_column, args.up_column,
                    args.down_column)

  # Create and save the plot
  create_plot(df, args.market.value, args.window, args.output,
              args.date_column, args.up_column, args.down_column,
              args.up_color, args.down_color)

  print(f"Plot saved as {args.output}")

  process_and_send_market_data(args.output, args.market)


if __name__ == "__main__":
  main()
