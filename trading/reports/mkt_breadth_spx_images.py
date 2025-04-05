from trading.common import utils

logger = utils.get_logger('market-analysis', use_rich=True)

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import openpyxl
import matplotlib.pyplot as plt
from openpyxl.utils.dataframe import dataframe_to_rows
from PIL import Image

from trading.services import telegram_runner
from trading.screener.screener_common import Markets


def load_excel_data(file_path):
  """
    Load Excel data with specific columns
    """
  columns_to_keep = [
      "Date/Time", "Above 10 DEMA", "Above 20 DEMA", "Above 50 DEMA",
      "Above 200 DEMA"
  ]

  df = pd.read_excel(file_path, usecols=columns_to_keep)
  df['Date/Time'] = pd.to_datetime(df['Date/Time'])
  return df


def get_color_gradient(value, min_val, max_val):
  """
    Generate color based on value position in range
    Returns RGB tuple with red for lowest, yellow for middle, green for highest
    """
  if max_val == min_val:
    normalized = 0.5
  else:
    normalized = (value - min_val) / (max_val - min_val)

  if normalized <= 0.5:
    # Red to Yellow
    red = 1.0
    green = 2 * normalized
    blue = 0.0
  else:
    # Yellow to Green
    red = 2 * (1 - normalized)
    green = 1.0
    blue = 0.0

  return (red, green, blue)


def create_formatted_screenshot(df,
                                n_days,
                                output_path,
                                keep_bottom_percent=0.2):
  """
    Create screenshot with 3-color gradient and crop to keep bottom portion
    """
  # Get min/max for each numeric column using full dataset
  numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
  col_ranges = {col: (df[col].min(), df[col].max()) for col in numeric_cols}

  # Get last n days for display
  recent_data = df.sort_values('Date/Time').tail(n_days)

  # Create figure and axis
  fig, ax = plt.subplots(figsize=(15, 8))
  ax.axis('tight')
  ax.axis('off')

  # Format date column
  recent_data = recent_data.copy()
  recent_data['Date/Time'] = recent_data['Date/Time'].dt.strftime('%Y-%m-%d')

  # Create cell colors array
  num_rows = len(recent_data)
  num_cols = len(recent_data.columns)
  cell_colors = [[(1, 1, 1) for _ in range(num_cols)] for _ in range(num_rows)]

  # Apply color gradient for numeric columns
  for col in numeric_cols:
    col_idx = list(recent_data.columns).index(col)
    min_val, max_val = col_ranges[col]
    for row_idx, value in enumerate(recent_data[col].values):
      cell_colors[row_idx][col_idx] = get_color_gradient(
          value, min_val, max_val)

  # Create table
  table = ax.table(cellText=recent_data.values,
                   colLabels=recent_data.columns,
                   cellLoc='center',
                   loc='center',
                   cellColours=cell_colors,
                   colColours=[(0.9, 0.9, 0.9)] * num_cols)

  # Style the table
  table.auto_set_font_size(False)
  table.set_fontsize(9)
  table.scale(1.2, 1.5)

  # Save the full figure first
  temp_path = 'temp_full_image.png'
  plt.savefig(temp_path,
              bbox_inches='tight',
              dpi=300,
              pad_inches=0.5,
              facecolor='white')
  plt.close()

  # Open the image with PIL and crop
  with Image.open(temp_path) as img:
    width, height = img.size

    # Calculate crop dimensions
    # Keep header (assuming it's about 10% of height) plus bottom portion
    header_height = int(height * 0.15)
    bottom_height = int(height * keep_bottom_percent)
    total_keep_height = header_height + bottom_height

    # Crop the image: keep full width, but only header + bottom portion
    cropped = img.crop((0, height - total_keep_height, width, height))
    cropped.save(output_path, 'PNG')

  # Clean up temporary file
  os.remove(temp_path)


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

  args = parser.parse_args()
  return args


def process_excel_for_photo(excel_file_path, n_days, index_name):
  """
    Main function to process Excel file and create formatted screenshot
    """
  # Load all data
  df = load_excel_data(excel_file_path)

  # Generate output filename
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
  output_path = f'data/tmp/excel_screenshot_{index_name}_{timestamp}.png'

  # Create and save screenshot
  create_formatted_screenshot(df, n_days, output_path)

  return output_path


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


def main(args):
  excel_file_path = args.file
  output_file = process_excel_for_photo(excel_file_path,
                                        n_days=args.window,
                                        index_name=args.market)
  process_and_send_market_data(output_file, index_name=args.market)
  logger.info(f"Screenshot saved as: {output_file}")


if __name__ == "__main__":
  # Replace with your Excel file path
  args = get_args()
  logger.info(f'args = {args}')
  main(args)
