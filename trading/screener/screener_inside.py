import pandas as pd
import numpy as np

from trading.common import utils

logger = utils.get_logger('screener_vol_squeeze')

from trading.common.utils import HLOC

import yfinance as yf


def find_inside_bar_pattern(df):
  """
    Find patterns with:
    1. Initial green candle followed by 5+ inside candles
    2. Initial candle has highest volume
    3. Continuously lower closing prices
    
    Parameters:
    df: pandas DataFrame with HLOC columns
    
    Returns:
    List of indices where the pattern starts
    """
  pattern_indices = []
  min_inside_bars = 3

  # Calculate green candles
  df['is_green'] = df[HLOC.Col_CLOSE] > df[HLOC.Col_Open]

  for i in range(len(df) - min_inside_bars - 1):
    # Check if current candle is green
    if not df['is_green'].iloc[i]:
      continue

    initial_high = df[HLOC.Col_High].iloc[i]
    initial_low = df[HLOC.Col_Low].iloc[i]
    initial_volume = df[HLOC.Col_Volume].iloc[i]
    initial_close = df[HLOC.Col_CLOSE].iloc[i]

    inside_bar_count = 0
    valid_pattern = True
    last_close = initial_close

    # Check subsequent candles
    for j in range(i + 1, len(df)):
      current_high = df[HLOC.Col_High].iloc[j]
      current_low = df[HLOC.Col_Low].iloc[j]
      current_volume = df[HLOC.Col_Volume].iloc[j]
      current_close = df[HLOC.Col_CLOSE].iloc[j]

      # logger.info(
      #     f'current_close: {current_close}, last_close: {last_close}; '
      #     f'initial_high = {initial_high}; initial_low = {initial_low}; ')

      # Check if it's an inside candle
      is_inside = (current_high <= initial_high and current_low >= initial_low)

      # Check if volume is lower than initial
      volume_valid = current_volume < initial_volume

      # Check if close is lower than previous
      price_dropping = current_close < last_close

      if not (is_inside and volume_valid and price_dropping):
        break

      inside_bar_count += 1
      last_close = current_close

    # If we found enough inside bars, save the pattern start index
    if inside_bar_count >= min_inside_bars:
      pattern_indices.append(
          {
              'start_idx': i,
              'date_begin': df.index[i],
              'num_inside_bars': inside_bar_count,
              'initial_volume': initial_volume,
              'initial_close': initial_close,
              'final_close': last_close
          })

  return pattern_indices


def print_pattern_details(df, patterns):
  """
    Print details about found patterns
    """
  for p in patterns:
    print(f"\nPattern found starting at index {p['start_idx']}:")
    print(f"Number of inside bars: {p['num_inside_bars']}")
    print(f"Initial volume: {p['initial_volume']}")
    print(f'date_begin = {p["date_begin"]}')
    print(f"Price movement: {p['initial_close']} -> {p['final_close']}")
    print(
        f"Price drop: {((p['final_close'] - p['initial_close']) / p['initial_close'] * 100):.2f}%"
    )


def main():
  # Example usage:
  """
  # Assuming df is your DataFrame with OHLCV data:
  df = pd.DataFrame({
      'open': [...],
      'high': [...],
      'low': [...],
      'close': [...],
      'volume': [...]
  })

  patterns = find_inside_bar_pattern(df)
  print_pattern_details(df, patterns)
  """
  df = yf.download(
      'PLTR',
      start='2010-01-01',
      end='2024-01-31',
      progress=False,
      multi_level_index=False)

  patterns = find_inside_bar_pattern(df)
  print_pattern_details(df, patterns)


if __name__ == '__main__':
  main()
