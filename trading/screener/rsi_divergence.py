import yfinance as yf
import pandas as pd
import numpy as np
import argparse
from enum import StrEnum
import sys

from trading.common import utils

logger = utils.get_logger('screener-rsi', False)

import pandas as pd
import numpy as np
import talib
import pandas_ta as ta
from pprint import pformat

from trading.screener import bollinger_band_width_percentile as bbwp
from trading.screener.screener_common import Markets
from trading.services import telegram_runner
from trading.common import utils

from collections import deque


# create an enum fo strings Hidden_Bearish, Hidden_Bullish, Regular_Bearish, Regular_Bullish
class DivergenceType(StrEnum):
  Hidden_Bearish = "Hidden_Bearish"
  Hidden_Bullish = "Hidden_Bullish"
  Regular_Bearish = "Regular_Bearish"
  Regular_Bullish = "Regular_Bullish"


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--dir",
                      default='data/nse-bbwp-screener',
                      help="Which directory to save to",
                      type=str)

  # arg called output_dir, default to be data/rsi
  parser.add_argument('--output-dir', '-o', default='data/rsi', type=str)

  # add tag as rsi screener
  parser.add_argument('--tag', default='RSI Divergence', type=str)

  parser.add_argument('--timeframe',
                      '-t',
                      default='1d',
                      type=str,
                      help='What timeframe/interval to use')

  # add arg called window, with a default of 20
  parser.add_argument('--window', '-w', default=100, type=int, help='Window')

  parser.add_argument('--market',
                      type=Markets,
                      default=Markets.NIFTY,
                      help='What market')

  args = parser.parse_args()
  return args


def is_pivot_low(data, idx, leftbars, rightbars):
  """
    Check if the current bar is a pivot low.

    Parameters:
    data: pd.Series - The series of low prices.
    idx: int - The index of the current bar.
    leftbars: int - Number of bars to the left to check.
    rightbars: int - Number of bars to the right to check.

    Returns:
    bool - True if pivot low, False otherwise.
    """
  if idx - leftbars < 0 or idx + rightbars >= len(data):
    return False

  current_low = data[idx]
  left_part = data[idx - leftbars:idx]
  right_part = data[idx + 1:idx + 1 + rightbars]

  return all(current_low < left_part) and all(current_low < right_part)


def pivotlow(data, leftbars, rightbars):
  """
    Calculate pivot low for the given series.

    Parameters:
    data: pd.Series - The series of low prices.
    leftbars: int - Number of bars to the left to check.
    rightbars: int - Number of bars to the right to check.

    Returns:
    pd.Series - Series containing pivot low points or NaN.
    """
  pivot_lows = [np.nan] * len(data)

  for idx in range(len(data)):
    if is_pivot_low(data, idx, leftbars, rightbars):
      pivot_lows[idx] = data[idx]

  return pd.Series(pivot_lows, index=data.index)


def calculate_low_lbr_new_1(df):
  # Create a new column initialized with NaN
  df['low_lbr_new_1'] = np.nan

  # Find indices where PL is not NaN
  pl_indices = df.index[df['Pivot Low'].notna()].tolist()

  # Iterate through PL indices
  for i in range(1, len(pl_indices)):
    current_index = pl_indices[i]
    previous_index = pl_indices[i - 1]

    # Assign the low_lbr value from the previous PL occurrence
    df.loc[current_index, 'low_lbr_new_1'] = df.loc[previous_index,
                                                    'low_lbr_new_0']

  return df


def check_divergence_last_5(df, divergence_column):
  # Get the last 5 rows of the DataFrame
  last_5_rows = df.tail(5)

  # Check if any of these rows have HiddenBullish as True
  has_hidden_bullish = last_5_rows[divergence_column].any()

  return has_hidden_bullish


def rsi_divergence(data,
                   rsi_period=14,
                   pivot_lookback_right=5,
                   pivot_lookback_left=5,
                   range_upper=60,
                   range_lower=5,
                   enable_bullish=True,
                   enable_hidden_bullish=True,
                   enable_bearish=True,
                   enable_hidden_bearish=True):
  """
    Calculates the RSI divergence signals for a given DataFrame.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing the stock data (e.g., 'Open', 'High', 'Low', 'Close').
    rsi_period (int): The period for the RSI calculation (default is 14).
    pivot_lookback_right (int): The number of bars to look back on the right side of the pivot (default is 5).
    pivot_lookback_left (int): The number of bars to look back on the left side of the pivot (default is 5).
    range_upper (int): The upper bound of the pivot range (default is 60).
    range_lower (int): The lower bound of the pivot range (default is 5).
    plot_bullish (bool): Whether to plot the regular bullish divergence (default is True).
    plot_hidden_bullish (bool): Whether to plot the hidden bullish divergence (default is True).
    plot_bearish (bool): Whether to plot the regular bearish divergence (default is True).
    plot_hidden_bearish (bool): Whether to plot the hidden bearish divergence (default is True).

    Ref: https://stackoverflow.com/questions/64019553/how-pivothigh-and-pivotlow-function-work-on-tradingview-pinescript

    Returns:
    pandas.DataFrame: The input DataFrame with the divergence signals added as new columns.
    """
  data['RSI'] = ta_rsi = talib.RSI(data['Close'], rsi_period)

  # data['RSI'] = data['rsi']

  def in_range(condition):
    bars = data.index.get_loc(
        condition.last_valid_index()) - data.index.get_loc(
            condition.shift(1).last_valid_index())
    return range_lower <= bars <= range_upper

  pivot_calculated = pivotlow(data['RSI'], pivot_lookback_left,
                              pivot_lookback_right)

  # data['InRange2'] = in_range('Pivot Low')
  # data['PP'] = pivot_calculated
  # logger.info(f'pivot_calculated = {pivot_calculated}')

  # -------- Regular Bullish
  # set "Pivot Low" to "PLV" if the RSI value from 5 days ago is the same as PLV
  data['Pivot Low'] = pivot_calculated.shift(pivot_lookback_left).fillna(
      np.nan)
  data = update_df_with_pl_distances(data)
  # data['Pivot Low'] = pivots['PLV']

  # data['Pivot High'] = pivots['PHV']

  # data['Px_LL'] = data['Low'].shift(pivot_lookback_right) < data['Low'].shift(
  #     pivot_lookback_right).where(data['Pivot Low'].notna(), np.nan)
  # data[DivergenceType.Regular_Bullish] = enable_bullish & data['RSI_HL'] & data[
  #     'Px_LL'] & data['Pivot Low'].notna()

  # -------- Hidden Bullish

  data['Prev_PL'] = data['Pivot Low'].shift(1).ffill()

  # data['RSI_LL'] = data['RSI'].shift(pivot_lookback_right) < data['RSI'].shift(
  #     pivot_lookback_right).where(data['Pivot Low'].notna(), np.nan)

  data['RSI_HL'] = ((data['Pivot Low'] > data['Prev_PL']) &
                    (data['Distance'] > pivot_lookback_left)
                    & data['Pivot Low'].notna())

  data['RSI_LL'] = ((data['Pivot Low'] < data['Prev_PL']) &
                    (data['Distance'] > pivot_lookback_left)
                    & data['Pivot Low'].notna())
  # data['rsi_lbr_2'] = data['RSI'].shift(pivot_lookback_right)
  data['low_lbr_new_0'] = data['Low'].shift(pivot_lookback_right)
  data = calculate_low_lbr_new_1(data)
  data['Pivot Low'].shift(1).ffill()

  data['Px_HL'] = ((data['low_lbr_new_0'] > data['low_lbr_new_1']) &
                   (data['Distance'] > pivot_lookback_left)
                   & data['Pivot Low'].notna())
  data['Px_LL'] = ((data['low_lbr_new_0'] < data['low_lbr_new_1']) &
                   (data['Distance'] > pivot_lookback_left)
                   & data['Pivot Low'].notna())
  # hidden bullish = priceHL and rsiLL and plFound
  data[DivergenceType.Hidden_Bullish] = enable_hidden_bullish & data[
      'RSI_LL'] & data['Px_HL'] & data['Pivot Low'].notna()
  data[DivergenceType.Regular_Bullish] = enable_hidden_bullish & data[
      'RSI_HL'] & data['Px_LL'] & data['Pivot Low'].notna()

  # -------- Regular Bearish

  # data['Pivot High'] = data['RSI'].rolling(
  #     pivot_lookback_left + pivot_lookback_right + 1).apply(
  #         lambda x: np.argmax(x) - pivot_lookback_left
  #         if np.max(x) > 30 else np.nan,
  #         raw=True)
  # data['Osc Lower High'] = data['RSI'].shift(
  #     pivot_lookback_right) < data['RSI'].shift(pivot_lookback_right + 1).where(
  #         data['Pivot High'].notna(), np.nan)
  # data['Price Higher High'] = data['High'].shift(
  #     pivot_lookback_right) > data['High'].shift(pivot_lookback_right +
  #                                                1).where(
  #                                                    data['Pivot High'].notna(),
  #                                                    np.nan)
  # data[DivergenceType
  #      .Regular_Bearish] = enable_bearish & data['Osc Lower High'] & data[
  #          'Price Higher High'] & data['Pivot High'].notna()

  # -------- Hidden Bearish
  # data['Osc Higher High'] = data['RSI'].shift(
  #     pivot_lookback_right) > data['RSI'].shift(pivot_lookback_right + 1).where(
  #         data['Pivot High'].notna(), np.nan)
  # data['Price Lower High'] = data['High'].shift(
  #     pivot_lookback_right) < data['High'].shift(pivot_lookback_right +
  #                                                1).where(
  #                                                    data['Pivot High'].notna(),
  #                                                    np.nan)
  # data[DivergenceType.
  #      Hidden_Bearish] = enable_hidden_bearish & data['Osc Higher High'] & data[
  #          'Price Lower High'] & data['Pivot High'].notna()

  return data


def run_rsi_stats(dataframes, window, threshold=0.1):
  cols_to_copy = [
      'Date',
      # 'PL',
      # 'plFound',
      # 'rsiLL',
      'RSI_LL',
      # 'priceHL',
      'Px_HL',
      # 'is_inRange',
      'Close',
      'Low',
      # 'rsi',
      # 'ta_rsi_lbr',
      # 'rsi_lbr',
      # 'low_lbr',
      # 'low_lbr_new_0',
      # 'low_lbr_1',
      # 'low_lbr_new_1',
      # 'bars_cnt',
      # 'is_inRange',
      # 'InRange2',
      'RSI',
      # 'rsi_lbr_2',
      'Distance',
      # 'Adj Close',
      # 'RSI1',
      # 'PLV',
      'Pivot Low',
      'Prev_PL',
      # 'Px_LL',
      # 'PL',
      # DivergenceType.Hidden_Bearish,
      DivergenceType.Hidden_Bullish,
      # DivergenceType.Regular_Bearish,
      DivergenceType.Regular_Bullish,
  ]
  hidden_bullish_tickers = []
  reg_bullish_tickers = []
  for symbol, df in dataframes.items():
    # logger.info(f'Sym: {symbol}; init = \n{df}; shape={df.shape}')
    if df.shape[0] < window or df.shape[1] < 5:
      logger.info(f'Sym: {symbol} shape is too small, skipping: {df.shape}')
      continue

    sym_rsi = rsi_divergence(df)[cols_to_copy]

    sma_50 = ta.sma(df['Close'], length=50)

    # logger.info(f'sym  = {symbol}; sma_50 = {sma_50}; ')
    if df['Close'].iloc[-1] >= sma_50.iloc[-1] * 0.97:
      logger.info(f'Sym: {symbol} under SMA 50. '
                  f'sma={sma_50.iloc[-1]}; close={df["Close"].iloc[-1]}')
    else:
      logger.info(f'Sym: {symbol} above SMA 50. '
                  f'sma={sma_50.iloc[-1]}; close={df["Close"].iloc[-1]}')
      continue

    logger.info(f'Sym: {symbol}; sym_rsi = \n{sym_rsi}')

    # Call the function
    is_hidden_bullish = check_divergence_last_5(sym_rsi,
                                                DivergenceType.Hidden_Bullish)
    is_reg_bullish = check_divergence_last_5(sym_rsi,
                                             DivergenceType.Regular_Bullish)

    logger.info(
        f"[{symbol}]:: Does contain HiddenBullish as True? {is_hidden_bullish}"
    )

    logger.info(
        f"[{symbol}]:: Does contain is_reg_bullish as True? {is_reg_bullish}")

    if is_hidden_bullish:
      hidden_bullish_tickers.append(symbol)
    if is_reg_bullish:
      reg_bullish_tickers.append(symbol)
  return hidden_bullish_tickers, reg_bullish_tickers


def main():
  utils.set_pandas_options()
  args = get_args()
  logger.info(f'args = {args}')
  # df = pd.read_excel('~/code/vg2.xlsx')

  # Convert the 'timestamp' column to date
  # df['Date'] = pd.to_datetime(df['timestamp']).dt.date
  # df['Date'] = pd.to_datetime(df['timestamp'])

  # Set 'timestamp' column as the index
  # df.set_index('Date', inplace=False)
  # logger.info(f'df = \n{df}')

  files_df = bbwp.process_files(args.dir, only_one=False)
  dfs, largest_date = bbwp.filter_away_stale_data(files_df, args.market,
                                                  args.timeframe)
  hidden_bullish_tickers, reg_bullish_tickers = run_rsi_stats(
      dfs, window=args.window)

  rsi_divergent_tickers = list(
      set(hidden_bullish_tickers + reg_bullish_tickers))
  logger.info(f'rsi_divergent_tickers = {rsi_divergent_tickers}')

  tickers_for_tradingview = bbwp.transform_tickers(rsi_divergent_tickers,
                                                   args.market)

  logger.info(f'tickers_for_tradingview = {tickers_for_tradingview}')

  bbwp.save_names_to_txt(tickers_for_tradingview, args.output_dir, args.market,
                         args.timeframe, largest_date)
  new_tickers = bbwp.get_new_tickers_compared_to_older_file(
      tickers_for_tradingview, args.output_dir, args.market, args.timeframe,
      largest_date)

  logger.info(f'new_tickers = {new_tickers}')

  send_telegram_msg(args, tickers_for_tradingview, new_tickers,
                    hidden_bullish_tickers, reg_bullish_tickers, args.market)


def send_telegram_msg(args, tickers_for_telegram, new_tickers,
                      hidden_bullish_tickers, reg_bullish_tickers, market):
  tickers_for_telegram = sorted(tickers_for_telegram)
  new_tickers = sorted(new_tickers)
  msg = ''
  if len(tickers_for_telegram) == 0:
    msg = f"Args={args} \n No tickers right now"
    return

  # Convert Namespace to dictionary
  namespace_dict = vars(args)
  # Pretty print the dictionary into a string
  namespace_str = pformat(namespace_dict)
  prefix_joiner = ',NSE.' if market == Markets.NIFTY else ','

  if market == Markets.NIFTY:
    nse_prefix = 'NSE:'
    prefixer_style = lambda tickers: (nse_prefix + (',' + nse_prefix).join(
        tickers)) if len(tickers) else ''
  else:
    prefixer_style = lambda tickers: ','.join(tickers)

  new_tickers_s = prefixer_style(new_tickers)
  all_tickers_s = prefixer_style(tickers_for_telegram)
  hidden_bullish_tickers_s = prefixer_style(hidden_bullish_tickers)
  reg_bullish_tickers_s = prefixer_style(reg_bullish_tickers)

  #   msg += f'\n\n All tickers: {all_tickers_s}'
  msg = f'''Args:
```python
{namespace_str}```

New tickers: ```
{new_tickers_s}```

Hidden Bullish: ```
{hidden_bullish_tickers_s}```

Reg Bullish: ```
{reg_bullish_tickers_s}``` '''
  bbwp._send_msg(msg)


def update_df_with_pl_distances(df, col_to_use='Pivot Low'):
  # Add new columns
  df['Distance'] = np.nan

  # Find indices of valid PL values
  valid_pl_indices = df.index[df[col_to_use].notna()].tolist()

  if len(valid_pl_indices) < 2:
    print("Not enough valid PL values to calculate distance.")
    return df

  for i in range(1, len(valid_pl_indices)):
    current_index = valid_pl_indices[i]
    previous_index = valid_pl_indices[i - 1]

    rows_between = current_index - previous_index

    # Update the current row with the calculations
    df.loc[current_index, 'Distance'] = rows_between

    # Fill in Rows_Between for rows in between
    df.loc[previous_index+1:current_index-1, 'Rows_Between'] = \
        range(1, rows_between)

  return df


def find_prev_pivots(df):

  # Sample DataFrame
  # data = {
  #     'PL':
  #         [
  #             np.nan, np.nan, np.nan, 36.63, np.nan, np.nan, 46.21, np.nan,
  #             np.nan, 43.72, np.nan, np.nan, 57.56, np.nan
  #         ],
  #     # 'Prev_PL':
  #     #     [
  #     #         35.64, 35.64, 35.64, 40.82, 40.82, 40.82, 36.63, 36.63, 36.63,
  #     #         46.21, 46.21, 46.21, 43.72, 43.72
  #     #     ]
  # }

  # df = pd.DataFrame(data)

  # Function to calculate previous pivot lows
  # def update_prev_pl(df):
  #   prev_pl = None
  #   for idx in range(len(df)):
  #     if not np.isnan(df.at[idx, 'PL']):
  #       prev_pl = df.at[
  #           idx, 'Prev_PL'] = prev_pl if prev_pl is not None else df.at[idx,
  #                                                                       'PL']
  #     else:
  #       df.at[idx, 'Prev_PL'] = prev_pl
  #   return df

  # Update the Prev_PL column based on the PL column
  df['Prev_PL'] = df['Pivot Low'].shift(1).ffill()
  logger.info(f'df = {df}')

  # Display the updated DataFrame
  return df


if __name__ == '__main__':
  main()
  # testme()
