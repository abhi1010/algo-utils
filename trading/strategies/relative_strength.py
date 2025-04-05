import datetime
import pandas as pd

from trading.common import utils

logger = utils.get_logger()


def create_simple_relative_strength(
    token_name,
    df,
    btc_data,
    ref_date,
    far_date,
    rs_type_name='near',
    btc_ref='btc_far_ref',
    close_column_name='close'):
  if df.empty:
    return None
  first_date = df.iloc[0].name
  logger.info(f'first_date = {first_date}; ref_date = {ref_date}')
  if first_date > far_date:
    logger.info(
        f'<{token_name}> {first_date} > {far_date}. Just released. Moving on')
    return None
  price_on_date = lambda d: d.loc[d.index == ref_date].iloc[0][close_column_name
                                                              ]

  ref_dt_in_index = ref_date in df.index
  if not ref_dt_in_index:
    return None

  token_close_price_on_far_ref_date = price_on_date(df)
  btc_close_price_on_far_ref_date = price_on_date(btc_data)
  df[rs_type_name] = df[close_column_name] / token_close_price_on_far_ref_date
  df[btc_ref] = btc_data[close_column_name] / btc_close_price_on_far_ref_date
  df[rs_type_name + '_rs'] = df[rs_type_name] / df[btc_ref] - 1
  return df


def calculate_relative_strength(
    df,
    close_column_name='Close',
    nifty_lose_column_name='Nifty_Close',
    offset=6):
  # Calculate 'cl-norm' for the close price
  cl_normalized = df[close_column_name] / df[close_column_name].shift(offset)

  # Calculate 'nf-norm' for the Nifty close price
  nifty_normalized = df[nifty_lose_column_name] / df[
      nifty_lose_column_name].shift(offset)

  # Calculate 'rs-5' using 'cl-norm' and 'nf-norm'
  df[f'RS_{offset}'] = cl_normalized / nifty_normalized - 1
  return df
