import argparse
import os
from enum import Enum

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import csv

from trading.stocks import dhan

from trading.common import utils
from trading.common.google_api_helper import GoogleSheetsApi

logger = utils.get_logger('dhan-trade-charts')
'''
Usage
  python trading/reports/trade_analysis_chart.py --source excel
  python trading/reports/trade_analysis_chart.py --source sheets
'''
RENAMES = {
    'AVANTI': 'AVANTIFEED',
    'JG Chemicals': 'JGCHEM',
    'VRAJ IRON AND STEEL': 'VRAJ',
    'KP Energy': 'KPEL',
}


# Download OHLC data from yfinance
def download_ohlc_data(broker, ticker, orig_sym, start_date, end_date):
  is_broker_dhan = broker == Broker.DHAN
  try:
    # ticker_to_use = renames[ticker]
    start_date = start_date - pd.DateOffset(days=90)
    end_date = end_date + pd.DateOffset(days=30)

    def attempt_download(ticker, start_date, end_date):
      ticker_name = f'{ticker}.NS' if is_broker_dhan else f'{ticker}'
      ticker_name_bo = f'{ticker}.BO' if is_broker_dhan else f'{ticker}'

      data = yf.download(ticker_name,
                         start=start_date,
                         end=end_date,
                         progress=False,
                         auto_adjust=True,
                         multi_level_index=False)
      logger.info(f'ticker: {ticker_name}; data = \n{data.shape}')

      if data.empty and isbroker_dhan:
        data = yf.download(ticker_name_bo,
                           start=start_date,
                           end=end_date,
                           progress=False,
                           auto_adjust=True,
                           multi_level_index=False)
        logger.info(f'ticker: {ticker_bo}; data = \n{data.shape}')
      return data

    try:
      if ticker:
        data = attempt_download(ticker, start_date, end_date)
      else:
        data = pd.dataframe()
      if data.empty and ticker in RENAMES:
        new_ticker = RENAMES[ticker]
        data = attempt_download(new_ticker, start_date, end_date)

    except Exception as e:
      logger.info(f"Error downloading data for {ticker}: {str(e)}")
      simple_ticker_name = f'{ticker}.BO' if is_broker_dhan else f'{ticker}'
      data = yf.download(simple_ticker_name,
                         start=start_date,
                         end=end_date,
                         progress=False,
                         auto_adjust=True,
                         multi_level_index=False)
      logger.info(f'ticker: {simple_ticker_name}; data = \n{data.shape}')
      logger.info(
          f'Using BO instead of NS for {ticker}. data = \n{data.shape}')
    return data
  except Exception as e:
    logger.info(f"Error downloading data for {ticker}: {str(e)}")
    return None


def hollow_candlesticks(ticker, ohlc: pd.DataFrame, trd_analysis_df,
                        price_bins: pd.DataFrame, start_date,
                        end_date) -> go.Figure:
  # strategy_name =
  start_date_yyyy_mm_dd, end_date_yyyy_mm_dd = get_dates_in_yyyy_mm_dd_format(
      start_date, end_date)
  fig_name = f"{ticker} --> Start: {start_date_yyyy_mm_dd}, End: {end_date_yyyy_mm_dd}"

  fig = make_subplots(
      rows=2,
      cols=2,
      shared_xaxes="columns",
      shared_yaxes="rows",
      column_width=[0.8, 0.2],
      row_heights=[0.8, 0.2],
      horizontal_spacing=0,
      vertical_spacing=0,
      subplot_titles=["Candlestick", "Price Bins", "Volume", ""])
  showlegend = True
  for index, row in ohlc.iterrows():
    color = dict(fillcolor=row["fill"], line=dict(color=row["color"]))
    fig.add_trace(go.Candlestick(
        x=[index],
        open=[row["Open"]],
        high=[row["High"]],
        low=[row["Low"]],
        close=[row["Close"]],
        increasing=color,
        decreasing=color,
        showlegend=showlegend,
        name="GE",
        legendgroup=fig_name,
    ),
                  row=1,
                  col=1)
    showlegend = False
  fig.add_trace(
      go.Bar(x=ohlc.index,
             y=ohlc["Volume"],
             text=ohlc["Percentage"],
             marker_line_color=ohlc["color"],
             marker_color=ohlc["fill"],
             name="Volume",
             texttemplate="%{text:.2f}%",
             hoverinfo="x+y",
             textfont=dict(color="white")),
      col=1,
      row=2,
  )
  fig.add_trace(
      go.Bar(y=price_bins["Close"],
             x=price_bins["Volume"],
             text=price_bins["Percentage"],
             name="Price Bins",
             orientation="h",
             marker_color="yellow",
             texttemplate="%{text:.2f}% @ %{y}",
             hoverinfo="x+y"),
      col=2,
      row=1,
  )

  # Buy/Sell markers
  buy_markers = trd_analysis_df[['Entry date', '# Entry price']]
  sell_markers = trd_analysis_df[['Exit date', '# Exit price']]
  buy_markers = buy_markers.rename(columns={
      'Entry date': 'Date',
      '# Entry price': 'Price'
  })
  sell_markers = sell_markers.rename(columns={
      'Exit date': 'Date',
      '# Exit price': 'Price'
  })

  fig.add_trace(go.Scatter(
      x=buy_markers['Date'],
      y=buy_markers['Price'],
      mode='markers',
      marker_color='white',
      marker_symbol="star-triangle-up",
      name='Buy',
      marker=dict(size=12, line=dict(width=2, color='white')),
  ),
                row=1,
                col=1)
  fig.add_trace(go.Scatter(x=sell_markers['Date'],
                           y=sell_markers['Price'],
                           mode='markers',
                           marker_color='black',
                           marker_symbol="star-triangle-down",
                           marker=dict(size=12,
                                       line=dict(width=2, color='white')),
                           name='Sell'),
                row=1,
                col=1)
  fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])],
                   rangeslider_visible=False,
                   col=1)
  fig.update_xaxes(showticklabels=True,
                   showspikes=True,
                   showgrid=True,
                   col=2,
                   row=1)
  fig.update_layout(template="plotly_dark",
                    hovermode="x unified",
                    title_text=fig_name)
  # fig.show("browser")
  return fig


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add args for broker
  parser.add_argument('--broker',
                      type=Broker,
                      default=Broker.TASTY,
                      help='Broker name')

  # add args for directory to save into
  parser.add_argument(
      '--source',
      type=str,
      default='excel',
      help='Which directory to save to',
  )
  # add args for directory to save into
  parser.add_argument(
      '--dir',
      type=str,
      default='data/images/reports/',
      help='Which directory to save to',
  )
  parser.add_argument('--excel-path',
                      type=str,
                      default='data/Broker.TASTY-20250111-trade_summary.xlsx',
                      help='Path to the excel file')

  args = parser.parse_args()

  return args


def get_dates_in_yyyy_mm_dd_format(start_date, end_date):

  start_date_yyyy_mm_dd = start_date.strftime('%Y-%m-%d')
  end_date_yyyy_mm_dd = end_date.strftime('%Y-%m-%d')
  return start_date_yyyy_mm_dd, end_date_yyyy_mm_dd


def get_file_path_name(dir_for_images, ticker, start_date, end_date):

  start_date_yyyy_mm_dd, end_date_yyyy_mm_dd = get_dates_in_yyyy_mm_dd_format(
      start_date, end_date)
  ticker_s = ticker.replace(' ', '_')
  file_path_name = f'{dir_for_images}/{start_date_yyyy_mm_dd}--{ticker_s}--{end_date_yyyy_mm_dd}.png'
  return file_path_name, ticker_s


# Main function
def save_fig_to_image(file_path_name, fig):
  # save as image
  fig.write_image(
      scale=1,
      width=1800,
      height=1200,
      # scale=4,
      engine='kaleido',
      file=file_path_name)


def match_ticker_with_dhan_symbol(ticker, dhan_scrips_list):

  def get_mapped_scrip(symbol):
    for dhan_scrip in dhan_scrips_list:
      if symbol.lower() in dhan_scrip['description'].lower():
        return dhan_scrip['symbol']

  mapped_scrip = get_mapped_scrip(ticker)
  if not mapped_scrip:
    last_word_removed_ticker = remove_last_word(ticker)
    mapped_scrip = get_mapped_scrip(last_word_removed_ticker)
  return mapped_scrip


def remove_last_word(line):
  """Removes the last word from a given line.

  Args:
    line: The input line.

  Returns:
    The line with the last word removed.
  """

  words = line.split()
  if len(words) > 0:
    words = words[:-1]
  return ' '.join(words)


def create_trade_analysis_charts():
  args = get_args()

  # check and create dir if it doesnt exist
  if not os.path.exists(args.dir):
    os.makedirs(args.dir)

  logger.info(f'Args = {args}')
  if args.source == 'excel':
    trd_analysis_df = pd.read_excel(args.excel_path)
  else:
    trd_analysis_df = read_sheet()
  logger.info(f'data 1 = \n{trd_analysis_df}')
  dhan_scrips = dhan.get_dhan_scrips_as_list_with_info(exchange_to_use='BSE')
  # for scrip in dhan_scrips:
  #   logger.info(f'scrip = {scrip}')

  counter = 0
  if trd_analysis_df is not None:
    for index, row in trd_analysis_df.iterrows():
      counter += 1
      # if counter > 5:
      #   return

      ticker = row['Name']
      if args.broker == Broker.DHAN:
        exch_symbol = match_ticker_with_dhan_symbol(ticker, dhan_scrips)
      else:
        exch_symbol = ticker
      start_date = pd.to_datetime(row['Entry date'])
      end_date = pd.to_datetime(row['Exit date'])

      file_path_name, ticker_s = get_file_path_name(args.dir, ticker,
                                                    start_date, end_date)
      if os.path.exists(file_path_name):
        logger.info(
            f'ticker_s = {ticker_s}; '
            f'Skipping as file found: file_path_name = {file_path_name}; ')
        continue
      logger.info(f'start_date = {start_date}; end_date = {end_date}')

      ohlc_data, price_bins = load_data(args.broker, ticker, exch_symbol,
                                        start_date, end_date)
      logger.info(f'ticker: {ticker}; ohlc_data = \n{ohlc_data}')
      # logger.info(f'ticker: {ticker}; price_bins = \n{price_bins}')

      if ohlc_data is not None:
        logger.info(
            f'file_path_name = {file_path_name}; ticker_s = {ticker_s}')

        fig = hollow_candlesticks(
            ticker, ohlc_data,
            trd_analysis_df[trd_analysis_df['Name'] == ticker], price_bins,
            start_date, end_date)

        save_fig_to_image(file_path_name, fig)

  # create_candlestick_chart(
  #     ohlc_data, data[data['Name'] == ticker], ticker, start_date,
  #     end_date, args.dir)


def load_data(broker, ticker, exch_symbol, start_date, end_date):
  # ohlc = web.DataReader("GE", "yahoo", start='2024-01-01', end='2024-10-01')
  logger.info(f'load_data: ticker = {ticker}; exch_symbol = {exch_symbol}; '
              f'start_date = {start_date}; end_date = {end_date}')
  ohlc = download_ohlc_data(broker, exch_symbol, ticker, start_date, end_date)
  # ohlc = yf.download(
  #     tickers=ticker, period="1d", start="2024-01-01", end="2024-10-01")
  ohlc["previousClose"] = ohlc["Close"].shift(1)
  ohlc["color"] = np.where(ohlc["Close"] > ohlc["previousClose"], "green",
                           "red")
  ohlc["fill"] = np.where(ohlc["Close"] > ohlc["Open"], "rgba(255, 0, 0, 0)",
                          ohlc["color"])
  ohlc["Percentage"] = ohlc["Volume"] * 100 / ohlc["Volume"].sum()
  price_bins = ohlc.copy()
  price_bins["Close"] = price_bins["Close"].round()
  price_bins = price_bins.groupby("Close", as_index=False)["Volume"].sum()
  price_bins[
      "Percentage"] = price_bins["Volume"] * 100 / price_bins["Volume"].sum()
  return ohlc, price_bins


def read_sheet():
  gapi = GoogleSheetsApi()
  df = gapi.read_sheet_as_df('1LQDvYuV87AJzgik-jBO9ZO5pjpNXfd5LvSoa9Vy6Knk',
                             range_name='Raw!A1:Z200')

  # filter out all rows where Strategy = "Wknd"
  df = df[df['Strategy'] != 'Wknd']
  # change "# Entry price" column to float type

  df['# Entry price'] = df['# Entry price'].replace(',', '',
                                                    regex=True).astype(float)
  df['# Exit price'] = df['# Exit price'].replace(',', '',
                                                  regex=True).astype(float)

  # change Entry date to datetime type
  df['Entry date'] = pd.to_datetime(df['Entry date'], format='mixed')

  # change Exit date to datetime type
  df['Exit date'] = pd.to_datetime(df['Exit date'], format='mixed')

  logger.info(f'sheet info = {df.info()}')
  # format_dt = '%Y-%m-%d'
  # df['Entry date'] = pd.to_datetime(
  #     df['Entry date'].astype(str), format="mixed")
  # df['Exit date'] = pd.to_datetime(df['Exit date'].astype(str), format="mixed")

  # df['epoch'] = df['Datetime'].apply(lambda x: int(x.timestamp() * 1000))
  return df


class Broker(str, Enum):
  DHAN = 'dhan'
  TASTY = 'tasty'


# Print the DataFrame
if __name__ == '__main__':
  # read_sheet()
  # dl()
  utils.set_pandas_options()
  create_trade_analysis_charts()
