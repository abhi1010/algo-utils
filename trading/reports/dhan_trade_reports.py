import pandas as pd
from datetime import datetime
import argparse
from enum import Enum

from trading.common import utils

logger = utils.get_logger('dhan-trade-reports')
'''
Usage
  python trading/reports/dhan_trade_reports.py \
     --csv-file-path TRADE_HISTORY_CSV_1101020069_2024-09-11_2024-10-11_0_.csv \
     --output-file-path  data/dhan-202410-trade_summary.xlsx
'''


def business_days_between(start_date, end_date, freq='B'):
  """Calculates the number of business days between two dates.

    Args:
        start_date: The start date.
        end_date: The end date.

    Returns:
        The number of business days between the two dates.  

    """

  # Convert dates to datetime objects
  start_date = pd.to_datetime(start_date)
  end_date = pd.to_datetime(end_date)

  # Create a date range with business days only
  business_days = pd.date_range(start_date, end_date, freq=freq)

  # Calculate the number of business days
  num_business_days = len(business_days)

  return num_business_days


class Broker(str, Enum):
  DHAN = 'dhan'
  TASTY = 'tasty'


class ColumnNamesDhan(str, Enum):
  '''

    ticker = row['Name']
    quantity = row['Quantity/Lot']
    price = row['Trade Price']
    value = price * quantity
    trade_type = row['Buy/Sell']
    trade_date = row['DateTime']
    trade_date_only = pd.to_datetime(row['Date'])
    '''
  Name = 'Name'
  Qty = 'Quantity/Lot'
  Price = 'Trade Price'
  TradeType = 'Buy/Sell'
  TradeDateTime = 'DateTime'
  TradeDateOnly = 'Date'
  TradeTimeOnly = 'Time'
  DtTmFormat = '%Y-%m-%d'


'''we have different column names based on the broker, we need to keep them separate
'''


class ColumnNamesTasty:
  '''
  Date	Type	Sub Type	Action	Symbol	Instrument Type	Description	Value	Quantity	Average Price	Commissions	Fees	Multiplier	Root Symbol	Underlying Symbol	Expiration Date	Strike Price	Call or Put	Order #	Currency
  '''
  Name = 'Symbol'
  Qty = 'Quantity'
  Price = 'Average Price'
  TradeType = 'Action'
  TradeDateTime = 'Date'
  TradeDateOnly = 'Date'
  TradeTimeOnly = 'Time'
  # DtTmFormat = "%Y-%m-%dT%H:%M:%S%z"
  DtTmFormat = "%Y-%m-%d"


def group_tasty_trade_history(df):
  # reverse the order of the rows
  df = df.iloc[::-1]
  # Convert 'Date' column to datetime and extract the date
  df['DateFull'] = pd.to_datetime(df['Date'])
  df['Date'] = pd.to_datetime(df['Date']).dt.date

  # Aggregate trades by action (Buy/Sell) and date
  def aggregate_trades(trades):
    aggregated = (trades.groupby(['Date', 'Symbol', 'Action'],
                                 as_index=False).agg({
                                     'DateFull': 'first',
                                     'Type': 'first',
                                     'Sub Type': 'first',
                                     'Instrument Type': 'first',
                                     'Description': 'first',
                                     'Value': 'sum',
                                     'Quantity': 'sum',
                                     'Commissions': 'sum',
                                     'Fees': 'sum',
                                     'Multiplier': 'first',
                                 }))
    # Recalculate the average price
    aggregated['Average Price'] = aggregated['Value'] / aggregated['Quantity']
    return aggregated

  # Apply aggregation
  aggregated_df = aggregate_trades(df)
  return aggregated_df


def create_trade_summary(broker, csv_file_path):
  # Read the CSV file
  is_broker_dhan = broker == Broker.DHAN
  ColNamesCls = ColumnNamesDhan if is_broker_dhan else ColumnNamesTasty

  df = pd.DataFrame()
  if csv_file_path.endswith('.csv'):
    df = pd.read_csv(csv_file_path)
  elif csv_file_path.endswith('.xlsx'):
    # df = pd.read_excel(csv_file_path, sheet_name='Tasty Jan 2025') # Jan2025
    df = pd.read_excel(csv_file_path, sheet_name='Jan2025')  # Jan2025

    df = df[(df['Instrument Type'] == 'Equity')].copy()
    df = df[df['Type'] == 'Trade'].copy()
    logger.info(f'df 0 = \n{df}')
    df = group_tasty_trade_history(df)
    logger.info(f'df 2 = \n{df}\n; shape={df.shape}')

  logger.info(f'df = \n{df}')

  # Convert Date and Time to datetime
  if is_broker_dhan:
    df[ColNamesCls.TradeDateTime] = pd.to_datetime(
        df[ColNamesCls.TradeDate] + ' ' + df[ColNamesCls.TradeTimeOnly])
  else:
    df[ColNamesCls.TradeDateTime] = pd.to_datetime(
        df[ColNamesCls.TradeDateTime])

  # Sort by DateTime to ensure chronological order
  if is_broker_dhan:
    df = df.sort_values(ColNamesCls.TradeDateTime)
  else:
    df = df.sort_values('DateFull')

  if is_broker_dhan:
    df = df[(df['Order'] == 'Delivery') & (df['Trade Value'] < 9000)]
    logger.info(f'df 0 = \n{df}')

    # filter df by Order as Delivery, and Trade Value < 9000, only accept those
    # df = df[(df['Order'] == 'DELIVERY') & (df['Trade Value'] < 9000)]
    df = df[(df['Order'] == 'DELIVERY')]

  logger.info(f'df 1 = \n{df}')
  # Initialize an empty list to store summarized trades
  summarized_trades = []

  # Dictionary to keep track of open positions
  open_positions = {}

  for _, row in df.iterrows():
    logger.info(f'row = {row}')

    ticker = row[ColNamesCls.Name]
    quantity = row[ColNamesCls.Qty]
    price = row[ColNamesCls.Price]
    value = price * quantity
    trade_type = row[ColNamesCls.TradeType]
    trade_date = row[ColNamesCls.TradeDateTime]

    trade_date_only = pd.to_datetime(row[ColNamesCls.TradeDateOnly])

    if trade_type in ['BUY', 'BUY_TO_OPEN']:
      if ticker not in open_positions:
        open_positions[ticker] = []
      open_positions[ticker].append({
          'quantity': quantity,
          'price': price,
          'date': trade_date_only
      })

    elif trade_type in ['SELL', 'SELL_TO_CLOSE']:
      if ticker in open_positions and open_positions[ticker]:
        while quantity > 0 and open_positions[ticker]:
          buy_trade = open_positions[ticker][0]
          sell_quantity = min(quantity, buy_trade['quantity'])
          entry_px = buy_trade['price'] * (1 if is_broker_dhan else -1)

          profit_loss = (price - entry_px) * sell_quantity
          profit_loss_percent = (price - entry_px) / entry_px * 100
          # Replace the original line with the new function
          days_held = business_days_between(buy_trade['date'],
                                            trade_date_only,
                                            freq='B')

          logger.info(
              f'ticker: {ticker}; trade_date_only={trade_date_only}, '
              f'buy_trade[date]={buy_trade["date"]}'
              f'; type= {type(trade_date_only)}, {type(buy_trade["date"])}'
              f'; days_held={days_held}')

          # days_held = (trade_date_only - buy_trade['date'])

          summarized_trades.append({
              'Name':
              ticker,
              'Strategy':
              "Wknd" if (days_held <= 1 and value > 6000) or (value > 10000) or
              (value > 9000 and quantity > 1) else '',
              'Entry date':
              buy_trade['date'].strftime(ColNamesCls.DtTmFormat),
              'Exit date':
              trade_date.strftime(ColNamesCls.DtTmFormat),
              'Qty':
              sell_quantity,
              '# Entry price':
              entry_px,
              '# Exit price':
              price,
              'Σ PL':
              profit_loss,
              '% PL':
              profit_loss_percent,
              'Status':
              'Closed',
              'Value':
              value,
              'Σ No.of days':
              days_held
          })

          quantity -= sell_quantity
          buy_trade['quantity'] -= sell_quantity
          if buy_trade['quantity'] == 0:
            open_positions[ticker].pop(0)

  # Add remaining open positions to the summary
  for ticker, positions in open_positions.items():
    for position in positions:
      summarized_trades.append({
          'Name':
          ticker,
          'Strategy':
          "",
          'Entry date':
          position['date'].strftime('%Y-%m-%d'),
          'Exit date':
          '',
          'Qty':
          position['quantity'],
          '# Entry price':
          entry_px,
          '# Exit price':
          None,
          'Σ PL':
          None,
          '% PL':
          None,
          'Status':
          'Open',
          'Σ No.of days': (pd.Timestamp.now() - position['date']).days,
          'Value':
          position['price'] * position['quantity']
      })

  # Create DataFrame from summarized trades
  summary = pd.DataFrame(summarized_trades)

  return summary


def get_args():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add args for csv file
  parser.add_argument('--csv-file-path',
                      type=str,
                      default='tasty-trades.xlsx',
                      help='Path to the CSV file')

  # add args for broker
  parser.add_argument('--broker',
                      type=Broker,
                      default=Broker.TASTY,
                      help='Broker name')

  # add args for output excel file
  parser.add_argument('--output-file-path',
                      type=str,
                      default='data/{BROKER}-{DATE}-trade_summary.xlsx',
                      help='Path to the output Excel file')
  args = parser.parse_args()

  return args


def main():

  # Specify the path to your CSV file
  args = get_args()
  logger.info(f'args = {args}')
  csv_file_path = args.csv_file_path

  # Create the summary
  summary = create_trade_summary(args.broker, csv_file_path)

  # Display the summary
  logger.info(f'summary = \n{summary.to_string(index=False)}')
  output_file_path = args.output_file_path.format(
      BROKER=args.broker, DATE=datetime.now().strftime('%Y%m%d'))
  # Save to Excel with proper float formatting
  with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
    summary.to_excel(writer, index=False, sheet_name='Trade Summary')
    workbook = writer.book
    worksheet = writer.sheets['Trade Summary']

  logger.info(
      f"Excel file {output_file_path} has been created with proper formatting."
  )


if __name__ == '__main__':
  main()
