import yfinance as yf
from finviz.screener import Screener
from dateutil.parser import *
from datetime import date
from datetime import datetime, timedelta
import pandas as pd
import pandas_ta as ta
from prettytable import PrettyTable

from trading.common import utils

now_s = datetime.now().strftime("us_automated_ta__%Y-%m-%d__%H-%M-%S")

pd.options.display.float_format = '{:.2f}'.format
logger = utils.get_logger(now_s)

from trading.options.option_utils import OptionHolder

# https://www.reddit.com/r/algotrading/comments/ldkt1z/options_trading_with_automated_ta/
# Summary: https://docs.google.com/document/d/1sm-fb_Jq-bF1UMkwflFBI5_ZN6WqfIJzaGeLVrGl4uM/edit
'''

Dont forget to use
  pip install git+https://github.com/abhi1010/finviz


 Few notes for myself
 1. Buy Today, Sell Tomorrow
 2. Buy >= 6 BS score
 3. Conditions for BS score
    RSI <=40 (14 days)
    Volume >=100
    Filled price <= Lower Bollinger band (5 days, with 2 SD)
    SMA ( 5 days) <= VWAP
    Spread >=0.05 (This might change in future) (Use historical spread perhaps for finding out)
    Filled price = Current Bid
    IV<=40
    Today gain <= 0
4. One interesting comment from fellow reader

    VWAP is not supposed to be calculated on daily data. This means you also need to pull 1min or 2min data from Yahoo (depending on your range) for each stock to calculate it properly.
    You don't seem to have the right settings for some of the TA indicators (e.g., RSI, SMA) based on what DJ has posted here.
    BBand Lower is supposed to be compared to the last filled price in the B-Score calc, not the prior close price.
    You're definitely not supposed to be summing the Volume numbers across several days. If you want to factor in more than one day I guess you could average, but I am pretty sure DJ is just taking the volume from either the prior day or the current day (probably the latter).
    The spread between his ideal buy/sell was on average 68% of the ask-bid spread from the data I examined, so using 60% for the value (as you do in your function) is a bit off. There is also contract-to-contract variation in how the spread is allocated that I have not been able to model, but this variation may not matter much given its small effect size.


    Similarly, you'll probably want to get current contract data (e.g., bid, ask, last, etc.) from your broker, rather than Yahoo since that is who you will be buying from.
    Finally, you do not seem to be filtering out options contracts with incomplete data (e.g., having no values for Bid, Ask, or IV), which needs to be done for the tests to apply properly.
5. Ignore everything with insider trading

'''

# class OptionHolder:

#   def __init__(self):
#     self.Ticker = ""
#     self.Strike = 0.0
#     self.Expiry = None
#     self.YahooString = ""
#     self.IsCall = False

#     self.BidPrice = 0.0
#     self.AskPrice = 0.0
#     self.FilledPrice = 0.0
#     self.FilledAmount = 0.0
#     self.CurrentSpread = 0.0
#     self.Volume = 0
#     self.OpenInterest = 0
#     self.PercentChange = 0.0
#     self.IV = 0.0
#     self.VWAP = 0.0

#     self.IdealBuy = 0.0
#     self.IdealSell = 0.0

#     self.BBandHigh = 0.0
#     self.BBandLow = 0.0
#     self.RSI = 0.0
#     self.SMA = 0.0
#     self.BuyScore = 0

#     self.HistoricData = None

#   def __str__(self):
#     return (
#         f"OptionHolder(Ticker={self.Ticker}, Strike={self.Strike}, Expiry={self.Expiry}, "
#         f"YahooString={self.YahooString}, IsCall={self.IsCall}, BidPrice={self.BidPrice}, "
#         f"AskPrice={self.AskPrice}, FilledPrice={self.FilledPrice}, FilledAmount={self.FilledAmount}, "
#         f"CurrentSpread={self.CurrentSpread}, Volume={self.Volume}, OpenInterest={self.OpenInterest}, "
#         f"PercentChange={self.PercentChange}, IV={self.IV}, VWAP={self.VWAP}, "
#         f"IdealBuy={self.IdealBuy}, IdealSell={self.IdealSell}, "
#         f"BBandHigh={self.BBandHigh}, BBandLow={self.BBandLow}, RSI={self.RSI}, SMA={self.SMA}, "
#         f"BuyScore={self.BuyScore})")


def get_stock_list(amount_to_return=3):
  return ['XLK']
  filters = ['cap_small', 'geo_usa', 'sh_avgvol_o300', 'sh_opt_option']

  # https://finviz.com/screener.ashx?v=111&f=cap_small,geo_usa,sh_avgvol_o300,sh_opt_option&o=-change
  stock_list = Screener(
      filters=filters, table='Overview', order='-change'
  )  # Get the performance table and sort it by price ascending

  # stockList = []

  for x in stock_list:
    stockList.append(x['Ticker'])
  logger.info(f'stockList = \n{stock_list}')

  return stockList[:amount_to_return]


def get_eligible_option_data_for_stock(stock=""):
  logger.info(f'stock ={stock}')
  stockTicker = yf.Ticker(stock)

  optionHolderReturn = []

  for x in stockTicker.options:

    expirationTime = parse(x)
    if expirationTime.year > date.today().year:

      for index, row in stockTicker.option_chain(x).calls.iterrows():

        historicData = get_daily_data_for_option(row["contractSymbol"])
        contr_sym = row['contractSymbol']
        logger.info(
            f'stock = {stock}; Calls contractSymbol = {contr_sym}; Expiry: {expirationTime}'
        )
        if len(historicData) < 8:
          continue

        newOptionHolder = OptionHolder()
        newOptionHolder.Ticker = stock
        newOptionHolder.IsCall = True
        newOptionHolder.Expiry = expirationTime
        newOptionHolder.FilledPrice = row["lastPrice"]
        newOptionHolder.BidPrice = row["bid"]
        newOptionHolder.AskPrice = row["ask"]
        newOptionHolder.Strike = row["strike"]
        newOptionHolder.CurrentSpread = newOptionHolder.AskPrice - newOptionHolder.BidPrice
        newOptionHolder.PercentChange = round(row["change"], 3)
        newOptionHolder.OpenInterest = row["openInterest"]
        newOptionHolder.IV = round(row["impliedVolatility"], 3)
        newOptionHolder.YahooString = row["contractSymbol"]
        newOptionHolder.HistoricData = historicData
        optionHolderReturn.append(newOptionHolder)

      for index, row in stockTicker.option_chain(x).puts.iterrows():

        historicData = get_daily_data_for_option(row["contractSymbol"])
        contr_sym = row['contractSymbol']
        logger.info(
            f'stock = {stock}; Puts contractSymbol = {contr_sym}; Expiry: {expirationTime}'
        )
        if len(historicData) < 8:
          continue

        newOptionHolder = OptionHolder()
        newOptionHolder.Ticker = stock
        newOptionHolder.IsCall = False
        newOptionHolder.Expiry = expirationTime
        newOptionHolder.FilledPrice = row["lastPrice"]
        newOptionHolder.BidPrice = row["bid"]
        newOptionHolder.AskPrice = row["ask"]
        newOptionHolder.Strike = row["strike"]
        newOptionHolder.CurrentSpread = newOptionHolder.AskPrice - newOptionHolder.BidPrice
        newOptionHolder.PercentChange = round(row["change"], 3)
        newOptionHolder.OpenInterest = row["openInterest"]
        newOptionHolder.IV = round(row["impliedVolatility"], 3)
        newOptionHolder.YahooString = row["contractSymbol"]
        newOptionHolder.HistoricData = historicData
        optionHolderReturn.append(newOptionHolder)

  return optionHolderReturn


def get_daily_data_for_option(option=""):
  stockTicker = yf.Ticker(option)

  hist_data = stockTicker.history(period="1mo", interval="1d")
  return hist_data


def get_ideal_buy_sell_magic(option):
  option.IdealBuy = round(option.BidPrice + (option.CurrentSpread * .2), 3)
  option.IdealSell = round(option.AskPrice - (option.CurrentSpread * .2), 3)


if __name__ == '__main__':

  pd.set_option('display.max_rows', 500)
  pd.set_option('display.max_columns', 500)
  pd.set_option('display.width', 1000)

  stockList = get_stock_list(5)

  eligibleOptions = []
  for x in stockList:
    logger.info("Checking if stock is eligible: " + x)
    eligibleOptions.extend(get_eligible_option_data_for_stock(x))

  if len(eligibleOptions) == 0:
    logger.info("No data to use")
  else:
    logger.info(f'eligibleOptions = {eligibleOptions}')

    for option in eligibleOptions:
      option.HistoricData["SMA"] = ta.sma(option.HistoricData["Close"], 5)
      option.HistoricData["RSI"] = ta.rsi(option.HistoricData["Close"], 5)
      option.HistoricData["VWAP"] = ta.vwap(
          low=option.HistoricData["Low"],
          high=option.HistoricData["High"],
          close=option.HistoricData["Close"],
          volume=option.HistoricData["Volume"])
      bbandData = ta.bbands(option.HistoricData["Close"], 5, 2)
      option.HistoricData["BBU"] = bbandData["BBU_5_2.0"]
      option.HistoricData["BBL"] = bbandData["BBL_5_2.0"]

      option.BBandHigh = round(option.HistoricData["BBU"][-1], 3)
      option.BBandLow = round(option.HistoricData["BBL"][-1], 3)
      option.RSI = round(option.HistoricData["RSI"][-1], 3)
      option.SMA = round(option.HistoricData["SMA"][-1], 3)
      option.VWAP = round(option.HistoricData["VWAP"][-1], 3)
      option.Volume = round(option.HistoricData["Volume"].sum(), 3)
      option.CurrentSpread = round(
          max(option.HistoricData["Close"][-5:]) -
          min(option.HistoricData["Close"][-5:]), 3)
      get_ideal_buy_sell_magic(option)

      logger.info(f'option.HistoricData ==== \n{option.HistoricData}')
      logger.info(f'option in elig: {option}')

    for option in eligibleOptions:
      if option.RSI <= 40:
        option.BuyScore = option.BuyScore + 1

      if option.Volume >= 100:
        option.BuyScore = option.BuyScore + 1

      if option.HistoricData["Close"][-1] <= option.BBandLow:
        option.BuyScore = option.BuyScore + 1

      if option.SMA <= option.VWAP:
        option.BuyScore = option.BuyScore + 1

      if option.CurrentSpread >= 0.05:
        option.BuyScore = option.BuyScore + 1

      if option.FilledPrice == option.BidPrice:
        option.BuyScore = option.BuyScore + 1

      if option.IV <= 40:
        option.BuyScore = option.BuyScore + 1

      if option.PercentChange <= 0:
        option.BuyScore = option.BuyScore + 1

    outputTable = PrettyTable()

    outputTable.field_names = [
        "Sr. no.", "Ticker", "Strike", "Expiry", "Bid", "Filled", "Ask",
        "Ideal (Buy/Sell)", "Spread", "Vol / OI", "BB (S/R)", "RSI", "VWAP",
        "SMA(5)", "Today Gain", "IV", "B-Score"
    ]

    outputTable.align["Sr. no."] = "c"
    outputTable.align["Ticker"] = "c"
    outputTable.align["Strike"] = "r"
    outputTable.align["Expiry"] = "c"
    outputTable.align["Bid"] = "r"
    outputTable.align["Filled"] = "r"
    outputTable.align["Ask"] = "r"
    outputTable.align["Ideal (Buy/Sell)"] = "c"
    outputTable.align["Spread"] = "r"
    outputTable.align["Vol / OI"] = "c"
    outputTable.align["BB (S/R)"] = "c"
    outputTable.align["RSI"] = "r"
    outputTable.align["VWAP"] = "r"
    outputTable.align["SMA(5)"] = "r"
    outputTable.align["Today Gain"] = "r"
    outputTable.align["IV"] = "r"
    outputTable.align["B-Score"] = "c"

    # Sort this shit somehow
    eligibleOptions.sort(key=lambda x: x.BuyScore, reverse=True)

    for index, option in enumerate(eligibleOptions):
      outputTable.add_row(
          [
              index, option.Ticker, option.Strike, option.Expiry,
              '{:.3f}'.format(option.BidPrice), '{:.3f}'.format(
                  option.FilledPrice), '{:.3f}'.format(option.AskPrice),
              '{:.3f}'.format(option.IdealBuy) + " / " +
              '{:.3f}'.format(option.IdealSell),
              '{:.3f}'.format(round(option.CurrentSpread, 3)),
              str(option.Volume) + " / " + str(option.OpenInterest),
              '{:.3f}'.format(option.BBandLow) + " / " +
              '{:.3f}'.format(option.BBandHigh), '{:.3f}'.format(option.RSI),
              '{:.3f}'.format(option.VWAP), '{:.3f}'.format(option.SMA),
              '{:.3f}'.format(option.PercentChange), option.IV,
              str(option.BuyScore) + " / 8"
          ])

    logger.info(outputTable)
