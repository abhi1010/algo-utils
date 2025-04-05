import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.parse import urlencode

# https://marketsetup.in/posts/option-chain/fetch/


def np_float(x):
  try:
    y = x.lstrip().rstrip().replace(',', '')
    return np.float64(y)
  except:
    return np.nan


def option_chain(symbol, instrument, date_="-"):
  base_url = "https://www.nseindia.com/option-chain"
  parameters = {
      "segmentLink": 17,
      "instrument": instrument,
      "symbol": symbol,
      "date": date_
  }
  url = base_url + urlencode(parameters)
  url = base_url

  headers = {
      'user-agent':
          'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
      'accept-language':
          'en,gu;q=0.9,hi;q=0.8',
      'accept-encoding':
          'gzip, deflate, br'
  }
  r = requests.get(url, headers=headers)

  bs = BeautifulSoup(r.text, features="lxml")
  table = bs.find("table", {"id": "octable"})
  # Get all rows
  table_rows = table.find_all('tr')

  l = []
  for tr in table_rows:
    td = tr.find_all('td')
    if td:
      row = [tr.text for tr in td]
      l.append(row)

  arr = []
  for r in l:
    row = [np_float(x) for x in r]
    arr.append(row)

  df = pd.DataFrame(arr[:-1])
  df.columns = [
      "CE Chart", "CE OI", "CE Change in OI", "CE Volume", "CE IV", "CE LTP",
      "CE Net Change", "CE Bid Qty", "CE Bid Price", "CE Ask Price",
      "CE Ask Quantity", "Strike Price", "PE Bid Qty", "PE Bid Price",
      "PE Ask Price", "PE Ask Qty", "PE Net Change", "PE LTP", "PE IV",
      "PE Volume", "PE Change in OI", "PE OI", "PE Chart"
  ]
  return df


option_chain("BANKNIFTY", "OPTIDX")
