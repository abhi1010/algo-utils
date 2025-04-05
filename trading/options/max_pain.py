import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from urllib.parse import urlencode

# https://marketsetup.in/posts/option-chain/max-pain/

from trading.options.option_chain import option_chain

nifty_chain = option_chain("NIFTY", "OPTIDX", "30JUL2020")


def total_loss_at_strike(chain, expiry_price):
  """Calculate loss at strike price"""
  # All call options with strike price below the expiry price will result in loss for option writers
  in_money_calls = chain[chain['Strike Price'] < expiry_price][[
      "CE OI", "Strike Price"
  ]]
  in_money_calls["CE loss"] = (
      expiry_price - in_money_calls['Strike Price']) * in_money_calls["CE OI"]

  # All put options with strike price above the expiry price will result in loss for option writers
  in_money_puts = chain[chain['Strike Price'] > expiry_price][[
      "PE OI", "Strike Price"
  ]]
  in_money_puts["PE loss"] = (in_money_puts['Strike Price'] -
                              expiry_price) * in_money_puts["PE OI"]
  total_loss = in_money_calls["CE loss"].sum() + in_money_puts["PE loss"].sum()

  return total_loss


strikes = list(nifty_chain['Strike Price'])
losses = [
    total_loss_at_strike(nifty_chain, strike) / 1000000 for strike in strikes
]
import matplotlib.pyplot as plt

plt.plot(strikes, losses)
plt.ylabel('Total loss in rs (Millon)')
plt.show()
m = losses.index(min(losses))
print("Max pain > {}".format(strikes[m]))