import os
import sys
import datetime
import argparse
import logging

import volvisualizer.volatility as vol

from trading.common import utils

logger = utils.get_logger('option_vizualizer')

# change logging of volatility to none
logging.getLogger('volatility').setLevel(logging.ERROR)

# Ref: https://github.com/GBERESEARCH/volvisualizer/blob/master/notebooks/Implied%20Volatility%20-%20SPX%20Walkthrough.ipynb

TICKERS_TO_VIZUALIZER = ['SPX', 'IWM', 'RSP', 'QQQ', 'GLD', 'TLT']
TICKERS_TO_VIZUALIZER = ['SPX', 'TLT']
TICKERS_TO_VIZUALIZER = ['TLT']
TICKERS_TO_VIZUALIZER = ['SPX', 'QQQ', 'GLD', 'TLT', 'USO']
TICKERS_TO_VIZUALIZER = ['SPY', 'USO']
'''
Usage:

python trading/reports/options_vizualizer.py --tickers SPX
python trading/reports/options_vizualizer.py --tickers IWM
python trading/reports/options_vizualizer.py --tickers RSP
python trading/reports/options_vizualizer.py --tickers QQQ
python trading/reports/options_vizualizer.py --tickers GLD
python trading/reports/options_vizualizer.py --tickers TLT
python trading/reports/options_vizualizer.py --tickers SPY QQQ GDX IJR TLT USO

'''


def get_args():

  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # add args for tickers, if empty, change tickers to TICKERS_TO_VIZUALIZER
  parser.add_argument(
      '--tickers', '-t', nargs='+', default=TICKERS_TO_VIZUALIZER)

  args = parser.parse_args()
  return args


def main():
  args = get_args()
  logger.info(f'args = {args}')
  for ticker in args.tickers:
    logger.info(f'Running for {ticker}')
    imp = vol.Volatility(ticker=ticker)
    imp.linegraph(save_image=True, image_folder='data/images', show_graph=True)
    imp.skewreport(direction='full')
    logger.info(f'saved for {ticker}')


if __name__ == '__main__':
  main()
