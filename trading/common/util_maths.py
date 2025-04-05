import pytz
from datetime import timedelta
import datetime
import json
import logging
import math
import pprint
import time
import csv
import calendar
from os import path, remove, listdir, environ
from json import loads
from re import match, IGNORECASE, search, compile, sub, findall, escape
from fnmatch import translate
from subprocess import check_output
from enum import IntEnum
"""
============================================================================================================
"""
"""
===========================================================================================
PRICE FUNCTIONS
===========================================================================================
"""

r = lambda x, y: x.replace(y, '')
PARSE_FLOAT = lambda x: float(
    x if (isinstance(x, int) or isinstance(x, float))\
    else r('-' + r(x, '-'), ',') if x and '-' in [x[0],x[-1]] else r(x, ',') if x else 0
)
ADD_COMMA = lambda t: "{:,.2f}".format(round(t, 2)) if t else ''


def convert_to_price(str_price):
  str_price = str_price.replace(',', '').replace('*', '')
  price_f = 0.0
  if str_price[-1] == '-':
    price_f = -float(str_price[0:-1])
  else:
    price_f = float(str_price)
  return price_f


def are_two_floats_same(price1, price2, threshold=0.9):
  if price1 == price2:
    return True
  diff = math.fabs(price1 - price2)
  return diff < threshold
