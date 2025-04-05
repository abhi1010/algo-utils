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
import http.client
from urllib.parse import urlparse
import requests
from zipfile import ZipFile

import io
from collections import defaultdict
"""
============================================================================================================
"""
"""
===========================================================================================
PRINT FUNCTIONS
===========================================================================================
"""


def get_pretty_print(item, depth=None):
  custom_printer = lambda a: (isinstance(a,list) or isinstance(a, tuple)) and \
                             len(a) > 0 and \
                             (isinstance(a[0], list) or isinstance(a[0], tuple))
  # if is_nested_dict(item):
  #     return print_nested_dict(item)
  if custom_printer(item):
    s = ''
    for i in item:
      s += '\n' + str(i)
    return s
  pp = pprint.PrettyPrinter(indent=4, depth=depth)
  return pp.pformat(item)


is_non_empty_dict = lambda x: isinstance(x, dict) and len(x) > 0
is_nested_dict = lambda x: is_non_empty_dict(x) and is_non_empty_dict(
    x[list(x.keys())[0]])


def get_groupdict(dt, prefix=''):
  _filler = lambda x: '{0:02d}'.format(x)
  prefix += '_' if prefix else ''

  prev_date = util_dates.date_with_offset(dt.date(), -1)
  prev_date_str = str(prev_date.year) + str(_filler(prev_date.month)) + str(
      _filler(prev_date.day))

  groupdict = {
      '{}year'.format(prefix): str(dt.date().year),
      '{}month'.format(prefix): _filler(dt.date().month),
      '{}day'.format(prefix): _filler(dt.date().day),
      '{}prev_day'.format(prefix): prev_date_str,
  }
  return groupdict


def get_nested_dict_as_str(d1):
  so = io.StringIO()

  k_names = set()
  k_lens = defaultdict(list)
  if is_nested_dict(d1):
    for k1 in d1:
      for k2, v2 in d1[k1].items():
        k_names.add(k2)
        k_lens[k2].append(len(str(v2)))
  print('k_names = {}'.format(k_names))
  k_list = sorted(list(k_names))
  print('k_lens={}'.format(k_lens))
  # k_fmt = '{:20} ' + ' '.join(['{:' + str(len(x)+3 if len(x) < 15 else 20)+ '}' for x in k_list])
  k_fmt = '{:20} ' + ' '.join(
      ['{:' + str(max(len(x), max(k_lens[x]))) + '}' for x in k_list])
  # print(k_fmt)
  k_fmt += '\n'

  header = k_fmt.format('Name', *k_list)
  so.write(header)
  # print(header)
  for k1 in d1:
    d2 = d1[k1]

    vals = [k1] + [d2[x] if x in d2 else '' for x in k_list]
    # print('kfmt = {}; vals={}'.format(k_fmt, vals))
    s = k_fmt.format(*vals)
    # print(s)
    # print('I want {} {}'.format(k1, s))
    so.write(s)

  # print('{} '.format(so.getvalue()))
  return so.getvalue()
