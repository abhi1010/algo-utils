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
DATE FUNCTIONS
===========================================================================================
"""


def find_days_since_near_far(far_date, near_date):
  now = datetime.datetime.now()
  days_since_far_date = (now - far_date).days
  days_since_near_date = (now - near_date).days
  return days_since_far_date, days_since_near_date


def convert_utc_to_sgt_datetime(utc_datetime):
  sg_timezone = pytz.timezone('Asia/Singapore')
  return utc_datetime.replace(tzinfo=pytz.utc).astimezone(sg_timezone)


def date_to_string_YYMM(date):
  return date.strftime('%y%m')


def date_to_string_dd_mon_year(date):
  return date.strftime('%d-%b-%Y')


def date_to_string_YYMM_MM_DD(
    date_obj=datetime.datetime.now().date(), offset=0):
  date_obj_with_offset = date_obj + timedelta(days=offset)
  date_obj_str = date_obj_with_offset.strftime('%Y-%m-%d')
  return date_obj_str


def date_to_string_fmt(date, fmt):
  return date.strftime(fmt)


def string_to_datetime(date_str, format):
  return datetime.datetime.strptime(date_str, format)


_DATE_FMT_SUGGESTION_0 = (
    r'(20\d{2})-?((?:1[0-2])|(?:0\d))-?(\d{1,2})', '%Y-%m-%d')
_DATE_FMT_SUGGESTION_1 = (r'(\d{1,2})-?(\d{1,2})-?(20\d{2})', '%d-%m-%Y')
_DATE_FMT_SUGGESTION_2 = (
    r'((?:[0-2]\d)|(?:3[01]))-?((?:1[0-2])|(?:0\d))-?(201[4-9])', '%d-%m-%Y')
DATE_FMT_SUGGESTIONS = [
    _DATE_FMT_SUGGESTION_0, _DATE_FMT_SUGGESTION_1, _DATE_FMT_SUGGESTION_2
]


def filename_to_date(file_name, date_suggestions_list=DATE_FMT_SUGGESTIONS):
  for pattern, suggestion in date_suggestions_list:
    match = findall(pattern, file_name)
    if match is not None and len(match) > 0:
      groups = match[0]
      year_date = '-'.join(groups)
      date_value = string_to_datetime(year_date, suggestion)
      return date_value
  return None
  # return datetime.datetime.today()


def date_with_offset(dt0, offset_days):
  dt2 = dt0 + timedelta(days=offset_days)
  return dt2


def guess_date_format(datetime_str):
  join = lambda *a: a[0].join(a[1:]).strip()
  timefmt = (':', '(([01]?\d|2[0-3]))', '([0-5]\d)', '([0-5]\d)')
  timefmt2 = (('',) + timefmt[1:])
  if match('^%s$' % join(*timefmt), datetime_str):
    return '%H:%M:%S'
  for sep in (
      '-',
      '/',
      '',
  ):
    datefmt = (sep, '(20\d{2}', '((0\d)|(1[0-2]))', '(([012]\d)|(3[01])))')
    for i in (4, 3, 1):
      if match('^%s$' % join(' ', join(*datefmt), join(*timefmt[:i])),
               datetime_str):
        return join(
            ' ', join(*(sep, '%Y', '%m', '%d')),
            join(*(':', '%H', '%M', '%S')[:i]))
    for i in (4, 3, 1):
      if match('^%s$' % join(' ', join(*datefmt), join(*timefmt2[:i])),
               datetime_str):
        return join(
            ' ', join(*(sep, '%Y', '%m', '%d')),
            join(*('', '%H', '%M', '%S')[:i]))


EXPIRY_CODES = 'FGHJKMNQUVXZ'
MONTHS = calendar.month_name[1:13]

fzPATTERN = compile(r'(\w+)/?([%s])(\d{1,2})(.?\w*)$' % EXPIRY_CODES)
ByPATTERN = compile(r'\b(%s) ?(\d{2})' % '|'.join(MONTHS), flags=IGNORECASE)
byPATTERN = compile(
    r'\b(%s) ?(\d{2})' % '|'.join([m[:3] for m in MONTHS]), flags=IGNORECASE)

single_sep = lambda s, sep=' ': sep.join([i for i in s.split(sep) if i])


def parse_expiry_and_reduce_symbol(symbol):
  """This function utilizes 'expiry_from_symbol' and 'expiry_from_normalized_name_YYMM'
    together to extract the expiry out of a given symbol. Reduces the headache to do the same work by
    other places twice"""

  parsers = [expiry_from_symbol, expiry_from_normalized_name_YYMM]

  def reduce_symbol(sym, fn_ptr):
    try_expiry, reduced_symbol = fn_ptr(sym)
    success = True if try_expiry else False
    return try_expiry, reduced_symbol, success

  for parser_fn in parsers:
    parsed_expiry, reduced_sym, did_work = reduce_symbol(symbol, parser_fn)
    if did_work:
      return parsed_expiry, reduced_sym, did_work
  return '', symbol, False


def expiry_from_ymm(y, mm):
  td = datetime.date.today() + timedelta(days=-365)  # 1 yr grace period
  yy = td.year - td.year % 10 + y
  xd = datetime.date(yy, mm, 20)
  return (yy % 100) * 100 + mm + (xd < td and 1000 or 0)


def expiry_from_symbol(symbol):
  s = ByPATTERN.search(symbol)
  if s:
    return expiry_from_tuple(symbol, s.group(0), '%B')
  s = byPATTERN.search(symbol)
  if s:
    return expiry_from_tuple(symbol, s.group(0), '%b')
  s = fzPATTERN.search(symbol.upper())
  if s:
    ticker, m, y, sfx = s.groups()
    mm = 1 + EXPIRY_CODES.index(m)
    yymm = len(y) == 1 and expiry_from_ymm(int(y), mm) or 100 * int(y) + mm
    return ('%04d' % yymm), ticker + sfx
  return None, None


def expiry_from_tuple(name, expiry_str, fmt):
  strip_space = lambda a: ' '.join([i for i in a.split(' ') if i])
  strip_tail = lambda a, tail: strip_space(
      a.endswith(tail) and a.replace(tail, '') or a)
  d = datetime.datetime.strptime(expiry_str.replace(' ', ''), fmt + '%y')
  return d.strftime('%y%m'), single_sep(
      strip_tail(name.replace(expiry_str, ''), 'FUTURE'))


def expiry_from_normalized_name_YYMM(symbol):
  p = r'(\w+)-?((?:\d\d)(?:(?:0\d)|(?:1[0-2])))$'
  match = search(p, symbol, flags=IGNORECASE)
  if match is not None:
    return match.group(2), match.group(1)
  return '', ''
