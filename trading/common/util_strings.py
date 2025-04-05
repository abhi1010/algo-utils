import sys
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
import io
from collections import defaultdict
"""
============================================================================================================
"""
"""
===========================================================================================
STRING FUNCTIONS
===========================================================================================
"""


def regex_passes_tests(pattern, text, flags=IGNORECASE):
  return search(pattern, text, flags) is not None


def remove_asterisks(word):
  return sub('\*[ \*]*\*', '', word).strip()


def remove_extra_spaces(word):
  word = word.replace('   ', ' ')
  return word.replace('  ', ' ').strip()


def get_first_two_words(sourcename):
  """
    Returns the first two words only - trimmed.

    No matter how big the info is, it only returns two words that may serve as the SOURCE for DbLayer
    """
  pattern = '^\w* \w*'
  match = search(pattern, sourcename)
  s = match.start()
  e = match.end()
  return sourcename[s:e]


def replace_ci(text, search, replace):
  pattern = compile(escape(search), IGNORECASE)
  return pattern.sub(replace, text)


def replace_ci_without_compile(text, search, replace):
  return sub(search, replace, text)


def split_str(s, NUMBER_OF_ITEMS=29):
  # s = 'NSE:ASTRAL,NSE:TIMKEN,NSE:TATASTEEL,NSE:MANAPPURAM,NSE:ULTRACEMCO,NSE:JYOTHYLAB,NSE:SKFINDIA,NSE:USHAMART,NSE:MARUTI,NSE:CUMMINSIND,NSE:NCC,NSE:GMRINFRA,NSE:BEML,NSE:BLUEDART,NSE:CSBBANK,NSE:BLUESTARCO,NSE:OLECTRA,NSE:JBMA,NSE:SWSOLAR,NSE:COROMANDEL,NSE:IOB,NSE:BIRLACORPN,NSE:REDINGTON,NSE:RTNINDIA,NSE:ABB,NSE:ADANIENT,NSE:NUVOCO,NSE:ESCORTS,NSE:EICHERMOT,NSE:CELLO,NSE:DATAPATTNS,NSE:PNB,NSE:EMAMILTD,NSE:BDL,NSE:JWL,NSE:NMDC,NSE:MTARTECH,NSE:CAMPUS,NSE:TITAGARH,NSE:ENGINERSIN,NSE:SCHAEFFLER,NSE:RITES,NSE:OBEROIRLTY,NSE:RAJESHEXPO,NSE:GSFC,NSE:NAVINFLUOR,NSE:HAL,NSE:J_KBANK,NSE:NHPC,NSE:BORORENEW,NSE:AMBUJACEM,NSE:JUBLFOOD,NSE:KSB,NSE:NLCINDIA,NSE:GODREJPROP,NSE:RAILTEL,NSE:SAIL,NSE:CUB,NSE:RKFORGE,NSE:DEEPAKNTR,NSE:POWERINDIA,NSE:ROUTE,NSE:IRCTC,NSE:CREDITACC,NSE:INDIACEM,NSE:GMDCLTD,NSE:M_M,NSE:BALAMINES,NSE:ENDURANCE,NSE:VTL,NSE:THERMAX,NSE:CRISIL,NSE:KOTAKBANK,NSE:HAPPYFORGE,NSE:ACE,NSE:ABFRL,NSE:SCHNEIDER,NSE:HDFCBANK,NSE:POLYCAB,NSE:GPPL,NSE:ATUL,NSE:ELECON,NSE:INTELLECT,NSE:KARURVYSYA,NSE:BHARATFORG,NSE:LLOYDSME,NSE:ASTERDM,NSE:EIHOTEL,NSE:NSLNISP,NSE:CHEMPLASTS,NSE:NESTLEIND,NSE:GRINDWELL,NSE:BAYERCROP,NSE:ACC,NSE:SOBHA,NSE:TRIDENT,NSE:JKLAKSHMI,NSE:LICI,NSE:JBCHEPHARM,NSE:MAXHEALTH,NSE:PRESTIGE,NSE:LINDEINDIA,NSE:POWERGRID,NSE:HAVELLS,NSE:AXISBANK'
  # this is a comma separated string, split it into multiple lists so that every list has only 29 elements
  parts = s.strip()
  prefix = 'LIST: \n'

  while (parts):
    sub_list_parts = parts.split(',', NUMBER_OF_ITEMS)
    if len(sub_list_parts) == 1:
      break
    sub_list_parts_s = ','.join(sub_list_parts[:NUMBER_OF_ITEMS])
    print(f'{prefix}: \n  {sub_list_parts_s}')
    # print(f'\nparts = {parts}; len = {len(parts)}')
    parts = sub_list_parts[-1]
    # print(f'\nanew parts = {parts}; len = {len(parts)}')
    print(f'-' * 80)


if __name__ == "__main__":
  # take first cli as string to split
  split_str(sys.argv[1])
