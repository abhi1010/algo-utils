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


def env_variables_set(key, value):
  environ[key] = value


def env_variables_get(key, default_value=''):
  return environ.get(key, default_value)


def run_cmd(cmd):
  logging.debug('Running command: {}'.format(cmd))
  return check_output(cmd)


def script_upload(script_upload_file_path, filename, full_path, url,
                  client_name, additional_args):
  logging.debug('''script_upload_file_path={},
                   filename={}
                   full_path={}
                   url={}
                   client_name={}
                   additional_args={}'''.format(script_upload_file_path,
                                                filename, full_path, url,
                                                client_name, additional_args))
  output = check_output([
      script_upload_file_path, filename, full_path, url, client_name,
      additional_args
  ])
  return output.decode(encoding='UTF-8', errors='ignore')
