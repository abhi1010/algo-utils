import pytz
from datetime import timedelta
import datetime
import json
import logging
import math
import pprint
import time
import csv
import os
import calendar

from trading.common import utils

now_s = datetime.datetime.now().strftime("util_files_%Y-%m-%d__%H-%M-%S")

logger = utils.get_logger(now_s)

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
import io
from collections import defaultdict


def get_latest_file_in_folder_by_pattern(folder_path, pattern):
  files = [
      os.path.join(folder_path, f)
      for f in os.listdir(folder_path)
      if pattern in f
  ]
  logger.info(f'files = {files}')
  if len(files):
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def get_latest_file_in_folder(base_dir, folder_path, extension_to_filter):
  extension_to_filter = extension_to_filter if extension_to_filter.startswith(
      '.') else '.' + extension_to_filter
  files = [
      os.path.join(base_dir, folder_path, f)
      for f in os.listdir(os.path.join(base_dir, folder_path))
      if f and f.endswith(extension_to_filter)
  ]
  if len(files):
    latest_file = max(files, key=os.path.getctime)
    return latest_file


def get_file_with_pattern(dir, pattern):
  files = listdir(path=dir)
  file_found = [
      file for file in files if match(pattern, file, flags=IGNORECASE)
  ]
  return file_found


def get_file_text(file_path):
  f_first, ext = path.splitext(file_path)
  if ext.upper() == '.ZIP':
    with ZipFile(file_path) as zfile:
      for finfo in zfile.infolist():
        ifile = zfile.open(finfo)
        return ifile.read()
  lines = get_file_lines(file_path)
  text = ''.join(lines)
  return text


def get_file_lines(file_path):
  # Try multiple encoding
  enc_list = ['utf-8', 'iso-8859-1']
  for encode in enc_list:
    try:
      with open(file_path, 'rt', encoding=encode) as file_placeholder:
        return file_placeholder.readlines()
    except:
      pass
  return []


def get_csv_lines(file_path):
  # Try multiple encoding
  enc_list = ['utf-8', 'iso-8859-1']
  for encode in enc_list:
    try:
      with open(file_path, 'rt', encoding=encode) as file_placeholder:
        reader = csv.reader(file_placeholder, delimiter=',')
        return [line for line in reader]
    except:
      pass
  return []


def read_json_text(file_path):
  if path.exists(file_path):
    logging.debug('Found JSON locally: {}'.format(file_path))
    file_text = get_file_text(file_path)
    #json_acceptable_string = file_text.replace("'", "\"")
    json_acceptable_string = file_text
    json_text = loads(json_acceptable_string)

  return json_text


def read_gzip_file(file_path):
  import gzip
  f = gzip.open(file_path, 'rb')
  file_content = f.read()
  f.close()
  return file_content


def get_csv_lines_without_ominous_mark(file_path):
  # Removes BOM mark, if it exists, from a file and rewrites it in-place
  s = open(file_path, mode='r', encoding='utf-8-sig').read()
  open(file_path, mode='w', encoding='utf-8').write(s)
  # Replace ^M line break
  lines = get_file_lines(file_path)
  new_lines = []
  for l in lines:
    new_lines.append(
        l.replace(r'\r',
                  '').replace('\n',
                              '').replace('/', '_').replace('"', '').split(','))
  return new_lines


def find_dirs_and_files(
    pattern, where='.', include_dirs=True, include_files=True, debug=False):
  '''Returns list of filenames from `where` path matched by 'which'
       shell pattern. Matching is case-insensitive.'''
  if not where or not path.exists(where):
    return []
  if debug:
    logging.debug("where={}; pattern={};".format(where, pattern))
  rule = compile(translate(pattern), IGNORECASE)
  list_files = [name for name in listdir(where) if rule.match(name)]
  if debug:
    logging.debug('list_files={}'.format(get_pretty_print(list_files)))

  if include_dirs and not include_files:
    list_files = [
        name for name in list_files if path.isdir(path.join(where, name))
    ]
  if include_files and not include_dirs:
    list_files = [
        name for name in list_files if path.isfile(path.join(where, name))
    ]
  if include_files:
    list_files = add_folder_to_list_of_files(where, list_files)
  return list_files


def add_folder_to_list_of_files(folder, files):
  return [
      path.join(folder, file)
      if not path.isabs(file) and path.exists(path.join(folder, file)) else file
      for file in files
  ]


def is_valid_file(file_name):
  """
    matching filename either based on extension or the filename in digits and/or dash
    """
  pattern = compile('(\.(csv|xls|txt|pdf)$)|(20\d{2})-?(\d{1,2})-?(\d{1,2})')
  res = pattern.search(file_name)
  return res is not None


def write_to_file(file_path, text_to_write, write_type='wt'):
  with open(file_path, write_type) as f:
    f.write(text_to_write)
    # f.write(text_to_write if isinstance(bytes, type(text_to_write)) \
    #             else bytes(text_to_write, 'UTF-8'))
    f.close()


def path_joins(items):
  p = ''
  if len(items) > 0:
    p = items[0]
    if len(items) > 1:
      for item in items[1:]:
        p = path.join(p, item)
  return p


def remove_files(items):
  for item in items:
    logging.debug("Removing Item={}".format(item))
    remove(item)


def get_report_json_folder(main_folder):
  return path_joins([main_folder, 'json'])


def get_report_html_folder(main_folder):
  return path_joins([main_folder, 'html'])


def find_full_path(relative_path):
  if path.isabs(relative_path):
    return relative_path
  return path.abspath(relative_path)


def create_zip(zip_file_path, files_with_full_path):
  with ZipFile(zip_file_path, 'w') as myzip:
    for file in files_with_full_path:
      myzip.write(file, path.basename(file))


def upload_to_ftp_service(sftp_svc, filepath):
  if sftp_svc:
    sftp_svc.upload_file(filepath)
