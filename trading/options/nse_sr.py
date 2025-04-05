# Libraries
import requests
import json
import math
import sys
import os

# Reference: https://dev.to/shahstavan/nse-option-chain-data-using-python-4fe5
'''

Gives me response like:

 ---------------------------------------------------------------------|
 Nifty        =>   Last Price:  19381.65  Nearest Strike:  19400
 ---------------------------------------------------------------------|
03-Aug-2023 18900 CE [         147 ] PE [       39587 ]
03-Aug-2023 18950 CE [          38 ] PE [       10789 ]
03-Aug-2023 19000 CE [        1011 ] PE [      128434 ]
03-Aug-2023 19050 CE [         482 ] PE [       15949 ]
03-Aug-2023 19100 CE [         754 ] PE [       74242 ]
03-Aug-2023 19150 CE [        2827 ] PE [       46929 ]
03-Aug-2023 19200 CE [        6709 ] PE [      111348 ]
03-Aug-2023 19250 CE [        8732 ] PE [       99231 ]
03-Aug-2023 19300 CE [       51460 ] PE [      230263 ]
03-Aug-2023 19350 CE [      167530 ] PE [      190807 ]
03-Aug-2023 19400 CE [      326394 ] PE [      115098 ]
03-Aug-2023 19450 CE [       86599 ] PE [       29350 ]
03-Aug-2023 19500 CE [      135595 ] PE [       36965 ]
03-Aug-2023 19550 CE [       95073 ] PE [       18702 ]
03-Aug-2023 19600 CE [      157914 ] PE [       62841 ]
03-Aug-2023 19650 CE [       86955 ] PE [       19152 ]
03-Aug-2023 19700 CE [      148398 ] PE [       24129 ]
03-Aug-2023 19750 CE [       84152 ] PE [        8159 ]
03-Aug-2023 19800 CE [      157682 ] PE [       11675 ]
03-Aug-2023 19850 CE [       64936 ] PE [        5575 ]
 ---------------------------------------------------------------------|
 Bank Nifty   =>   Last Price:  44513.45  Nearest Strike:  44600
 ---------------------------------------------------------------------|
03-Aug-2023 43600 CE [         260 ] PE [       56106 ]
03-Aug-2023 43700 CE [         405 ] PE [       77726 ]
03-Aug-2023 43800 CE [         738 ] PE [       87052 ]
03-Aug-2023 43900 CE [        1134 ] PE [       57819 ]
03-Aug-2023 44000 CE [        6374 ] PE [      137215 ]
03-Aug-2023 44100 CE [        5665 ] PE [      107057 ]
03-Aug-2023 44200 CE [       31578 ] PE [      129439 ]
03-Aug-2023 44300 CE [       28988 ] PE [      193542 ]
03-Aug-2023 44400 CE [      132710 ] PE [      322334 ]
03-Aug-2023 44500 CE [      296699 ] PE [      358132 ]
03-Aug-2023 44600 CE [      403066 ] PE [      183505 ]
03-Aug-2023 44700 CE [      164637 ] PE [       44311 ]
03-Aug-2023 44800 CE [      134665 ] PE [       47609 ]
03-Aug-2023 44900 CE [      122790 ] PE [       12556 ]
03-Aug-2023 45000 CE [      194951 ] PE [       46639 ]
03-Aug-2023 45100 CE [       95421 ] PE [       11615 ]
03-Aug-2023 45200 CE [      116418 ] PE [       13275 ]
03-Aug-2023 45300 CE [       97611 ] PE [       21821 ]
03-Aug-2023 45400 CE [       85639 ] PE [       16289 ]
03-Aug-2023 45500 CE [      200893 ] PE [       41706 ]
 ---------------------------------------------------------------------|
 Major Support in Nifty:19400
 Major Resistance in Nifty:19300
 Major Support in Bank Nifty:44600
 Major Resistance in Bank Nifty:44500
'''


# Python program to print
# colored text and background
def strRed(skk):
  return "\033[91m {}\033[00m".format(skk)


def strGreen(skk):
  return "\033[92m {}\033[00m".format(skk)


def strYellow(skk):
  return "\033[93m {}\033[00m".format(skk)


def strLightPurple(skk):
  return "\033[94m {}\033[00m".format(skk)


def strPurple(skk):
  return "\033[95m {}\033[00m".format(skk)


def strCyan(skk):
  return "\033[96m {}\033[00m".format(skk)


def strLightGray(skk):
  return "\033[97m {}\033[00m".format(skk)


def strBlack(skk):
  return "\033[98m {}\033[00m".format(skk)


def strBold(skk):
  return "\033[1m {}\033[0m".format(skk)


# Method to get nearest strikes
def round_nearest(x, num=50):
  return int(math.ceil(float(x) / num) * num)


def nearest_strike_bnf(x):
  return round_nearest(x, 100)


def nearest_strike_nf(x):
  return round_nearest(x, 50)


# Urls for fetching Data
url_oc = "https://www.nseindia.com/option-chain"
url_bnf = 'https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY'
url_nf = 'https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY'
url_indices = "https://www.nseindia.com/api/allIndices"
URL_TICKER = 'https://www.nseindia.com/api/option-chain-equities?symbol=RELIANCE'

# Headers
headers = {
    'user-agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 Safari/537.36',
    'accept-language':
        'en,gu;q=0.9,hi;q=0.8',
    'accept-encoding':
        'gzip, deflate, br'
}

sess = requests.Session()
cookies = dict()


# Local methods
def set_cookie():
  request = sess.get(url_oc, headers=headers, timeout=5)
  cookies = dict(request.cookies)


def save_dictionary_to_local_file(url, data):
  # wrap whole function in try catch block
  try:
    last_part_of_url = url.split('/')[-1].split('symbol=')[-1]
    file_path = os.path.join('data', 'samples', f'{last_part_of_url}.json')
    print(
        f'save_dictionary_to_local_file: url = {url}; file path = {file_path}; type ={type(data)}; '
    )
    with open(file_path, 'wt') as outfile:
      # j_data = json.loads(data)
      json.dump(data, outfile, indent=4)
  except Exception as e:
    print(f'Exception: {e}')
    return


def get_data(url):
  set_cookie()
  response = sess.get(url, headers=headers, timeout=5, cookies=cookies)
  if (response.status_code == 401):
    set_cookie()
    response = sess.get(url_nf, headers=headers, timeout=5, cookies=cookies)
  if (response.status_code == 200):
    ans = response.text
    print(f'URL = {url}; type: {type(ans)}; ans = {ans}')
    save_dictionary_to_local_file(url, data=response.json())
    return response.text

  return ""


def set_header():
  response_text = get_data(url_indices)
  data = json.loads(response_text)
  for index in data["data"]:
    if index["index"] == "NIFTY 50":
      nf_ul = index["last"]
      print("nifty")
    if index["index"] == "NIFTY BANK":
      bnf_ul = index["last"]
      print("banknifty")
  bnf_nearest = nearest_strike_bnf(bnf_ul)
  nf_nearest = nearest_strike_nf(nf_ul)
  return bnf_ul, nf_ul, bnf_nearest, nf_nearest


# Showing Header in structured format with Last Price and Nearest Strike


def print_header(index="", ul=0, nearest=0):
  print(
      strPurple(index.ljust(12, " ") + " => ") +
      strLightPurple(" Last Price: ") + strBold(str(ul)) +
      strLightPurple(" Nearest Strike: ") + strBold(str(nearest)))


def print_hr():
  print(strYellow("|".rjust(70, "-")))


# Fetching CE and PE data based on Nearest Expiry Date
def print_oi(num, step, nearest, url):
  strike = nearest - (step * num)
  start_strike = nearest - (step * num)
  response_text = get_data(url)
  data = json.loads(response_text)

  currExpiryDate = data["records"]["expiryDates"][0]
  for item in data['records']['data']:
    if item["expiryDate"] == currExpiryDate:
      if item["strikePrice"] == strike and item[
          "strikePrice"] < start_strike + (step * num * 2):
        #print(strCyan(str(item["strikePrice"])) + strGreen(" CE ") + "[ " + strBold(str(item["CE"]["openInterest"]).rjust(10," ")) + " ]" + strRed(" PE ")+"[ " + strBold(str(item["PE"]["openInterest"]).rjust(10," ")) + " ]")
        print(
            data["records"]["expiryDates"][0] + " " + str(item["strikePrice"]) +
            " CE " + "[ " +
            strBold(str(item["CE"]["openInterest"]).rjust(10, " ")) + " ]" +
            " PE " + "[ " +
            strBold(str(item["PE"]["openInterest"]).rjust(10, " ")) + " ]")
        strike = strike + step


# Finding highest Open Interest of People's in CE based on CE data
def highest_oi_CE(num, step, nearest, url):
  strike = nearest - (step * num)
  start_strike = nearest - (step * num)
  response_text = get_data(url)
  data = json.loads(response_text)
  currExpiryDate = data["records"]["expiryDates"][0]
  max_oi = 0
  max_oi_strike = 0
  for item in data['records']['data']:
    if item["expiryDate"] == currExpiryDate:
      if item["strikePrice"] == strike and item[
          "strikePrice"] < start_strike + (step * num * 2):
        if item["CE"]["openInterest"] > max_oi:
          max_oi = item["CE"]["openInterest"]
          max_oi_strike = item["strikePrice"]
        strike = strike + step
  return max_oi_strike


# Finding highest Open Interest of People's in PE based on PE data
def highest_oi_PE(num, step, nearest, url):
  strike = nearest - (step * num)
  start_strike = nearest - (step * num)
  response_text = get_data(url)
  data = json.loads(response_text)
  currExpiryDate = data["records"]["expiryDates"][0]
  max_oi = 0
  max_oi_strike = 0
  for item in data['records']['data']:
    if item["expiryDate"] == currExpiryDate:
      if item["strikePrice"] == strike and item[
          "strikePrice"] < start_strike + (step * num * 2):
        if item["PE"]["openInterest"] > max_oi:
          max_oi = item["PE"]["openInterest"]
          max_oi_strike = item["strikePrice"]
        strike = strike + step
  return max_oi_strike


def main():
  bnf_ul, nf_ul, bnf_nearest, nf_nearest = set_header()

  print('\033c')
  print_hr()
  print_header("Nifty", nf_ul, nf_nearest)
  print_hr()
  print_oi(10, 50, nf_nearest, url_nf)
  print_hr()
  print_header("Bank Nifty", bnf_ul, bnf_nearest)
  print_hr()
  print_oi(10, 100, bnf_nearest, url_bnf)
  print_hr()

  # Finding Highest OI in Call Option In Nifty
  nf_highestoi_CE = highest_oi_CE(10, 50, nf_nearest, url_nf)

  # Finding Highet OI in Put Option In Nifty
  nf_highestoi_PE = highest_oi_PE(10, 50, nf_nearest, url_nf)

  # Finding Highest OI in Call Option In Bank Nifty
  bnf_highestoi_CE = highest_oi_CE(10, 100, bnf_nearest, url_bnf)

  # Finding Highest OI in Put Option In Bank Nifty
  bnf_highestoi_PE = highest_oi_PE(10, 100, bnf_nearest, url_bnf)

  print(strCyan(str("Major Support in Nifty:")) + str(nf_highestoi_CE))
  print(strCyan(str("Major Resistance in Nifty:")) + str(nf_highestoi_PE))
  print(strPurple(str("Major Support in Bank Nifty:")) + str(bnf_highestoi_CE))
  print(
      strPurple(str("Major Resistance in Bank Nifty:")) + str(bnf_highestoi_PE))


if __name__ == "__main__":
  main()
