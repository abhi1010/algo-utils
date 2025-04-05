from dhanhq import dhanhq
from enum import Enum
import pandas as pd
import datetime
from datetime import timedelta
import math
import re
import argparse
import pickle

from trading.common import utils

logger = utils.get_logger('dhan-common')

from trading.common.util_prints import *
from trading.common.util_dates import *
from trading.portfolio.tracker import *
from trading.configs import txn_weekend

TOKEN = os.getenv('DHAN_TOKEN')
DHAN_DIR = os.getenv('DHAN_DIR', 'data/dhan-prod/')
DHAN_SCRIP_FILE = os.getenv('DHAN_SCRIP_FILE',
                            'data/dhan-samples/api-scrip-master.csv')
DEFAULT_TAG = os.getenv('DHAN_DEFAULT_TAG', 'weekend')


# add a class DHAN_TXN_TYPE that is of string enum type
class DHAN_TXN_TYPE(str, Enum):
  ORDERS = 'orders'
  TRADEBOOK = 'tradebook'
  TRADE_HISTORY = 'trade_history'
  TRADES = 'trades'
  POSITIONS = 'positions'
  HOLDINGS = 'holdings'

  def __str__(self):
    return self.value


DHAN_INFO_COLS = {
    DHAN_TXN_TYPE.ORDERS: ['tradingSymbol', 'createTime'],
    DHAN_TXN_TYPE.TRADES: ['tradingSymbol', 'exchangeTime'],
    DHAN_TXN_TYPE.TRADEBOOK: ['tradingSymbol', 'exchangeTime'],
    DHAN_TXN_TYPE.TRADE_HISTORY: ['tradingSymbol', 'exchangeTime'],
    DHAN_TXN_TYPE.POSITIONS: ['tradingSymbol', ''],
    DHAN_TXN_TYPE.HOLDINGS: ['tradingSymbol', '']
}

OrderColumns = [
    'dhanClientId', 'orderId', 'exchangeOrderId', 'correlationId',
    'orderStatus', 'transactionType', 'exchangeSegment', 'productType',
    'orderType', 'validity', 'tradingSymbol', 'securityId', 'quantity',
    'disclosedQuantity', 'price', 'triggerPrice', 'afterMarketOrder',
    'boProfitValue', 'boStopLossValue', 'legName', 'createTime', 'updateTime',
    'exchangeTime', 'drvExpiryDate', 'drvOptionType', 'drvStrikePrice',
    'omsErrorCode', 'omsErrorDescription', 'filled_qty', 'algoId'
]

OrderColumnsToKeep = [
    'orderStatus', 'transactionType', 'tradingSymbol', 'quantity', 'price',
    'omsErrorDescription'
]

OrderColumnsToRemove = set(OrderColumns).difference(set(OrderColumnsToKeep))

PositionColumns = [
    'dhanClientId', 'tradingSymbol', 'securityId', 'positionType',
    'exchangeSegment', 'productType', 'buyAvg', 'costPrice', 'buyQty',
    'sellAvg', 'sellQty', 'netQty', 'realizedProfit', 'unrealizedProfit',
    'rbiReferenceRate', 'multiplier', 'carryForwardBuyQty',
    'carryForwardSellQty', 'carryForwardBuyValue', 'carryForwardSellValue',
    'dayBuyQty', 'daySellQty', 'dayBuyValue', 'daySellValue', 'drvExpiryDate',
    'drvOptionType', 'drvStrikePrice', 'crossCurrency'
]

PositionColumnsToKeep = [
    'tradingSymbol', 'securityId', 'positionType', 'costPrice',
    'unrealizedProfit', 'netQty'
]

HoldingsColumns = [
    'availableQty', 'avgCostPrice', 'collateralQty', 'dpQty', 'exchange',
    'isin', 'securityId', 't1Qty', 'totalQty', 'tradingSymbol'
]
HoldingsColumnsToKeep = ['totalQty', 'tradingSymbol', 'avgCostPrice']
HoldingsColumnsToRemove = set(HoldingsColumns).difference(
    set(HoldingsColumnsToKeep))

PositionColumnsToRemove = set(PositionColumns).difference(
    set(PositionColumnsToKeep))

TradesHistoryColumns = [
    'dhanClientId', 'orderId', 'exchangeOrderId', 'exchangeTradeId',
    'transactionType', 'exchangeSegment', 'productType', 'orderType',
    'tradingSymbol', 'customSymbol', 'securityId', 'tradedQuantity',
    'tradedPrice', 'isin', 'instrument', 'sebiTax', 'stt', 'brokerageCharges',
    'serviceTax', 'exchangeTransactionCharges', 'stampDuty', 'createTime',
    'updateTime', 'exchangeTime', 'drvExpiryDate', 'drvOptionType',
    'drvStrikePrice'
]

TradesHistoryColumnsToKeep = [
    'orderId',
    'transactionType',
    'exchangeSegment',
    'tradingSymbol',
    'customSymbol',
    'tradedQuantity',
    'tradedPrice',
]

TradesHistoryColumnsToRemove = set(TradesHistoryColumns).difference(
    set(TradesHistoryColumnsToKeep))

FundColumns = [
    "dhanClientId",
    "availabelBalance",
    "sodLimit",
    "collateralAmount",
    "receiveableAmount",
    "utilizedAmount",
    "blockedPayoutAmount",
    "withdrawableBalance",
]

FundColumnsToKep = [
    "availabelBalance",
    "sodLimit",
    "collateralAmount",
    "receiveableAmount",
    "utilizedAmount",
    "blockedPayoutAmount",
    "withdrawableBalance",
]

FundColumnsToRemove = set(FundColumns).difference(set(FundColumnsToKep))


class Fields:
  OrderStatus = 'orderStatus'


GAINERS_DIR = 'data/gainers'


class OrderStatusVals:
  CANCELLED = 'CANCELLED'
  REJECTED = 'REJECTED'
  TRADED = 'TRADED'
  PENDING = 'PENDING'


class StatusDhan(str, Enum):
  SUCCESS = 'success'
  FAILURE = 'failure'

  def __str__(self) -> str:
    return self.value


class DhanScripColumns:
  SEM_TRADING_SYMBOL = 'SEM_TRADING_SYMBOL'
  SEM_INSTRUMENT_NAME = 'SEM_INSTRUMENT_NAME'
  SEM_EXM_EXCH_ID = 'SEM_EXM_EXCH_ID'
  SEM_CUSTOM_SYMBOL = 'SEM_CUSTOM_SYMBOL'
  SEM_SMST_SECURITY_ID = 'SEM_SMST_SECURITY_ID'
  SEM_EXPIRY_CODE = 'SEM_EXPIRY_CODE'
  SEM_EXPIRY_DATE = 'SEM_EXPIRY_DATE'
  SEM_STRIKE_PRICE = 'SEM_STRIKE_PRICE'
  SEM_OPTION_TYPE = 'SEM_OPTION_TYPE'
  SEM_TICK_SIZE = 'SEM_TICK_SIZE'
  SEM_EXPIRY_FLAG = 'SEM_EXPIRY_FLAG'
