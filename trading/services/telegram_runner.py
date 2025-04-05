import os
import sys
import asyncio
import time
import pandas as pd

from trading.common import utils

logger = utils.get_logger('telegram_runner')

from trading.stocks import dhan, dhan_utils
from trading.common import util_commands
from trading.common import util_prints
from trading.configs.configs_weekend import ConfigManager, WEEKEND_CONFIGS, DEFAULT_TAG

import telebot
from telebot import types, async_telebot

TOKEN = os.environ.get('TELEGRAM_TOKEN', TOKEN)

# bot.infinity_polling()
PARSE_MODE = 'MarkdownV2'

API_ID = os.environ.get('TELEGRAM_API_ID', API_ID)
API_HASH = os.environ.get('TELEGRAM_API_HASH', API_HASH)

MY_USER_ID = os.environ.get('MY_USER_ID', MY_USER_ID)

bot = async_telebot.AsyncTeleBot(TOKEN)

SCRIPT_CODE = 'scripts/update-codes.sh'
REPLACEMENTS = {
    r'.': r'\.',
    r'{': r'\{',
    r'}': r'\}',
    r'-': r'\-',
    r'_': r'\_',
    r'=': r'\:',
    r'(': r'\(',
    r')': r'\)',
    r'<': r'\<',
    r'>': r'\>',
}


def escape_msg(msg, replacements=REPLACEMENTS):
  for old, new in replacements.items():
    msg = msg.replace(old, new)
  return msg


@bot.message_handler(commands=['start'])
async def start_message(message):
  await bot.send_message(message.chat.id, 'Hello!')


def read_file(filename):
  return pd.read_csv(filename)


@bot.message_handler(commands=['config'])
async def show_config(message):
  logger.info(f'received command config')
  markup = types.ReplyKeyboardMarkup(row_width=4)

  b0 = types.KeyboardButton("Show Configs")
  b1 = types.KeyboardButton(text="wk 🔀 Opening")
  b2 = types.KeyboardButton(text="wk 🔀 Closing")
  b3 = types.KeyboardButton(text='wk 🔀 Enforce position cap')
  b4 = types.KeyboardButton(text="wk 🔀 SL for Positions")
  b5 = types.KeyboardButton(text="wk 🔀 SL for Holdings")
  b6 = types.KeyboardButton(text="wk SL for Positions")
  b7 = types.KeyboardButton(text="wk SL for Holdings")
  b8 = types.KeyboardButton(text="wk notional")
  b9 = types.KeyboardButton(text="wk num tickers")

  markup.add(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9)

  await bot.send_message(message.chat.id,
                         "Choose a button for configs",
                         reply_markup=markup)


def _get_markups(key):
  # You can then create new buttons in response to the user's input
  markup = types.ReplyKeyboardMarkup(row_width=2)
  new_button1 = types.KeyboardButton(f"wk {key} SL = 1")
  new_button2 = types.KeyboardButton(f"wk {key} SL = 2")
  new_button3 = types.KeyboardButton(f"wk {key} SL = 3")
  new_button4 = types.KeyboardButton(f"wk {key} SL = 4")
  markup.add(new_button1, new_button2, new_button3, new_button4)
  return markup


@bot.message_handler(regexp=r'^(?i:wk notional)$')
async def configs_update_wk_notional(message):

  # You can then create new buttons in response to the user's input
  markup = types.ReplyKeyboardMarkup(row_width=3)
  new_button1 = types.KeyboardButton(f"wk notional = 15000")
  new_button2 = types.KeyboardButton(f"wk notional = 17500")
  new_button3 = types.KeyboardButton(f"wk notional = 20000")
  new_button4 = types.KeyboardButton(f"wk notional = 22500")
  markup.add(new_button1, new_button2, new_button3, new_button4)
  await bot.send_message(message.chat.id,
                         'Choose notional value',
                         reply_markup=markup)


@bot.message_handler(regexp=r'^(?i:wk notional = (\d+(?:\.\d+)?))$')
async def configs_update_wk_notional_value(message):
  logger.info(f'message text = {message.text}')
  key = WEEKEND_CONFIGS.NOTIONAL_AMOUNT_FOR_POSITIONS
  val = int(message.text.split('=')[-1])
  config_manager = ConfigManager()
  config_manager.change_value(DEFAULT_TAG, key, val)
  res = get_all_configs()
  await bot.send_message(message.chat.id, res, parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:wk num tickers = (\d+(?:\.\d+)?))$')
async def configs_update_wk_num_tickers(message):
  logger.info(f'message text = {message.text}')
  key = WEEKEND_CONFIGS.NUM_OF_TICKERS
  val = int(message.text.split('=')[-1])
  config_manager = ConfigManager()
  config_manager.change_value(DEFAULT_TAG, key, val)
  res = get_all_configs()
  await bot.send_message(message.chat.id, res, parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:wk num tickers)$')
async def configs_update_wk_num_tickers_value(message):
  markup = types.ReplyKeyboardMarkup(row_width=3)
  new_button1 = types.KeyboardButton(f"wk num tickers = 3")
  new_button2 = types.KeyboardButton(f"wk num tickers = 4")
  new_button3 = types.KeyboardButton(f"wk num tickers = 5")
  markup.add(new_button1, new_button2, new_button3)
  await bot.send_message(message.chat.id,
                         'Choose number of tickers',
                         reply_markup=markup)


@bot.message_handler(regexp=r'^(?i:wk SL for Positions)$')
async def configs_update_sl_positions(message):
  markup = _get_markups('Positions')
  await bot.send_message(message.chat.id,
                         'Choose SL Value for positions',
                         reply_markup=markup)


@bot.message_handler(regexp=r'^(?i:wk SL for Holdings)$')
async def configs_update_sl_holdings(message):
  markup = _get_markups('Holdings')
  await bot.send_message(message.chat.id,
                         'Choose SL Value for Holdings',
                         reply_markup=markup)


@bot.message_handler(regexp=r'^(?i:wk .* SL = (\d|\d\.\d))$')
async def configs_update_sl_position_value(message):
  logger.info(f'message text = {message.text}')
  key = WEEKEND_CONFIGS.STOP_LOSS_FOR_POSITIONS if 'position' in message.text.lower(
  ) else WEEKEND_CONFIGS.STOP_LOSS_FOR_HOLDINGS
  val = float(message.text.split('=')[-1])
  config_manager = ConfigManager()
  config_manager.change_value(DEFAULT_TAG, key, val)
  res = get_all_configs()
  await bot.send_message(message.chat.id, res, parse_mode=PARSE_MODE)


def toggle_config(config_name):
  config_manager = ConfigManager()
  config_manager.toggle_config(DEFAULT_TAG, config_name)
  res = get_all_configs()
  return res


@bot.message_handler(regexp=r'^(?i:Show Configs)$')
async def configs_show_configs(message):
  msg = 'Unknown exception'
  try:
    res = get_all_configs()
    logger.info(f'configs = {res}')
    msg = res
  except Exception as e:
    logger.exception(f'exception occurred : {str(e)}')
  finally:
    await bot.send_message(message.chat.id, msg, parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:wk 🔀 Opening)$')
async def configs_toggle_opening(message):
  res = toggle_config(WEEKEND_CONFIGS.ENABLE_OPEN_POSITIONS)
  logger.info(f'res = \n{res}')
  await bot.send_message(message.chat.id, res, parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:wk 🔀 Closing)$')
async def configs_toggle_closing(message):
  res = toggle_config(WEEKEND_CONFIGS.ENABLE_CLOSE_POSITIONS)
  logger.info(f'res = \n{res}')
  await bot.send_message(message.chat.id, res, parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:wk 🔀 Enforce position cap)$')
async def configs_toggle_enforce(message):
  res = toggle_config(WEEKEND_CONFIGS.ENABLE_ENFORCE_POSITION_CAP)
  logger.info(f'res = \n{res}')
  await bot.send_message(message.chat.id, res, parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:wk 🔀 SL for Positions)$')
async def configs_toggle_wk_sl_positions(message):
  res = toggle_config(WEEKEND_CONFIGS.ENABLE_STOP_LOSS_FOR_POSITIONS)
  logger.info(f'res = \n{res}')
  await bot.send_message(message.chat.id, res, parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:wk 🔀 SL for Holdings)$')
async def configs_toggle_wk_sl_holdings(message):
  res = toggle_config(WEEKEND_CONFIGS.ENABLE_STOP_LOSS_FOR_HOLDINGS)
  logger.info(f'res = \n{res}')
  await bot.send_message(message.chat.id, res, parse_mode=PARSE_MODE)


def get_all_configs():
  config_manager = ConfigManager()
  results = config_manager.show_all_configs()
  msg = f"Configs: \n```{results}```"
  return escape_msg(msg)


@bot.message_handler(commands=['dhan'])
async def dhan_requests(message):
  logger.info(f'received command dhan')
  keyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
  button0 = types.KeyboardButton(text='Funds')
  button1 = types.KeyboardButton(text="Orders")
  button2 = types.KeyboardButton(text="Trades")
  button3 = types.KeyboardButton(text="Holdings")
  button4 = types.KeyboardButton(text="Positions")
  button5 = types.KeyboardButton(text="PnL")
  button6 = types.KeyboardButton(text="Gainers")

  keyboard.add(button0,
               button1,
               button2,
               button3,
               button4,
               button5,
               button6,
               row_width=4)
  await bot.send_message(message.chat.id,
                         'Pls select your dhan request',
                         reply_markup=keyboard)


@bot.message_handler(regexp=r'^(?i:gainers)$')
async def dhan_gainers(message):
  gainers = dhan.read_gainers_list()
  res = gainers.to_string(index=False)
  logger.info(f'gainers = \n{res}')
  await bot.send_message(message.chat.id, f"Gainers: \n```{res}```")


@bot.message_handler(regexp=r'^(?i:PnL)$')
async def dhan_daily_run(message):
  filtered_rows = dhan_utils.DhanYfinanceTickers.get_pnl_view_from_closed_positions_or_holdings(
  )
  total_pnl = f'{filtered_rows["pnl"].sum():.2f}'
  logger.info(f'total_pnl = {total_pnl}')
  res = filtered_rows.to_string(index=False)
  logger.info(f'filtered_rows = \n{res}')

  full_msg = f'''Pnl: \n```
{res}``` \n\n Total PnL: {total_pnl}'''
  full_msg = escape_msg(full_msg)
  logger.info(f'full_msg = {full_msg}')
  await bot.send_message(message.chat.id, full_msg, parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:Orders)$')
async def dhan_orders(message):
  dhan_ins = dhan.DhanTracker()
  orders = dhan_ins.get_orders(only_valid_orders=False)
  df = pd.DataFrame(orders)
  orders_s = dhan.show_df_in_table_format('orders', df)
  logger.info(f'show_df_in_table_format = {orders_s}')
  await bot.send_message(message.chat.id,
                         f"Orders: \n```{orders_s}```",
                         parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:funds)$')
async def dhan_orders(message):
  dhan_ins = dhan.DhanTracker()
  fund_limits = dhan_ins.get_fund_limits()

  data = {
      k: f"{v:,}" if isinstance(v, (int, float)) else v
      for k, v in fund_limits.items()
  }

  data_s = util_prints.get_pretty_print(data)
  logger.info(f'data_s = {data_s}')

  await bot.send_message(message.chat.id,
                         f"Funds: \n```{data_s}```",
                         parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:trades)$')
async def dhan_trades(message):
  dhan_ins = dhan.DhanTracker()
  trades = dhan_ins.get_trades()
  df = pd.DataFrame(trades)
  trades_s = dhan.show_df_in_table_format('trades', df)
  await bot.send_message(message.chat.id,
                         f"Trades: \n```{trades_s}```",
                         parse_mode=PARSE_MODE)


@bot.message_handler(regexp=r'^(?i:holdings)$')
async def dhan_holdings(message):

  dhan_ins = dhan.DhanTracker()
  try:
    holdings = dhan_ins.get_holdings()
    df = pd.DataFrame(holdings)
    holdings_s = dhan.show_df_in_table_format('holdings', df)
    logger.info(f'holdings_s = {holdings_s}')
    await bot.send_message(message.chat.id,
                           f"Holdings: \n```{holdings_s}```",
                           parse_mode=PARSE_MODE)
  except Exception as e:
    logger.exception(f'Exception in dhan_holdings: {str(e)}')


@bot.message_handler(regexp=r'^(?i:positions)$')
async def dhan_positions(message):

  dhan_ins = dhan.DhanTracker()
  try:
    positions = dhan_ins.get_positions()
    df = pd.DataFrame(positions)
    logger.info(f'positions={positions}')
    # positions = read_file('data/dhan-prod/2023-12-04-positions-18-18-40.csv')
    positions_s = dhan.show_df_in_table_format('positions', df)
    logger.info(f'positions_s = {positions_s}')
    await bot.send_message(message.chat.id,
                           f"Positions: \n```{positions_s}```",
                           parse_mode=PARSE_MODE)
  except Exception as e:
    logger.exception(f'Exception in dhan_positions: {str(e)}')


@bot.message_handler(commands=['logs'])
async def log_requests(message):

  keyboard = types.ReplyKeyboardMarkup(row_width=1, resize_keyboard=True)
  button1 = types.KeyboardButton(text="codes")
  # button2 = types.KeyboardButton(text="Trades")
  # button3 = types.KeyboardButton(text="Positions")
  # button3 = types.KeyboardButton(text="Daily Run")

  keyboard.add(button1)
  await bot.send_message(message.chat.id,
                         'Which logs do you want to see',
                         reply_markup=keyboard)


@bot.message_handler(regexp=r'^codes$')
async def dhan_positions(message):

  try:
    results = util_commands.run_cmd('scripts/update-codes.sh')
    logger.info(f'res = {results}')

    await bot.send_message(message.chat.id,
                           f"update code logs: \n```{results}```",
                           parse_mode=PARSE_MODE)
  except Exception as e:
    logger.exception(f'Exception in dhan_positions: {str(e)}')


@bot.message_handler(func=lambda message: True)
async def handle_buttons(message):
  # You can handle different button clicks here
  if message.text == "Button 1":
    bot.send_message(message.chat.id, "You clicked Button 1!")
  elif message.text == "Button 2":
    bot.send_message(message.chat.id, "You clicked Button 2!")
  # Add more button handling as needed

  # You can then create new buttons in response to the user's input
  markup = types.ReplyKeyboardMarkup(row_width=2)
  new_button1 = types.KeyboardButton("New Button 1")
  new_button2 = types.KeyboardButton("New Button 2")
  markup.add(new_button1, new_button2)

  await bot.send_message(message.chat.id,
                         "Choose a new button:",
                         reply_markup=markup)


def send_image_with_text(image_path,
                         index_name='SPX',
                         token='GET_TOKEN_FROM_ENV_VARIABLE'):
  """
    Send image with formatted date text to Telegram
    Args:
        image_path: Path to the image file
        index_name: Name of the index (default 'SPX')
        token: Telegram bot token
    """
  try:
    from datetime import datetime
    import telebot
    from telebot.formatting import escape_markdown

    # Initialize bot
    bot = telebot.TeleBot(token, parse_mode=PARSE_MODE)

    # Generate formatted date text
    current_date = datetime.now().strftime("%Y%m%d")
    caption_text = escape_markdown(f"{index_name} {current_date}")

    # Send image with caption
    with open(image_path, 'rb') as photo:
      bot.send_photo(chat_id=MY_USER_ID,
                     photo=photo,
                     caption=caption_text,
                     parse_mode=PARSE_MODE)

    logger.info(f'Sent image {image_path} with caption {caption_text}')

  except Exception as e:
    logger.exception(f'Exception in telegram send_image_with_text: {str(e)}')


def send_text(
    text_lines,
    token='GET_TOKEN_FROM_ENV_VARIABLE',
):
  try:
    bot_normal = telebot.TeleBot(token, parse_mode=PARSE_MODE)
    full_text = escape_msg('\n'.join(text_lines))
    logger.info(f'full_text = {full_text}')
    bot_normal.send_message(MY_USER_ID, full_text, parse_mode=PARSE_MODE)
  except Exception as e:
    logger.exception(f'Exception in telegram send_text: {str(e)}')


if __name__ == '__main__':
  asyncio.run(bot.polling())
