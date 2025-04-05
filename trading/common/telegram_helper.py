import telebot
from telethon.sync import TelegramClient
from telethon.tl.types import InputPeerUser, InputPeerChannel
from telethon import TelegramClient, sync, events

API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')

TOKEN = os.getenv('TELEGRAM_TOKEN')
MY_USER_ID = int(os.getenv('TELEGRAM_MY_USER_ID'))
MY_PHONE_NUMBER = os.getenv('TELEGRAM_MY_PHONE_NUMBER')


def send(message):

  # your phone number
  phone = MY_PHONE_NUMBER

  # creating a telegram session and assigning
  # it to a variable client
  client = TelegramClient('session', API_ID, API_HASH)

  # connecting and building the session
  client.connect()

  # in case of script ran first time it will
  # ask either to input token or otp sent to
  # number or sent or your telegram id
  if not client.is_user_authorized():

    client.send_code_request(phone)

    # signing in the client
    client.sign_in(phone, input('Enter the code: '))

  try:
    # receiver user_id and access_hash, use
    receiver = InputPeerUser(MY_USER_ID, 0)
    # my user_id and access_hash for reference

    # sending message using telegram client
    client.send_message(receiver, message, parse_mode='html')
  except Exception as e:

    # there may be many error coming in while like peer
    # error, wrong access_hash, flood_error, etc
    print(e)

  # disconnecting the telegram session
  client.disconnect()


if __name__ == '__main__':
  send('hello')
