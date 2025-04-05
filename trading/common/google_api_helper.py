import os
import sys
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Path to your service account key file
SERVICE_ACCOUNT_FILE = 'trading/common/google-sheets.json'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']


class GoogleSheetsApi:

  def __init__(self) -> None:
    creds = None

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(SERVICE_ACCOUNT_FILE):
      creds = Credentials.from_authorized_user_file(SERVICE_ACCOUNT_FILE,
                                                    SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
      else:
        file_json = 'trading/configs/client_secret_157272493243-ll2il8t90jm1mppo1sm0vnootoufrnd3.apps.googleusercontent.com.json'

        flow = InstalledAppFlow.from_client_secrets_file(file_json, SCOPES)
        creds = flow.run_local_server(port=0)
      # Save the credentials for the next run
      with open(SERVICE_ACCOUNT_FILE, "w") as token:
        token.write(creds.to_json())

    # Build the service
    self.service = build('sheets',
                         'v4',
                         credentials=creds,
                         cache_discovery=False)

  def read_sheet_as_df(self, sheet_id, range_name='Raw!A1:Z100'):

    # Call the Sheets API
    sheet = self.service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id,
                                range=range_name).execute()
    values = result.get('values', [])

    # Convert to DataFrame if data is present
    if values:
      df = pd.DataFrame(values[1:], columns=values[0])
      print(df)
      return df
    else:
      print('No data found.')

    return pd.DataFrame()


import os
import pandas as pd
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build


class GoogleSheetsApi:

  def __init__(self) -> None:
    creds = None

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(SERVICE_ACCOUNT_FILE):
      creds = Credentials.from_authorized_user_file(SERVICE_ACCOUNT_FILE,
                                                    SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
      if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
      else:
        file_json = 'trading/configs/client_secret_1572t90jm1mppo1sm0vnootoufrnd3.apps.googleusercontent.com.json'
        flow = InstalledAppFlow.from_client_secrets_file(file_json, SCOPES)
        creds = flow.run_local_server(port=0)
      # Save the credentials for the next run
      with open(SERVICE_ACCOUNT_FILE, "w") as token:
        token.write(creds.to_json())

    # Build the service
    self.service = build('sheets',
                         'v4',
                         credentials=creds,
                         cache_discovery=False)

  def read_sheet_as_df(self, sheet_id, range_name='Raw!A1:Z100'):
    # Call the Sheets API
    sheet = self.service.spreadsheets()
    result = sheet.values().get(spreadsheetId=sheet_id,
                                range=range_name).execute()
    values = result.get('values', [])

    # Convert to DataFrame if data is present
    if values:
      df = pd.DataFrame(values[1:], columns=values[0])
      print(df)
      return df
    else:
      print('No data found.')
      return pd.DataFrame()

  def update_sheet_with_df(self, sheet_id, df, range_name='Raw!A1:Z'):
    """
        Updates a Google Sheet with new data from a DataFrame, avoiding duplicates based on the first column (datetime).

        Args:
            sheet_id (str): The ID of the Google Sheet to update
            df (pandas.DataFrame): DataFrame containing the new data
            range_name (str): The range to update in A1 notation

        Returns:
            bool: True if successful, False otherwise
        """
    try:
      # First, read existing data
      existing_df = self.read_sheet_as_df(sheet_id, range_name)

      if existing_df.empty:
        # If sheet is empty, write the entire DataFrame including headers
        values = [df.columns.tolist()] + df.values.tolist()
        body = {'values': values}

        self.service.spreadsheets().values().update(spreadsheetId=sheet_id,
                                                    range=range_name,
                                                    valueInputOption='RAW',
                                                    body=body).execute()
        return True

      # Get the name of the first column (datetime column)
      datetime_col = existing_df.columns[0]

      # Convert datetime columns to same format if needed
      if pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = df[datetime_col].astype(str)
      if pd.api.types.is_datetime64_any_dtype(existing_df[datetime_col]):
        existing_df[datetime_col] = existing_df[datetime_col].astype(str)

      # Find new rows by comparing datetime column
      existing_dates = set(existing_df[datetime_col].astype(str))
      new_rows = df[~df[datetime_col].astype(str).isin(existing_dates)]

      if new_rows.empty:
        print("No new data to add.")
        return True

      # Append new rows to the sheet
      # Calculate the start row for appending (existing rows + 1 for header + 1 for next row)
      start_row = len(existing_df) + 2
      append_range = f"{range_name.split('!')[0]}!A{start_row}"

      body = {'values': new_rows.values.tolist()}

      self.service.spreadsheets().values().append(
          spreadsheetId=sheet_id,
          range=append_range,
          valueInputOption='RAW',
          insertDataOption='INSERT_ROWS',
          body=body).execute()

      print(f"Added {len(new_rows)} new rows to the sheet.")
      return True

    except Exception as e:
      print(f"Error updating sheet: {str(e)}")
      return False


if __name__ == '__main__':
  gapi = GoogleSheetsApi()

  df = gapi.read_sheet_as_df('1LQDvYuV87AJzSoa9Vy6Knk')
  print(f'received df = \n{df}')
