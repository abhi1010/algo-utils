import pandas as pd
from difflib import SequenceMatcher
import re
'''
Reads 2 files

1. MOMO results for all tickers
2. Current portfolio that i would have downloaded from dhan


Dhan portfolio doesn't have yfinance names.
I need those names so that I can compare the momo results and find out which
tickers to buy or sell in the next cycle.
This is used for that

'''


def clean_company_name(name):
  """Clean company names for better matching."""
  # Remove common suffixes and special characters
  name = re.sub(r'\.(NS|BO)$', '', str(name))
  name = re.sub(r'Limited|Ltd|WAM|LLC|Inc|Corp|ETF',
                '',
                name,
                flags=re.IGNORECASE)
  name = re.sub(r'[^\w\s]', ' ', name)
  return ' '.join(name.lower().split())


def get_similarity_ratio(str1, str2):
  """Calculate similarity ratio between two strings."""
  return SequenceMatcher(None, clean_company_name(str1),
                         clean_company_name(str2)).ratio()


def find_best_ticker_match(company_name, momo_df, threshold=0.6):
  """Find the best matching ticker for a company name."""
  best_ratio = 0
  best_match = None

  for _, row in momo_df.iterrows():
    ratio = get_similarity_ratio(company_name, row['Name'])
    if ratio > best_ratio and ratio >= threshold:
      best_ratio = ratio
      best_match = row

  return best_match


# Read the CSV files
try:
  momo_df = pd.read_csv('data/momo-nse/250204-all.csv')
  portfolio_df = pd.read_csv('Portfolio.csv')
except Exception as e:
  print(f"Error reading CSV files: {e}")
  exit(1)

# Create a new column for tickers
portfolio_df['Ticker'] = None
portfolio_df['Match_Score'] = None

# Find matches for each portfolio company
for idx, row in portfolio_df.iterrows():
  match = find_best_ticker_match(row['Name'], momo_df)
  if match is not None:
    portfolio_df.at[idx, 'Ticker'] = match['Ticker']
    portfolio_df.at[idx, 'Match_Score'] = get_similarity_ratio(
        row['Name'], match['Name'])

# Sort columns to put Ticker near the front
cols = portfolio_df.columns.tolist()
cols = ['Name', 'Ticker', 'Match_Score'] + [
    col for col in cols if col not in ['Name', 'Ticker', 'Match_Score']
]
portfolio_df = portfolio_df[cols]

# Save the updated portfolio
try:
  portfolio_df.to_csv('data/portfolio-updated.csv', index=False)
  print("Updated portfolio saved to data/portfolio-updated.csv")

  # Print matching results with proper formatting
  print("\nMatching Results:")
  for _, row in portfolio_df.iterrows():
    print(f"\nPortfolio Company: {row['Name']}")
    if pd.notna(row['Ticker']):
      print(f"Matched Ticker: {row['Ticker']}")
    else:
      print("Matched Ticker: No match")

    if pd.notna(row['Match_Score']):
      print(f"Match Score: {row['Match_Score']:.2f}")
    else:
      print("Match Score: N/A")
except Exception as e:
  print(f"Error saving updated portfolio: {e}")
