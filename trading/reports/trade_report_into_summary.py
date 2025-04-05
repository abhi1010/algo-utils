import os
import sys
import argparse
import datetime

import pandas as pd


# Function to analyze trades by strategy
def analyze_trades(data):
  report = {}

  # Group by Strategy
  grouped = data.groupby('Strategy')

  for strategy, group in grouped:
    total_trades = len(group)
    winning_trades = group[group['Σ PL'] > 0]
    losing_trades = group[group['Σ PL'] <= 0]

    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    avg_pl_winners = winning_trades['% PL'].mean() if win_count > 0 else 0
    avg_pl_losers = losing_trades['% PL'].mean() if loss_count > 0 else 0

    avg_days_winners = winning_trades['Σ No.of days'].mean(
    ) if win_count > 0 else 0
    avg_days_losers = losing_trades['Σ No.of days'].mean(
    ) if loss_count > 0 else 0

    report[strategy] = {
        'Total Trades': total_trades,
        'Winning Trades': win_count,
        'Losing Trades': loss_count,
        'Win Rate (%)': win_rate,
        'Avg PL% (Winners)': avg_pl_winners,
        'Avg PL% (Losers)': avg_pl_losers,
        'Avg Days Opened (Winners)': avg_days_winners,
        'Avg Days Opened (Losers)': avg_days_losers,
    }

  return report


def main():

  # Load the CSV file
  file_path = 'data/dhan-202409-trade_summary-updated.xlsx'  # Replace with your actual file path
  data = pd.read_excel(file_path)

  # Generate the report
  trade_report = analyze_trades(data)

  # Print the report
  for strategy, metrics in trade_report.items():
    print(f"\nStrategy: {strategy}")
    for metric, value in metrics.items():
      print(f"{metric}: {value}")

  # Convert report to DataFrame for saving to Excel
  report_df = pd.DataFrame.from_dict(trade_report, orient='index')

  # Save the report to an Excel file
  output_file_path = 'data/trade_analysis_report.xlsx'  # Specify your desired output file name
  report_df.to_excel(output_file_path, sheet_name='Trade Analysis')

  print(f"Trade analysis report saved to {output_file_path}")


if __name__ == '__main__':
  main()
