'''
Use this python file when there's a plit in the data file.
Finam does not allow for splits in the data file.

The only way to fix it would be to check for splits automatically and then fix them.'''

import os, sys
import pandas as pd
import re
import numpy as np
import argparse

FILE = '/tmp/p/AAPL-MINUTES5.csv'
FIXED_FILE = '/tmp/p/AAPL-MINUTES5-FIXED.csv'


def add_arguments(parser):
    parser = argparse.ArgumentParser(description='Fix splits in data file.')
    parser.add_argument('--file', type=str, default=FILE)
    parser.add_argument('--fixed-file', type=str, default=FIXED_FILE)
    return parser.parse_args()


def get_splits(df):
    '''
    We need to find splits that happened.
    Splits will tell what multiplier to use for the prices.
    By the end of the function, you have a list of "index when the split happened" and "multiplier"
    '''
    splits_by_idx = []
    df['pct_change'] = df['<OPEN>'].pct_change()
    df['more_than_50'] = df['pct_change'] < -0.45
    column_names = df.columns.tolist()
    print(f'column_names = {column_names}')
    a2 = df.where(df['more_than_50'] == True).dropna().index

    for idx in list(a2):
        split_ratio = round(df.iloc[idx - 1]['<OPEN>'] /
                            df.iloc[idx]['<OPEN>'])
        # print(f'open price = {df.iloc[idx]["<OPEN>"]}')
        # print(f'idx = {idx} ; split_Ratio = {split_ratio}')
        rows_to_show = 1
        rng_df = df[idx - rows_to_show:idx + rows_to_show]
        # print(f'rng_df = {rng_df}')
        splits_by_idx.append((idx, split_ratio))

    curr = 1
    # We want cumulative numbers, so we need to multiply the previous one
    multipliers = []
    for idx, v in splits_by_idx[::-1]:
        curr = curr * v
        multipliers.append((idx, curr))
    return multipliers[::-1]


def show_splits_df_demo(df, splits_by_idx):
    '''
    Show the df around the times when the split happened.
    Use this to confirm that the prices are as expected'''
    rows_to_show = 1
    for idx, v in splits_by_idx:
        rng_df = df[idx - rows_to_show:idx + rows_to_show]
        print(f'rng_df = {rng_df}')


def redo_market_data_by_splits(df, splits):
    """
    HLOC to be reduced by the amount of "cumulative" split that happened
    Vol needs to be multiplied
    """
    splits.append((len(df), 1))
    split_index = 0
    split_to_use = splits[split_index]

    for idx, data in df.iterrows():
        if idx >= split_to_use[0]:
            split_index += 1
            split_to_use = splits[split_index]

        if idx == 0:
            continue

        df.loc[idx, '<OPEN>'] = data['<OPEN>'] / split_to_use[1]
        df.loc[idx, '<CLOSE>'] = data['<CLOSE>'] / split_to_use[1]
        df.loc[idx, '<HIGH>'] = data['<HIGH>'] / split_to_use[1]
        df.loc[idx, '<LOW>'] = data['<LOW>'] / split_to_use[1]
        df.loc[idx, '<VOL>'] = data['<VOL>'] * split_to_use[1]

    # show_splits_df_demo(df, splits)


def main():
    args = add_arguments(sys.argv)
    df = pd.read_csv(args.file)
    # df['Date'] = pd.to_datetime(df['Date'])
    splits = get_splits(df)
    # print(f'splits = {splits}')

    redo_market_data_by_splits(df, splits)
    remove_columns(df)
    show_splits_df_demo(df, splits)
    save_to_new_csv(df, args.fixed_file)


def save_to_new_csv(df, fixed_file):
    df.to_csv(fixed_file, index=False)


def remove_columns(df):
    df.drop(['pct_change', 'more_than_50'], axis=1, inplace=True)


if __name__ == '__main__':
    main()
