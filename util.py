"""MLT: Utility code.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

import os
import pandas as pd
import numpy as np
import fix_yahoo_finance as yf
import math

def symbol_to_path(symbol, base_dir=None):
    """Return CSV file path given ticker symbol."""
    if base_dir is None:
        base_dir = os.environ.get("MARKET_DATA_DIR", '../data/')
    return os.path.join(base_dir, "{}.csv".format(str(symbol)))

def get_data(symbols,start_date,end_date, addSPY=True, colname = 'Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    dates = pd.date_range(start_date, end_date)
    
    df = pd.DataFrame(index=dates)
    
    includes_spy = True if 'SPY' in symbols else False

    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent
        symbols = ['SPY'] + symbols

    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol,'data'), index_col='Date',
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])
        df_temp = df_temp.rename(columns={colname: symbol})
        df = df.join(df_temp)
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])

    if not includes_spy: 
        symbols.remove('SPY')
    print('*** USING TEST DATA ***')
    return df[symbols]

"""Added by Krys Mathis"""
def get_yahoo_data(symbols, start_date, end_date, addSPY=True, colname = 'Adj Close'):
    """Read stock data (adjusted close) for given symbols from CSV files."""
    
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)
    
    includes_spy = True if 'SPY' in symbols else False
    
    if not includes_spy and addSPY:  # add SPY for reference, if absent
        symbols.insert(0, 'SPY')

    for symbol in symbols:

        df_yahoo = yf.download(symbol, start_date, end_date)
        df_yahoo = df_yahoo.rename(columns={colname: symbol})
        df_yahoo = df_yahoo[symbol]
        df = df.join(df_yahoo)
        
        if symbol == 'SPY':  # drop dates SPY did not trade
            df = df.dropna(subset=["SPY"])
        
    if not includes_spy: 
        symbols.remove('SPY')
    
    return df[symbols]


def plot_data(df, title="Stock prices", xlabel="Date", ylabel="Price"):
    import matplotlib.pyplot as plt
    """Plot stock prices with a custom title and meaningful axis labels."""
    ax = df.plot(title=title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def get_orders_data_file(basefilename):
    return open(os.path.join(os.environ.get("ORDERS_DATA_DIR",'orders/'),basefilename))

def get_learner_data_file(basefilename):
    return open(os.path.join(os.environ.get("LEARNER_DATA_DIR",'Data/'),basefilename),'r')

def get_robot_world_file(basefilename):
    return open(os.path.join(os.environ.get("ROBOT_WORLDS_DIR",'testworlds/'),basefilename))
