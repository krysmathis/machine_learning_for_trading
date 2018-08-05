
import os
import pandas as pd
import numpy as np
import math


def compute_portfolio_stats(allocs,prices,rfr=0, sf=252):
    """
        Returns the cumulative return, average daily return,
        standard deviation of daily return and sharpe ratio
        
        Parameters:
        prices = dataframe of adj close prices with each stock
                 listed as the column header with dates as index
        allocs = a list of percentages that add to 100
        rfr = risk free rate
        sf = sampling frequency, daily = 252, weekly = 52, monthly = 12
    """
    # normalized prices
    normed = prices/prices.iloc[0]
    
    alloced = normed * allocs

    # position values
    start_val = 1 # included to simplify adding ability to calc as $
    pos_vals = alloced * start_val

    # portfolio value
    port_val = pos_vals.sum(axis=1)
    
    daily_rets = port_val/port_val.shift(1) - 1
    daily_rets = daily_rets[1:]

    # cumulative return
    cr = port_val.iloc[-1]/port_val.iloc[0] -1

    # avg daily return
    adr = daily_rets.mean()

    # std dev of daily return
    sddr = daily_rets.std()
    
    #sharpe_ratio
    k = math.sqrt(252)
        
    sr = k * ((daily_rets - 0).mean() / daily_rets.std())
    
    return cr, adr, sddr, sr


