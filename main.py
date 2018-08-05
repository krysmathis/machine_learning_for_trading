

import util
import pandas as pd
import portfolio as portfolio

#Daily Portfolio Value
start_val = 1000000
start_date = '2010-01-01'
end_date = '2010-12-31'
symbols = ['AXP', 'HPQ','IBM','HNZ']
allocs = [0.0,0.0,0.0,1.0]

dates = pd.date_range(start_date, end_date)  # one month only
df_prices = util.get_yahoo_data(symbols,start_date, end_date)
df_prices.describe()

cr, adr, sddr, sr = \
    portfolio.compute_portfolio_stats(allocs=allocs, \
    prices=df_prices, \
    rfr = 0, sf = 252.0)

print(sr,sddr,adr,cr)