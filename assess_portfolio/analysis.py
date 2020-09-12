"""Analyze a portfolio.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved
"""

"""student:Erika Gemzer | gth659q | Summer 2018"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from util import get_data, plot_data
import datetime as dt


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(start_date=dt.datetime(2008, 1, 1), end_date=dt.datetime(2009, 1, 1), \
                     syms=['GOOG', 'AAPL', 'GLD', 'XOM'], \
                     allocs=[0.1, 0.2, 0.3, 0.4], \
                     start_val=1000000, rfr=0.0, sf=252.0, \
                     gen_plot=False):
    """
    Inputs:
        start_date: first day to look at the portfolio (YYYY, MM, DD)
        end_date: last day to look at the portfolio (YYYY, MM, DD)
        syms: a list of stock tickers in the portfolio
        allocs: a list of the relative allocations (floats) of each equity in the portfolio
        start_val: starting value of the portfolio (dollars), a float
        rfr: risk free rate of return, a float
        sf: sampling frequency (number of days the stock traded), a float
        gen_plot: if True, generates plots and saves to folder. Otherwise, no plots generated.
    Returns:
        cr: cumulative return, a numpy 64 bit float
        adr: average daily return (if sf == 252 this is daily return), a numpy 64 bit float
        sddr: std of daily returns, a numpy 64 bit float
        sr: sharpe ratio, risk-adjusted returns, a numpy 64 bit float
        ev: end value of the portfolio in the date range, a numpy 64 bit float
    """

    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    fill_missing_values(prices_all)
    plot_data(prices_all, title="Stock prices", xlabel="Date", ylabel="Price")
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    port_val = prices_SPY
    port_val = get_daily_portfolio_val(prices, allocs, start_val)

    # Get portfolio statistics (note: std_daily_ret = volatility)
    cr, adr, sddr, sr = compute_portfolio_stats(prices, port_val, sf, rfr)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        normed_SPY = prices_SPY / prices_SPY.ix[0, :]  # normalized by initial equities value and alloc
        normed_port = port_val / port_val.ix[0, :]  # normalized by initial stock value and alloc
        df_temp = pd.concat([normed_port, normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp, title="Daily Portfolio Value and SPY", xlabel="Date", ylabel="Normalized Price")
        plt.savefig('result.png')
        plt.switch_backend('Agg')
        pass

    # compute portfolio end value
    ev = port_val[-1]  # one way to calculate ev

    return cr, adr, sddr, sr, ev

def get_daily_portfolio_val(prices, allocs, sv): #computes daily portfolio value
    """
    Inputs:
        prices: a pandas dataframe of prices for various stock tickers, indexed by date
        allocs: a list of the relative allocations (floats) of each equity in the portfolio
        sv: a float, portfolio starting value
    Returns:
        port_val: a pandas dataframe/series of total portfolio value, indexed by date
    """
    prices_copy = prices.copy()
    normed = prices_copy / prices.ix[0, :]  # normalized by first row
    alloced = normed * allocs  # multiply by the allocations for each equity
    pos_vals = alloced * sv  # positions values
    port_val = pos_vals.sum(axis=1)  # portfolio value
    return port_val

def compute_portfolio_stats(prices, port_val, sf, rfr):
    # Get portfolio statistics (note: std_daily_ret = volatility)
    """cr: Cumulative return
    adr: Average period return (if sf == 252 this is daily return)
    sddr: Standard deviation of daily return
    sr: Sharpe ratio"""
    daily_rets = prices.copy()  # copy the data frame
    daily_rets[1:] = (prices[1:] / prices[:-1].values) - 1  # all rows from zero to the end
    daily_rets.ix[0, :] = 0  # set all values for row 0 to 0
    cr = (port_val[-1] / port_val[0]) - 1  # cumulative return
    # print cr  # debugging
    adr = ((port_val[1:].values / port_val[:-1].values)-1.0).mean()  # average daily return
    #print adr #debugging
    #print port_val[1:] #debugging
    #print (port_val[1:] / port_val[:-1].values)-1.0  #debugging
    sddr = ( (port_val[1:].values / port_val[:-1].values) - 1.0).std(ddof=1)  #std daily returns
    #print sddr  #debugging
    #print sf #debugging
    #print rfr #debugging
    sr = np.sqrt(sf) * ((adr - rfr) / sddr)  # sharpe ratio
    return cr, adr, sddr, sr


def fill_missing_values(df_data):
    """Fill missing values in data frame, in place."""
    df_data.fillna(method="ffill", inplace=True) # forward fill as best practice
    df_data.fillna(method="bfill", inplace=True) # backfill 2nd resort
    df_data.fillna(1.0, inplace=True) # last resort


def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,1,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    rfr = 0.0
    sample_freq = 252

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(start_date, end_date, \
                                             syms=symbols, \
                                             allocs=allocations, \
                                             start_val = start_val, rfr = rfr, sf = sample_freq,
                                             gen_plot=False)
    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    test_code()
