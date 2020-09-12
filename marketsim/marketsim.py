"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved

student:Erika Gemzer | gth659q | Summer 2018
passed all test cases on buffet01 on 6/29/2018.
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data


def compute_portvals(orders_file = "./additional_orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):
    """Summary: a market simulator that accepts trading orders and keeps track of a portfolio's value over time
    and then assesses the performance of that portfolio.

    Inputs:
        orders_file: the name of a file from which to read orders (string or file object)
        start_val: the starting value of the portfolio (initial cash available)
        commission: fixed amount in dollars charged for each transaction (both entry and exit)
        impact:  amount the price moves against the trader compared to the historical data at each transaction

    Returns: portvals, a dataframe with one column containing the value of the portfolio
    for each trading day (the index), from start_date to end_date, inclusive.
    """

    # Get the key info from the orders file
    start_date, end_date, ordersDF, syms_traded = order_info(orders_file)

    # Create a dataframe with adjusted close prices for the equities, plus all cash in the account
    pricesDF = get_data(syms_traded, pd.date_range(start_date, end_date)).dropna() #daily prices for each equity, including SPY (autoadded) with date index
    fill_missing_values(pricesDF) #fill any "nan" values
    pricesDF = pricesDF[syms_traded]  # only portfolio symbols
    pricesDF["cash"] = 1.0 #add cash position column, set to 1.0 for all rows (for now)
    # Initialize a dataframe to hold all of the trade information from the orders
    tradesDF = pd.DataFrame(np.zeros((pricesDF.shape)), pricesDF.index, pricesDF.columns) #tradesDF has the same structure as pricesDF, os we copy it but fill it with zeros

    # place trade information into the tradesDF dataframe by iterating over the orders dataframe by index and row
    for index, row in ordersDF.iterrows():
        # import pdb
        # pdb.set_trace()
        # Total value of shares purchased or sold
        trade_value = pricesDF.loc[index, row["Symbol"]] * row["Shares"]
        # Transaction cost 
        transaction_fee = commission + impact * trade_value # we pay commission + 0.5% of the trade value as market impact (BUY and SELL)

        # Update the number of shares and cash position for each date based on the transactions completed
        if row["Order"] == "BUY":
            tradesDF.loc[index, row["Symbol"]] = tradesDF.loc[index, row["Symbol"]] + row["Shares"] #adds the number of shares bought (from the ordersDF dataframe) into the tradesDF dataframe.
            tradesDF.loc[index, "cash"] = tradesDF.loc[index, "cash"] + trade_value * (-1.0) - transaction_fee #adds the transaction_fee into the tradesDF dataframe
        else:  #else, it's a SELL order
            tradesDF.loc[index, row["Symbol"]] = tradesDF.loc[index, row["Symbol"]] - row["Shares"]  #adds the number of shares sold (from the ordersDF dataframe) into the tradesDF dataframe.
            tradesDF.loc[index, "cash"] = tradesDF.loc[index, "cash"] + trade_value - transaction_fee  #adds the transaction_fee into the tradesDF dataframe plus the positive cash impact of the sale

    # Establish for each day how much of each asset is in the portfolio
    holdingsDF = pd.DataFrame(np.zeros((pricesDF.shape)), pricesDF.index, pricesDF.columns)
    for row in range(len(holdingsDF)):
        if row == 0:  # this is the first day.
            holdingsDF.iloc[0, :-1] = tradesDF.iloc[0, :-1].copy() # equities are copied from the first day of trades
            holdingsDF.iloc[0, -1] = tradesDF.iloc[0, -1] + start_val # cash is initialized with the initial cash position
        else:  # all subsequent days, show cumulatives
            holdingsDF.iloc[row] = holdingsDF.iloc[row - 1] + tradesDF.iloc[row]
        row = row + 1

    # Create a dataframe that represents the daily value of each portfolio asset
    dailyValueDF = pricesDF * holdingsDF

    # Create portvals dataframe
    portValDF = pd.DataFrame(dailyValueDF.sum(axis=1), dailyValueDF.index, ["portValDF"])
    #print "type of portValDF: ", type(portValDF)
    return portValDF


def order_info(orders_file ="./orders/orders-02.csv"):
    """Summary: Derive start and end dates as well as syms from the orders dataframe and symbols
        Inputs: orders_file: the name of a file from which to read orders (string or file object)
        Returns:
            orders_df: a dataframe with all orders
            start_date: the first date of trading in the orders file, a datetime object
            end_date: the last day of trading in the orders file, a datetime object
            sysms_traded: the symbols traded in the orders file, a list of equity symbols
    """
    # Read in the orders_file and sort it by date
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df.sort_index(ascending=True, inplace=True)

    #get key date info and the symbols traded on those dates
    start_date = orders_df.index.min() #earliest date is the smallest
    end_date = orders_df.index.max()  # max date is the largest
    syms_traded= orders_df.Symbol.unique().tolist() #equities involved in the orders
    return start_date, end_date, orders_df, syms_traded


def compute_portfolio_stats(port_val, sf, rfr): # Get portfolio statistics (note: std_daily_ret = volatility)
    """
    Inputs:
        port_val: a pandas series of total portfolio value, indexed by date
        sf: sampling frequency (number of days the stock traded), a float
        rfr: risk free rate of return, a float
    Returns:
        cr: cumulative return, a numpy 64 bit float
        adr: average daily return (if sf == 252 this is daily return), a numpy 64 bit float
        sddr: std of daily returns, a numpy 64 bit float
        sr: sharpe ratio, risk-adjusted returns, a numpy 64 bit float
    """
    daily_rets = port_val.copy()  # copy the data frame
    daily_rets[1:] = (port_val[1:] / port_val[:-1].values) - 1  # daily return = (value today / value yesterday) -1
    daily_rets[0] = 0  # set the initial return for row 0 to 0
    cr = (port_val[-1] / port_val[0]) - 1  # cumulative return
    adr = ((port_val[1:].values / port_val[:-1].values) - 1.0).mean()  # average daily return
    sddr = ((port_val[1:].values / port_val[:-1].values) - 1.0).std(ddof=1)  # std of daily returns
    sr = np.sqrt(sf) * ((adr - rfr) / sddr)  # sharpe ratio, risk-adjusted returns
    return cr, adr, sddr, sr


def fill_missing_values(df_data): # forward and backfills data from a dataframe
    df_data.fillna(method="ffill", inplace=True) # forward fill as best practice
    df_data.fillna(method="bfill", inplace=True) # backfill 2nd resort

##THIS FUNCTION ONLY NEEDED FOR TEST CASE 1##
# def get_daily_portfolio_val(prices, allocs, sv): #computes daily portfolio value
#     """
#     Inputs:
#         prices: a pandas dataframe of prices for various stock tickers, indexed by date
#         allocs: a list of the relative allocations (floats) of each equity in the portfolio
#         sv: a float, portfolio starting value
#     Returns:
#         port_val: a pandas dataframe/series of total portfolio value, indexed by date
#     """
#     prices_copy = prices.copy()
#     normed = prices_copy / prices.ix[0, :]  # normalized by first row
#     alloced = normed * allocs  # multiply by the allocations for each equity
#     pos_vals = alloced * sv  # positions values
#     port_val = pos_vals.sum(axis=1)  # portfolio value
#     return port_val


def test_code(): # helper function to test code; not called in autograding
    # ##TEST 1 -- USED TO VALIDATE compute_portfolio_stats WITH ANALYSIS.PY##
    # start_date = dt.datetime(2010,1,1)
    # end_date = dt.datetime(2010,12,31)
    # symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    # allocations = [0.2, 0.3, 0.4, 0.1]
    # start_val = 1000000
    #
    # # Read in adjusted closing prices for given symbols, date range
    # dates = pd.date_range(start_date, end_date)
    # prices_all = get_data(symbols, dates)  # automatically adds SPY
    # fill_missing_values(prices_all)
    # prices = prices_all[symbols]  # only portfolio symbols
    # port_val = get_daily_portfolio_val(prices, allocations, start_val)
    #
    # # Assess the portfolio
    # cr, adr, sddr, sr = compute_portfolio_stats(port_val, sf=252.0, rfr = 0.0)
    #
    # # Print statistics
    # print "Sharpe Ratio:", sr
    # print "Volatility (stdev of daily returns):", sddr
    # print "Average Daily Return:", adr
    # print "Cumulative Return:", cr

    ##TEST 2##
    of = "./additional_orders/orders.csv"
    sv = 1000000

    # Process orders
    port_val = compute_portvals(orders_file = of, start_val = sv)
    if isinstance(port_val, pd.DataFrame):
        port_val = port_val[port_val.columns[0]] # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get basic info from the file
    start_date, end_date, orders_df, syms_traded = order_info(of)

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(port_val, sf=252.0, rfr = 0.0)

    #get $SPX stats
    SPX_prices = get_data(["$SPX"], pd.date_range(start_date, end_date)).dropna()
    fill_missing_values(SPX_prices)
    SPX_prices = SPX_prices['$SPX']  # only $SPX, drop SPY
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = compute_portfolio_stats(SPX_prices, sf=252.0, rfr = 0.0)

    # Compare portfolio against $SPX in the same timeframe
    print "Test 2 results for orders.csv: "
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX : {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX : {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX : {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX : {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(port_val[-1])

    ##TEST 3##
    of = "./additional_orders/orders2.csv"
    sv = 1000000

    # Process orders
    port_val = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(port_val, pd.DataFrame):
        port_val = port_val[port_val.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get basic info from the file
    start_date, end_date, orders_df, syms_traded = order_info(of)

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(port_val, sf=252.0, rfr=0.0)

    # get SPX stats
    SPX_prices = get_data(["$SPX"], pd.date_range(start_date, end_date)).dropna()
    fill_missing_values(SPX_prices)
    SPX_prices = SPX_prices['$SPX']  # only $SPX, drop SPY
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = compute_portfolio_stats(SPX_prices,
                                                                                                  sf=252.0, rfr=0.0)

    # Compare portfolio against $SPX in the same timeframe
    print  "--------------------------------------"
    print "Test 3 results for orders2: "
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX : {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX : {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX : {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX : {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(port_val[-1])

    ##TEST 4##
    of = "./additional_orders/orders-short.csv"
    sv = 1000000

    # Process orders
    port_val = compute_portvals(orders_file=of, start_val=sv)
    if isinstance(port_val, pd.DataFrame):
        port_val = port_val[port_val.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get basic info from the file
    start_date, end_date, orders_df, syms_traded = order_info(of)

    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = compute_portfolio_stats(port_val, sf=252.0, rfr=0.0)

    # get SPX stats
    SPX_prices = get_data(["$SPX"], pd.date_range(start_date, end_date)).dropna()
    fill_missing_values(SPX_prices)
    SPX_prices = SPX_prices['$SPX']  # only $SPX, drop SPY
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = compute_portfolio_stats(SPX_prices,
                                                                                                  sf=252.0, rfr=0.0)

    # Compare portfolio against $SPX in the same timeframe
    print  "--------------------------------------"
    print "Test 4 results for orders-short: "
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX : {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX : {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX : {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX : {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(port_val[-1])


def author():
    return 'gth659q'  # my Georgia Tech username.

if __name__ == "__main__":
    test_code()
