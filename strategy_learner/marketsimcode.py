"""MC2-P1: Market simulator.

Copyright 2017, Georgia Tech Research Corporation
Atlanta, Georgia 30332-0415
All Rights Reserved

Code implemented by: Erika Gemzer | gth659q | Summer 2018

Passed all test cases from grade_strategy_learner on Buffet01 7/30/2018 in 71.80 seconds
"""

import pandas as pd
import numpy as np
import datetime as dt
import os
from util import get_data, plot_data

def compute_portvals_single_stock(ordersDF, symbol, start_val=100000, cash_pos=0.0, long_pos=1.0, short_pos=-1.0, commission=0.00, impact=0.0, num_shares = 1000):
    """Summary: a market simulator that accepts trading orders and keeps track of a single-stock portfolio's value over time
    and then assesses the performance of that portfolio.

    Inputs:
        ordersDF: the name of a file from which to read orders (string or file object)
        symbol: The single stock symbol whose portfolio values will be computed
        start_val: the starting value of the portfolio (initial cash available)
        commission: fixed amount in dollars charged for each transaction (both entry and exit)
        impact:  amount the price moves against the trader compared to the historical data at each transaction
        num_shares: The number of shares that can be traded in one order, defaults to 1000

    Returns: portvals, a dataframe with one column containing the value of the portfolio for each trading day (the index), from start_date to end_date, inclusive.
    """

    # Sort the ordersDF by date
    ordersDF.sort_index(ascending=True, inplace=True)

    # get key date info and the symbols traded on those dates
    start_date = ordersDF.index.min()  # earliest date is the smallest
    end_date = ordersDF.index.max()  # max date is the largest

    # Create a dataframe with adjusted close prices for the symbol, drop nas, includes SPY automatically
    pricesDF = get_data([symbol], pd.date_range(start_date, end_date)).dropna()
    pricesDF = pricesDF[symbol]  # only portfolio symbol, drop SPY

    # fill any "nan" values remove SPY
    pricesDF.fillna(method="ffill", inplace=True)  # forward fill as best practice
    pricesDF.fillna(method="bfill", inplace=True)  # backfill 2nd resort
    pricesDF = pricesDF.to_frame()  # series to frame

    # Initialize a dataframe to hold all of the trade information from the orders
    tradesDF = pd.DataFrame(np.zeros((pricesDF.shape)), pricesDF.index, pricesDF.columns)  # tradesDF has the same structure as pricesDF, so we copy it but fill it with zeros
    tradesDF.sort_index(ascending=True, inplace=True)

    # add the "price" of cash for portfolio adily value calculation later
    pricesDF["cash"] = 1.0  # add cash position column, set to 1.0 for all rows (for now)


    # Establish for each day how much of each asset is in the portfolio
    holdingsDF = pd.DataFrame(np.zeros((pricesDF.shape)), pricesDF.index, pricesDF.columns)
    # Total value of shares purchased or sold

    for row in range(0, tradesDF.shape[0]):
        traded_share_value = pricesDF.iloc[row][symbol] * num_shares
        transactionFee = commission + impact * abs(traded_share_value)  # we pay commission + part of the trade value as market impact (BUY and SELL)
        oldPos = float(ordersDF.iloc[row-1])
        position = float(ordersDF.iloc[row])

        #print "day ", row, "position: ", position, "traded_share_value: ", traded_share_value

        # first day where a trade is made
        if row == 0 or holdingsDF.iloc[row - 1]["cash"] == start_val:
            traded_share_value = pricesDF.iloc[row][symbol] * num_shares
            transactionFee = commission + impact * abs(traded_share_value)  # we pay commission + part of the trade value as market impact (BUY and SELL)
            if position == long_pos: #going long on day 1, risky!
                tradesDF.iloc[row][symbol] = num_shares
                holdingsDF.iloc[row][symbol] = tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = start_val - traded_share_value - transactionFee  # adds the transactionFee into the tradesDF dataframe
            elif position == short_pos: #shorting on day 1, bold!
                tradesDF.iloc[row][symbol] = -num_shares
                holdingsDF.iloc[row][symbol] = tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = start_val + traded_share_value - transactionFee  # adds the transactionFee into the tradesDF dataframe
            else: #cash position
                tradesDF.iloc[row][symbol] = cash_pos
                holdingsDF.iloc[row][symbol] = tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = start_val

        # last day (whether or not a trade is needed)
        elif row == len(ordersDF) - 1:
            sharestoLiq = -(tradesDF.iloc[row - 1][symbol])
            tradesDF.iloc[row][symbol] = sharestoLiq
            traded_share_value = pricesDF.iloc[row][symbol] * sharestoLiq
            transactionFee = commission + impact * abs(traded_share_value)

            #this better be zero!
            holdingsDF.iloc[row][symbol] = tradesDF.iloc[row - 1][symbol] + sharestoLiq
            #print "zero check!: ", holdingsDF.iloc[row][symbol]
            if sharestoLiq < 0: # if we have to sell shares to get to 0 holdings / positions
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] + traded_share_value - transactionFee
            elif sharestoLiq > 0: # if we have to buy shares to get to 0 holdings / positions
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] - traded_share_value - transactionFee
            elif sharestoLiq == 0: #if we aren't holding anything, we keep the cash we have in hand
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"]

        # all other days, look holistically
        else:
            # moving from a cash to a long position, buy 1000 shares
            if oldPos == cash_pos and position == long_pos and holdingsDF.iloc[row - 1][symbol] == 0:
                tradesDF.iloc[row][symbol] = num_shares
                traded_share_value = pricesDF.iloc[row][symbol] * num_shares
                transactionFee = commission + impact * abs(traded_share_value)
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol] + tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] - traded_share_value - transactionFee  # adds the transactionFee into the tradesDF dataframe
                # print "oldPos: ", oldPos, "newPos: ", position, "prev shares: ", holdingsDF.iloc[row - 1][symbol], "date: ", row, "shares updated: ", holdingsDF.iloc[row][symbol], "cash updated: ", holdingsDF.iloc[row]["cash"]

            # moving from a short to a long position, buy 2000 shares
            elif oldPos == short_pos and position == long_pos and holdingsDF.iloc[row - 1][symbol] < 0:
                tradesDF.iloc[row][symbol] = 2 * num_shares
                traded_share_value = pricesDF.iloc[row][symbol] * 2 * num_shares
                transactionFee = commission + impact * abs(traded_share_value)
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol] + tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] - traded_share_value - transactionFee
                # print "oldPos: ", oldPos, "newPos: ", position, "prev shares: ", holdingsDF.iloc[row - 1][symbol], "date: ", row, "shares updated: ", holdingsDF.iloc[row][symbol], "cash updated: ", holdingsDF.iloc[row]["cash"]

            # moving from a short to a cash position, buy 1000 shares
            elif oldPos == short_pos and position == cash_pos and holdingsDF.iloc[row - 1][symbol] < 0:
                tradesDF.iloc[row][symbol] = num_shares
                traded_share_value = pricesDF.iloc[row][symbol] * num_shares
                transactionFee = commission + impact * abs(traded_share_value)
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol] + tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] - traded_share_value - transactionFee
                # print "oldPos: ", oldPos, "newPos: ", position, "prev shares: ", holdingsDF.iloc[row - 1][symbol], "date: ", row, "shares updated: ", holdingsDF.iloc[row][symbol], "cash updated: ", \
                holdingsDF.iloc[row]["cash"]

            # moving from a cash to a short position, sell 1000 shares
            elif oldPos == cash_pos and position == short_pos and holdingsDF.iloc[row - 1][symbol] == 0:
                tradesDF.iloc[row][symbol] = -num_shares
                traded_share_value = pricesDF.iloc[row][symbol] * num_shares
                transactionFee = commission + impact * abs(traded_share_value)
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol] + tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] + traded_share_value - transactionFee
                # print "oldPos: ", oldPos, "newPos: ", position, "prev shares: ", holdingsDF.iloc[row - 1][symbol], "date: ", row, "shares updated: ", holdingsDF.iloc[row][symbol], "cash updated: ", holdingsDF.iloc[row]["cash"]

            # moving from a long position to a short position, sell 2000 shares
            elif oldPos == long_pos and position == short_pos and holdingsDF.iloc[row - 1][symbol] > 0:
                tradesDF.iloc[row][symbol] = -2 * num_shares
                traded_share_value = pricesDF.iloc[row][symbol] * 2 * num_shares
                transactionFee = commission + impact * abs(traded_share_value)
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol] + tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] + traded_share_value - transactionFee
                # print "oldPos: ", oldPos, "newPos: ", position, "prev shares: ", holdingsDF.iloc[row - 1][symbol], "date: ", row, "shares updated: ", holdingsDF.iloc[row][symbol], "cash updated: ", holdingsDF.iloc[row]["cash"]

            # moving from a long position to a cash position, sell 1000 shares
            elif oldPos == long_pos and position == cash_pos and holdingsDF.iloc[row - 1][symbol] > 0:
                tradesDF.iloc[row][symbol] = - num_shares
                traded_share_value = pricesDF.iloc[row][symbol] * num_shares
                transactionFee = commission + impact * abs(traded_share_value)
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol] + tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] + traded_share_value - transactionFee
                # print "oldPos: ", oldPos, "newPos: ", position, "prev shares: ", holdingsDF.iloc[row - 1][symbol], "date: ", row, "shares updated: ", holdingsDF.iloc[row][symbol], "cash updated: ", \
                holdingsDF.iloc[row]["cash"]


            # if we are maintaining the current position or already holding too many stock, make no trades
            else:
                tradesDF.iloc[row][symbol] = 0.0
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row-1]["cash"]
                # print "oldPos: ", oldPos, "newPos: ", position, "prev shares: ", holdingsDF.iloc[row - 1][symbol], "date: ", row, "shares updated: ", holdingsDF.iloc[row][symbol], "cash updated: ", holdingsDF.iloc[row]["cash"]

    # Create a dataframe that represents the daily value of the portfolio asset
    dailyValueDF = pricesDF * holdingsDF

    # Create portvals dataframe
    portValDF = pd.DataFrame(dailyValueDF.sum(axis=1), holdingsDF.index, ["portValDF"])
    return portValDF, tradesDF, holdingsDF, pricesDF

def order_info(orders_file):
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


def compute_portfolio_stats(port_val, sf = 252.0, rfr = 0.0): # Get portfolio statistics (note: std_daily_ret = volatility)
    """
    Inputs:
        port_val: a pandas series of total portfolio value, indexed by date
        sf: sampling frequency (number of days the stock traded), a float. Typically 252.0
        rfr: risk free rate of return, a float. Typically 0.0
    Returns:
        cr: cumulative return, a numpy 64 bit float
        adr: average daily return (if sf == 252 this is daily return), a numpy 64 bit float
        sddr: std of daily returns, a numpy 64 bit float
        sr: sharpe ratio, risk-adjusted returns, a numpy 64 bit float
    """
    end = len(port_val)
    daily_rets = port_val.copy()  # copy the data frame
    daily_rets[1:] = (port_val[1:] / port_val[:-1].values) - 1  # daily return = (value today / value yesterday) -1
    daily_rets[0] = 0  # set the initial return for row 0 to 0
    cr = (port_val.iloc[-1][0] / port_val.iloc[0][0]) - 1  # cumulative return
    adr = ((port_val[1:].values / port_val[:-1].values) - 1.0).mean()  # average daily return
    sddr = ((port_val[1:].values / port_val[:-1].values) - 1.0).std(ddof=1)  # std of daily returns
    sr = np.sqrt(sf) * ((adr - rfr) / sddr)  # sharpe ratio, risk-adjusted returns
    return cr, adr, sddr, sr


def fill_missing_values(df_data): # forward and backfills data from a pandas data series
    df_data.fillna(method="ffill", inplace=True) # forward fill as best practice
    df_data.fillna(method="bfill", inplace=True) # backfill 2nd resort


def compute_portVals_from_trades (tradesDF, pricesDF, symbol, start_val=100000, commission=0.00, impact=0.0):
    """does the same thing as compute_portvals_single_stock but this function had to be created to satisfy the autograder and structure of the professor's code. To remove this function later, there are commented out stuff in the TestPolicy function to uncomment, and change the return to return orders instead of trades."""

    # Establish for each day how much of each asset is in the portfolio
    holdingsDF = pd.DataFrame(np.zeros((pricesDF.shape)), pricesDF.index, pricesDF.columns)

    for row in range(0, tradesDF.shape[0]):

        num_shares = tradesDF.iloc[row][symbol] # may be positive (BUY) or negative (SELL) or 0.0 (no trades that day)
        traded_share_value = pricesDF.iloc[row][symbol] * -1 * num_shares # may be pos or negative depending on the trade (BUY vs SELL)
        transactionFee = commission + impact * abs(traded_share_value)  # we pay commission + part of the trade value as market impact (BUY and SELL)

        # first day that a trade is made
        if row == 0 or holdingsDF.iloc[row - 1]["cash"] == start_val:
            holdingsDF.iloc[row][symbol] = tradesDF.iloc[row][symbol]
            holdingsDF.iloc[row]["cash"] = start_val + traded_share_value - transactionFee  #cash impact from trading

        # all other days
        else:
            # if we're not buying and selling that day, holdings are the same as the day before
            if tradesDF.iloc[row][symbol] == 0.0:
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"]

            # if we are buying or selling, both the shares (symbol column) and cash will change
            else:
                holdingsDF.iloc[row][symbol] = holdingsDF.iloc[row - 1][symbol] + tradesDF.iloc[row][symbol]
                holdingsDF.iloc[row]["cash"] = holdingsDF.iloc[row - 1]["cash"] + traded_share_value - transactionFee  # cash impact from trading

    # Create a dataframe that represents the daily value of the portfolio asset
    dailyValueDF = pricesDF * holdingsDF

    # Create portvals dataframe
    portValDF = pd.DataFrame(dailyValueDF.sum(axis=1), holdingsDF.index, ["portValDF"])
    return portValDF




def author():
    return 'gth659q'  # my Georgia Tech username.

if __name__ == "__main__":
    print "this code written by Erika Gemzer, gth659q"