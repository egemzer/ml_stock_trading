"""
Implementing technical indicators

Code written by: Erika Gemzer, gth659q, Summer 2018

Passed all test cases from grade_strategy_learner on Buffet01 7/30/2018 in 71.80 seconds

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from util import get_data, plot_data
import copy
import datetime as dt

def calc_momentum (prices, window=4):
    """Calculate the momentum indicator using this formula:
    momentum[t] = (price[t]/price[t-window]) - 1.

    Inputs / Parameters:
        prices: adjusted close price, a pandas series for a given symbol
        window: number of days to look back.  Default set to 4.

    Returns: momentum, time series data frame of the same size as the input data
    """
    momentum = pd.Series(np.nan, index=prices.index)
    momentum.iloc[window:] = prices.iloc[window:] / prices.values[:-window] - 1
    #momentum.fillna(method="bfill", inplace=True)
    return momentum


def calc_sma(prices, window = 4):
    """Calculate simple moving average indicator using the formula:
    sma = (price / rolling_mean) - 1

    Inputs / Parameters:
        prices: adjusted close price, a pandas series for a given symbol
        window: number of days to look back.  Default set to 4.

    Returns: sma, The simple moving average indicator
    """

    rolling_mean = prices.rolling(window=window).mean()
    sma = (prices / rolling_mean) - 1
    #sma.fillna(method="bfill", inplace=True)
    return sma


def calc_bollinger (prices, window = 4):
    """Calculates Bollinger value, indicating how many std a price is from the mean

    Inputs / Parameters:
        prices: adjusted close price, a pandas series for a given symbol
        window: number of days to look back.  Default set to 4.

    Returns: bollinger_value
    """

    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    bollinger_value = (prices - rolling_mean) / rolling_std
    #bollinger_value.fillna(method="bfill", inplace=True)
    return bollinger_value


def calc_volatility (prices, window = 4):
    """Calculates volatility of a stock.

    Inputs / Parameters:
        prices: adjusted close price, a pandas series for a given symbol
        window: number of days to look back.  Default set to 4.

    Returns: volatility, a statistical measure of the dispersion of returns for a given security or market index.
    """

    rolling_std = prices.rolling(window=window).std()
    volatility = rolling_std
    #volatility.fillna(method="bfill", inplace=True)
    return volatility

def demoIndicators(pricesDF):
    """Compute technical indicators and plots to demonstrate their use alongside normalized pricing data for an equity.

    Inputs / parameters: prices: adjusted close price, a pandas series for a given symbol

    Returns:
        plots of each indicator
    """

    # Calculate momentum (-0.5 to +0.5), the price slope within a time window
    momentum = calc_momentum(pricesDF)
    momentumNormed = (momentum - np.mean(momentum)) / np.std(momentum)

    # Calculate Simple Moving Average (SMA) indicator
    sma = calc_sma(pricesDF)
    smaNormed = (sma - np.mean(sma)) / np.std(sma)

    # Calculate Bollinger value (-1 to +1)
    bollinger_value = calc_bollinger(pricesDF)
    bvNormed = (bollinger_value - np.mean(bollinger_value)) / np.std(bollinger_value)

    # Calculate volatility
    volatility = calc_volatility(pricesDF)
    volatilityNormed = (volatility - np.mean(volatility)) / np.std(volatility)

    # Create a dataframe with the technical indicators
    indicatorsDF = pd.concat([momentum, sma, bollinger_value, volatility], axis=1)
    indicatorsDF.columns = ["momentum", "sma", "bollinger value", "volatility"]

    # normalize the prices for plotting
    pricesNormed = pricesDF / pricesDF[0]
    dates = pricesNormed.index.values

    # plots, plots, plots (normalized indicator vs normalized stock price)
    plt.plot(dates, pricesNormed, label="Normalized JPM Prices")

    # # Momentum
    # plt.plot(dates, momentumNormed, label="Normalized Momentum")
    # plt.title("Normalized Momentum vs JPM (equity) Stock Price")

    # # SMA
    # plt.plot(dates, smaNormed, label="Normalized SMA")
    # plt.title("Normalized SMA vs JPM (equity) Stock Price")

    # # Bollinger Value
    # plt.plot(dates, bvNormed, label="Normalized Bollinger Value (R)")
    # plt.title("Normalized Bollinger Value (R) vs JPM (equity) Stock Price")


    #Volatility
    plt.plot(dates, volatilityNormed, label="Normalized Volatility")
    plt.title("Normalized Volatility vs JPM (equity) Stock Price")

    # more general plot stuff
    plt.xlabel("Date")
    plt.ylabel("Normalized Value")
    plt.legend(loc="lower right")
    plt.savefig('result.png')
    plt.switch_backend('Agg')


def author(self):
    return 'gth659q'  # my Georgia Tech username.

if __name__=="__main__":
    symbol = "JPM"
    start_date = dt.datetime(2008, 1, 1, 0, 0)
    end_date = dt.datetime(2008, 12, 31, 0, 0)


    # Create a dataframe with adjusted close prices for the symbol, drop nas, includes SPY automatically
    pricesDF = get_data([symbol], pd.date_range(start_date, end_date)).dropna()
    pricesDF = pricesDF[symbol]  # only portfolio symbol, drop SPY

    # fill any "nan" values remove SPY
    pricesDF.fillna(method="ffill", inplace=True)  # forward fill as best practice
    pricesDF.fillna(method="bfill", inplace=True)  # backfill 2nd resort

    demoIndicators(pricesDF)
