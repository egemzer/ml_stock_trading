"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch

Code implemented by: Erika Gemzer, gth659q, Summer 2018

Passed all test cases from grade_strategy_learner on Buffet01 7/30/2018 in 71.80 seconds

"""

import datetime as dt
import pandas as pd
import util as ut
import numpy as np
import QLearner as ql
import indicators
import random as rand
from util import get_data
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import marketsimcode

class StrategyLearner(object):

    # constructor
    def __init__(self, impact=0.0, num_shares = 1000, epochs = 100, num_steps=10, commission = 0.00, verbose = False, **kwargs):
        """Create a strategy learner (Q-learning based) that can learn a trading policy

        Inputs / Parameters:
            impact: The amount the price moves against the trader compared to the historical data at each transaction
            num_shares: The number of shares that can be traded in one order
            epochs: The number of times to train the QLearner
            num_steps: The number of steps used in getting thresholds for the discretization process. It is the number of groups to put data into.
            commission: The fixed amount in dollars charged for each transaction
            verbose: If False, no plots. If True, print and plot data in add_evidence
            **kwargs: These are the arguments for QLearner
        """
        # Set constants for positions (which become our order signals)
        self.SHORT = -1.0
        self.CASH = 0.0
        self.LONG = 1.0

        self.epochs = epochs
        self.num_steps = num_steps
        self.num_shares = num_shares
        self.impact = impact
        self.commission = commission
        self.verbose = verbose

        # Initialize a QLearner for this Strategy Learner
        self.QLearner = ql.QLearner(**kwargs)


    def addEvidence(self, symbol="IBM", sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), sv=10000):

        """Creates a QLearner, and trains it for trading.

        Inputs / Parameters:
            symbol: The stock symbol to act on
            sd: A datetime object that represents the start date
            ed: A datetime object that represents the end date
            sv: Start value of the portfolio which contains only the one symbol
        """

        # Get adjusted close prices for the given symbol on the given date range
        dates = pd.date_range(sd, ed)
        prices_all = get_data([symbol], dates) #includes SPY due to util function
        pricesDF = prices_all[[symbol]] # only the symbol
        # Get features and thresholds
        indicatorsDF = self.getIndicators(pricesDF[symbol])
        thresholds = self.setThresholds(indicatorsDF, self.num_steps)
        cum_returns = []

        for epoch in range(1, self.epochs + 1):
            # Initial position is holding nothing
            position = self.CASH
            # Create a series that captures order signals based on actions taken
            orders = pd.Series(index=indicatorsDF.index)
            # Iterate over the data by date
            for day, date in enumerate(indicatorsDF.index):
                # Get a state
                state = self.getState(indicatorsDF.loc[date], thresholds)
                # On the first day, get an action without updating the Q-table
                if date == indicatorsDF.index[0]:
                    action = self.QLearner.querysetstate(state)
                    newPos = float(action - 1)
                # On other days, calculate the reward and update the Q-table
                else:
                    prev_price = pricesDF[symbol].iloc[day - 1]
                    curr_price = pricesDF[symbol].loc[date]
                    reward = self.calcDailyReward(prev_price,curr_price, position)
                    action = self.QLearner.query(state, reward)
                    newPos = float(action - 1)

                # Add new_pos to orders, update current position
                orders.loc[date] = newPos
                position = newPos

            #get the portfolio values (which also creates the tradesDF and pricesDF, in the background
            portvals, tradesDF, holdingsDF, pricesDF = marketsimcode.compute_portvals_single_stock(ordersDF=orders, symbol=symbol, start_val=sv, commission=self.commission, impact=self.impact, num_shares = self.num_shares)
            cum_return = marketsimcode.compute_portfolio_stats(portvals)[0]
            cum_returns.append(cum_return)

            # Check for convergence after running for at least 30 epochs
            if epoch > 20:
                # Stop if the cum_return doesn't improve for 10 epochs
                if self.checkConvergence(cum_returns):
                    break

        #print "orders series from learner", orders
        #print "tradesDF from learner: ", tradesDF

        if self.verbose:
            plt.plot(cum_returns)
            plt.xlabel("Epoch")
            plt.ylabel("Cumulative return (%)")
            # plt.show()
            plt.savefig('result.png')
            plt.switch_backend('Agg')

    def getIndicators(self, prices):
        """Compute technical indicators and use them as features to be fed into a Q-learner. By default, uses 4 days for rolling indicators.

        Inputs / parameters: prices: Dataframe of adjusted close prices of the given symbol

        Returns: indicatorsDF: A pandas dataframe of the technical indicators
        """

        # Calculate momentum (-0.5 to +0.5), the price slope within a time window
        momentum = indicators.calc_momentum(prices)

        # Calculate Simple Moving Average (SMA) indicator
        sma = indicators.calc_sma(prices)

        # Calculate Bollinger value (-1 to +1)
        bollinger_value = indicators.calc_bollinger(prices)

        # Calculate volatility
        volatility = indicators.calc_volatility(prices)

        # Create a dataframe with the technical indicators
        indicatorsDF = pd.concat([momentum, sma, bollinger_value, volatility], axis=1)
        indicatorsDF.columns = ["momentum", "sma", "bollinger value", "volatility"]
        #drop NaN values which will be the first N-days (where N is the window for the rolling indicators). This means no trading on those days
        indicatorsDF.dropna(inplace=True)

        return indicatorsDF


    def setThresholds(self, indicatorsDF, num_steps):
        """Compute the thresholds to be used in the discretization of features for the QLearner.

            Inputs / Parameters:
            indicatorsDF: a dataframe with the technical indicators, indexed by dates
            num_steps: the number of steps (from the constructor)

            Returns: thresholds, a 2D numpy array where the first dimension indicates the indices of features in indicatorsDF. The second dimension refers to the value of a feature at a particular threshold.
        """
        #define the step size for each threshold
        stepSize = int(round(indicatorsDF.shape[0] / num_steps))

        indicatorsDF_copy = indicatorsDF.copy()
        thresholds = np.zeros(shape=(indicatorsDF.shape[1], num_steps))

        #for each indicator, set the thresholds using the step size and sorted data
        for i, feat in enumerate(indicatorsDF.columns):
            indicatorsDF_copy.sort_values(by=[feat], inplace=True)
            for step in range(0, self.num_steps):
                if step < num_steps - 1:
                    thresholds[i, step] = indicatorsDF_copy[feat].iloc[((step + 1) * stepSize)]
                # The last threshold must be the largest value in indicatorsDF_copy
                else:
                    thresholds[i, step] = indicatorsDF_copy[feat].iloc[-1]

        return thresholds


    def getState(self, indicators, thresholds):
        """Discretize features and return a state.

        Inputs / Parameters:
            indicators: The technical indicators to be discretized. These are the indicators computed in getIndicators(), a pandas data series
            thresholds: a series of the thresholds for each step for each technical indicator

        Returns: state, a state in the Q-table from which we will query for an action. It indicates an index of the first dimension in the Q-table
        """
        state = 0
        i = 0
        while i < len(indicators):
            #print "indicator: ", i, "np.amin(thresholds[i]", np.argmin(thresholds[i]), "indicator: ", indicators[i]
            if indicators[i] < np.amin(thresholds[i]): #if the indicator value falls below the lowest step
                state = state + pow((self.num_steps/2), i)
            else:
                thres_i = np.where(thresholds[i] <= indicators[i])
                thres_max = np.argmax(thres_i)
                state = state + int(thres_max * pow((self.num_steps/2), i))
            i += 1
        return state


    def calcDailyReward(self, prev_price, curr_price, position):
        """Calculate the daily reward as a percentage change in prices:

        - If position is LONG: if the price goes up (curr_price > prev_price), we get a positive reward; otherwise, we get a negative reward
        - If position is SHORT: if the price goes down, we get a positive reward; otherwise, we a negative reward
        - If position is CASH: we get no reward since we took no risk

        All rewards are adjusted for impact if the previous position indicated a buy or sell
        """

        if position != 0.0:
            reward = position * ((curr_price / prev_price) - 1) - self.impact
        else:
            reward = position * ((curr_price / prev_price) - 1)
        return reward


    def checkConvergence(self, cum_returns, plateau=10):
        """Check if the cumulative returns have converged.

            Inputs / Parameters:
            cum_returns: A list of cumulative returns for respective epochs
            plateau: The number of epochs with no improvement in cum_returns

            Returns: True if converged, False otherwise
        """
        # The number of epochs should be at least plateau before checking
        # for convergence
        if plateau > len(cum_returns):
            return False
        latest_returns = cum_returns[-plateau:]
        # If all the latest returns are the same, return True
        if len(set(latest_returns)) == 1:
            return True
        max_return = max(cum_returns)
        if max_return in latest_returns:
            # If one of recent returns improves, not yet converged
            if max_return not in cum_returns[:len(cum_returns) - plateau]:
                return False
            else:
                return True
        # If none of recent returns is greater than max_return, it has converged
        return True


    def generateOrders(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1,0,0), \
        ed=dt.datetime(2010,1,1,0,0), \
        sv = 100000):
        """this method generates orders to be used with testPolicy

            Inputs / Parameters:
            symbol: the stock symbol we're acting upon
            sd: datetime object, the start date for analysis
            ed: datetime object, the end date for analysis
            sv: starting value of the portfolio. Note the portfolio only contains the symbol

            Returns: orders, a pandas series of the positions for each day
        """

        # Get adjusted close prices for the given symbol on the given date range
        prices_all = get_data([symbol], pd.date_range(sd, ed)).dropna() # includes SPY due to util function
        pricesDF = prices_all[symbol] #remove SPY
        pricesDF = pricesDF.to_frame()  # series to frame

        # Get features and thresholds
        indicatorsDF = self.getIndicators(pricesDF[symbol])
        thresholds = self.setThresholds(indicatorsDF, self.num_steps)

        # Create a series to capture the order signals based on the actions taken by the trader
        orders = pd.Series(index=indicatorsDF.index)
        # Iterate over all the data, by date (index)

        for date in indicatorsDF.index:
            # Get a state, action, and new position
            state = self.getState(indicatorsDF.loc[date], thresholds)
            action = self.QLearner.querysetstate(state)
            newPos = float(action - 1)

            # Add new_pos to orders
            orders.loc[date] = newPos
            # Update current position
            position = newPos

        return orders


    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1,0,0), \
        ed=dt.datetime(2010,1,1,0,0), \
        sv = 100000):
        """ this method uses the existing policy and tests it against new data

        Inputs / Parameters:
            symbol: the stock symbol we're acting upon
            sd: datetime object, the start date for analysis
            ed: datetime object, the end date for analysis
            sv: starting value of the portfolio. Note the portfolio only contains the symbol

        Returns: trades, a dataframe orders for each day (+/- 2000 shares of the given symbol)

        """
        orders = self.generateOrders(symbol = symbol, sd = sd, ed = ed, sv = sv)

        # Create trades dataframe from the learner's suggested strategy
        portvals, trades, holdingsDF, pricesDF = marketsimcode.compute_portvals_single_stock(ordersDF=orders, symbol=symbol, start_val=sv, commission=self.commission, impact=self.impact, num_shares=self.num_shares)
        # if self.verbose: print type(trades)  # it better be a DataFrame!
        # if self.verbose: print trades
        # if self.verbose: print pricesDF
        return trades



    def author(self):
        return 'gth659q'  # my Georgia Tech username.



if __name__=="__main__":

    print"############ Erika's 1st Test Cases (Benchmark) ###############"
    verbose = False
    start_val = 100000
    benchmarkSymbol = "JPM"
    commission = 0.00
    impact = 0.0
    num_shares = 1000


    print "In-sample training period"
    start_date = dt.datetime(2008, 1, 1,0,0)
    end_date = dt.datetime(2009, 12, 31,0,0)

    # Create benchmark data series. Benchmark is a portfolio starting with
    # $100,000, investing in 1000 shares of symbol and holding that position
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([benchmarkSymbol], dates).dropna()
    indexDates = prices_all.index
    zeroes = [0.0] * len(prices_all)
    benchmarkTrades = pd.DataFrame({"Date": indexDates, benchmarkSymbol: zeroes})
    benchmarkTrades = benchmarkTrades.set_index('Date')
    benchmarkTrades.iloc[0][0] = 1000 #set to buy LONG on day1
    benchmarkTrades.iloc[(len(prices_all)-1)][0] = -1000  # set to sell all on the last day

    benchmarkOrders = pd.Series(index=indexDates, data=zeroes)
    benchmarkOrders.iloc[0] = 1.0  # set to buy LONG on day1
    benchmarkOrders.iloc[(len(prices_all)-1)] = -1.0  # set to sell all on the last day
    # Train and test a StrategyLearner

    stl = StrategyLearner(num_shares=num_shares, impact=impact,
                          commission=commission, verbose=True,
                          num_states=3000, num_actions=3)

    stl.addEvidence(symbol=benchmarkSymbol, sd=start_date, ed=end_date, sv=start_val)

    learnerOrders = stl.generateOrders(symbol=benchmarkSymbol, sd=start_date, ed=end_date)
    learnerTrades = stl.testPolicy(symbol=benchmarkSymbol, sd=start_date, ed=end_date)

    print "Erika's 1st test In-sample Testing Results"
    # Retrieve performance stats via a market simulator
    print ("Performances during training period for {}".format(benchmarkSymbol))
    print ("Date Range: {} pyto {}".format(start_date, end_date))

    benchmarkPortVal, benchmarkTradesDF, benchmarkHoldingsDF, benchmarkPricesDF = marketsimcode.compute_portvals_single_stock(benchmarkOrders, symbol=benchmarkSymbol,start_val=start_val, commission=commission, impact=impact, num_shares = num_shares)

    benchmarkPortVal2 = marketsimcode.compute_portVals_from_trades(tradesDF = benchmarkTrades, pricesDF = benchmarkPricesDF, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact)
    #print "benchmarkPortValDF with new function: ", benchmarkPortVal2
    print "benchmarkPortVal: ", benchmarkPortVal2.iloc[-1][0]

    learnerPortVal, learnerTradesDF, learnerHoldingsDF, learnerPricesDF = marketsimcode.compute_portvals_single_stock(learnerOrders, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact, num_shares = num_shares)

    learnerPortVal2 = marketsimcode.compute_portVals_from_trades(tradesDF = learnerTrades, pricesDF = learnerPricesDF, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact)
    #print "LearnerPortValDF with new function: ", learnerPortVal2
    print "learnerPortVal: ", learnerPortVal2.iloc[-1][0]

    print "Out-of-sample or testing period"
    # Perform similar steps as above except no training of the data
    start_date = dt.datetime(2010, 1, 1,0,0)
    end_date = dt.datetime(2011, 12, 31,0,0)

    #new benchmark, same criteria
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([benchmarkSymbol], dates).dropna()
    indexDates = prices_all.index
    zeroes = [0.0] * len(prices_all)
    benchmarkTrades = pd.DataFrame({"Date": indexDates, benchmarkSymbol: zeroes})
    benchmarkTrades = benchmarkTrades.set_index('Date')
    benchmarkTrades.iloc[0][0] = 1000 #set to buy LONG on day1
    benchmarkTrades.iloc[(len(prices_all)-1)][0] = -1000  # set to sell all on the last day

    benchmarkOrders = pd.Series(index=indexDates, data=zeroes)
    benchmarkOrders.iloc[0] = 1.0  # set to buy LONG on day1
    benchmarkOrders.iloc[(len(prices_all)-1)] = -1.0  # set to sell all on the last day

    learnerOrders = stl.generateOrders(symbol=benchmarkSymbol, sd=start_date, ed=end_date)
    learnerTrades = stl.testPolicy(symbol=benchmarkSymbol, sd=start_date, ed=end_date)

    print "Erika's 1st test Out-of-sample Testing Results"
    # Retrieve performance stats via a market simulator
    print ("\nPerformances during testing period for {}".format(benchmarkSymbol))
    print ("Date Range: {} to {}".format(start_date, end_date))

    # Retrieve performance stats via a market simulator
    print ("Performances during training period for {}".format(benchmarkSymbol))
    print ("Date Range: {} pyto {}".format(start_date, end_date))

    benchmarkPortVal, benchmarkTradesDF, benchmarkHoldingsDF, benchmarkPricesDF = marketsimcode.compute_portvals_single_stock(
        benchmarkOrders, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact,
        num_shares=num_shares)

    benchmarkPortVal2 = marketsimcode.compute_portVals_from_trades(tradesDF=benchmarkTrades, pricesDF=benchmarkPricesDF,
                                                                   symbol=benchmarkSymbol, start_val=start_val,
                                                                   commission=commission, impact=impact)
    #print "benchmarkPortValDF with new function: ", benchmarkPortVal2
    print "benchmarkPortVal: ", benchmarkPortVal2.iloc[-1][0]

    learnerPortVal, learnerTradesDF, learnerHoldingsDF, learnerPricesDF = marketsimcode.compute_portvals_single_stock( learnerOrders, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact,
        num_shares=num_shares)

    learnerPortVal2 = marketsimcode.compute_portVals_from_trades(tradesDF=learnerTrades, pricesDF=learnerPricesDF,
                                                                 symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact)
    #print "LearnerPortValDF with new function: ", learnerPortVal2
    print "learnerPortVal: ", learnerPortVal2.iloc[-1][0]

    print "############### Testing with Impact = 0.05 ################"
    verbose = False
    start_val = 100000
    benchmarkSymbol = "JPM"
    commission = 0.00
    num_shares = 1000
    impact = 0.005


    print "In-sample training period"
    start_date = dt.datetime(2008, 1, 1,0,0)
    end_date = dt.datetime(2009, 12, 31,0,0)

    # Create benchmark data series. Benchmark is a portfolio starting with
    # $100,000, investing in 1000 shares of symbol and holding that position
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([benchmarkSymbol], dates).dropna()
    indexDates = prices_all.index
    zeroes = [0.0] * len(prices_all)
    benchmarkTrades = pd.DataFrame({"Date": indexDates, benchmarkSymbol: zeroes})
    benchmarkTrades = benchmarkTrades.set_index('Date')
    benchmarkTrades.iloc[0][0] = 1000 #set to buy LONG on day1
    benchmarkTrades.iloc[(len(prices_all)-1)][0] = -1000  # set to sell all on the last day

    benchmarkOrders = pd.Series(index=indexDates, data=zeroes)
    benchmarkOrders.iloc[0] = 1.0  # set to buy LONG on day1
    benchmarkOrders.iloc[(len(prices_all)-1)] = -1.0  # set to sell all on the last day
    # Train and test a StrategyLearner

    stl2 = StrategyLearner(num_shares=num_shares, impact=0.005,
                          commission=commission, verbose=True,
                          num_states=3000, num_actions=3)

    stl2.addEvidence(symbol=benchmarkSymbol, sd=start_date, ed=end_date, sv=start_val)

    learnerOrders2 = stl2.generateOrders(symbol=benchmarkSymbol, sd=start_date, ed=end_date)
    learnerTrades2 = stl2.testPolicy(symbol=benchmarkSymbol, sd=start_date, ed=end_date)

    print "Erika's high impact test In-sample Testing Results"
    # Retrieve performance stats via a market simulator
    print ("Performances during training period for {}".format(benchmarkSymbol))
    print ("Date Range: {} pyto {}".format(start_date, end_date))

    benchmarkPortVal, benchmarkTradesDF, benchmarkHoldingsDF, benchmarkPricesDF = marketsimcode.compute_portvals_single_stock(benchmarkOrders, symbol=benchmarkSymbol,start_val=start_val, commission=commission, impact=0.005, num_shares = num_shares)

    benchmarkPortVal2 = marketsimcode.compute_portVals_from_trades(tradesDF = benchmarkTrades, pricesDF = benchmarkPricesDF, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=0.005)
    #print "benchmarkPortValDF with new function: ", benchmarkPortVal2
    print "benchmarkPortVal: ", benchmarkPortVal2.iloc[-1][0]

    learnerPortVal, learnerTradesDF, learnerHoldingsDF, learnerPricesDF = marketsimcode.compute_portvals_single_stock(learnerOrders2, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=0.005, num_shares = num_shares)

    learnerPortVal2 = marketsimcode.compute_portVals_from_trades(tradesDF = learnerTrades2, pricesDF = learnerPricesDF, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=0.005)
    #print "LearnerPortValDF with new function: ", learnerPortVal2
    print "learnerPortVal: ", learnerPortVal2.iloc[-1][0]




    print "Out-of-sample or testing period"
    # Perform similar steps as above except no training of the data
    start_date = dt.datetime(2010, 1, 1,0,0)
    end_date = dt.datetime(2011, 12, 31,0,0)

    #new benchmark, same criteria
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([benchmarkSymbol], dates).dropna()
    indexDates = prices_all.index
    zeroes = [0.0] * len(prices_all)
    benchmarkTrades = pd.DataFrame({"Date": indexDates, benchmarkSymbol: zeroes})
    benchmarkTrades = benchmarkTrades.set_index('Date')
    benchmarkTrades.iloc[0][0] = 1000 #set to buy LONG on day1
    benchmarkTrades.iloc[(len(prices_all)-1)][0] = -1000  # set to sell all on the last day

    benchmarkOrders = pd.Series(index=indexDates, data=zeroes)
    benchmarkOrders.iloc[0] = 1.0  # set to buy LONG on day1
    benchmarkOrders.iloc[(len(prices_all)-1)] = -1.0  # set to sell all on the last day

    learnerOrders3 = stl.generateOrders(symbol=benchmarkSymbol, sd=start_date, ed=end_date)
    learnerTrades3 = stl.testPolicy(symbol=benchmarkSymbol, sd=start_date, ed=end_date)

    print "Erika's high impact test Out-of-sample Testing Results"
    # Retrieve performance stats via a market simulator
    print ("\nPerformances during testing period for {}".format(benchmarkSymbol))
    print ("Date Range: {} to {}".format(start_date, end_date))

    # Retrieve performance stats via a market simulator
    print ("Performances during training period for {}".format(benchmarkSymbol))
    print ("Date Range: {} pyto {}".format(start_date, end_date))

    benchmarkPortVal, benchmarkTradesDF, benchmarkHoldingsDF, benchmarkPricesDF = marketsimcode.compute_portvals_single_stock(
        benchmarkOrders, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=0.005,
        num_shares=num_shares)

    benchmarkPortVal2 = marketsimcode.compute_portVals_from_trades(tradesDF=benchmarkTrades, pricesDF=benchmarkPricesDF, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=0.005)
    #print "benchmarkPortValDF with new function: ", benchmarkPortVal2
    print "benchmarkPortVal: ", benchmarkPortVal2.iloc[-1][0]

    learnerPortVal, learnerTradesDF, learnerHoldingsDF, learnerPricesDF = marketsimcode.compute_portvals_single_stock( learnerOrders3, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=0.005,
        num_shares=num_shares)

    learnerPortVal2 = marketsimcode.compute_portVals_from_trades(tradesDF=learnerTrades3, pricesDF=learnerPricesDF,
                                                                 symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=0.005)
    #print "LearnerPortValDF with new function: ", learnerPortVal2
    print "learnerPortVal: ", learnerPortVal2.iloc[-1][0]


