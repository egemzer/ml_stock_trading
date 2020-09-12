"""
This goes with the StrategyLearner

Code implemented by: Erika Gemzer, gth659q, Summer 2018

"""

import datetime as dt
import pandas as pd
from util import get_data
from marketsimcode import compute_portVals_from_trades, compute_portvals_single_stock, compute_portfolio_stats
from StrategyLearner import StrategyLearner as sl
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def author():
    return 'gth659q'  # my Georgia Tech username.

def benchmark(start_val = 100000, benchmarkSymbol = "JPM", commission = 0.00, impact = 0.00, num_shares = 1000, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31)):

    # Create benchmark data series. Benchmark is a portfolio starting with
    # $100,000, investing in 1000 shares of symbol and holding that position
    dates = pd.date_range(sd, ed)
    prices_all = get_data([benchmarkSymbol], dates).dropna()
    print "Prices during in sample period: ", prices_all
    indexDates = prices_all.index
    zeroes = [0.0] * len(prices_all)

    # Benchmark trades and orders
    benchmarkTrades = pd.DataFrame({"Date": indexDates, benchmarkSymbol: zeroes})
    benchmarkTrades = benchmarkTrades.set_index('Date')
    benchmarkTrades.iloc[0][0] = 1000  # set to buy LONG on day1
    benchmarkTrades.iloc[(len(prices_all) - 1)][0] = -1000  # set to sell all on the last day

    benchmarkOrders = pd.Series(index=indexDates, data=zeroes)
    benchmarkOrders.iloc[0] = 1.0  # set to buy LONG on day1
    benchmarkOrders.iloc[(len(prices_all) - 1)] = -1.0  # set to sell all on the last day

    # Train and test a StrategyLearner
    stl = sl(num_shares=num_shares, impact=impact,
             commission=commission, verbose=True,
             num_states=3000, num_actions=3)

    stl.addEvidence(symbol=benchmarkSymbol, sd=sd, ed=ed, sv=start_val)

    # Learner trades and orders
    learnerTrades = stl.testPolicy(symbol=benchmarkSymbol, sd=sd, ed=ed)
    learnerOrders = stl.generateOrders(symbol=benchmarkSymbol, sd=sd, ed=ed)

    # Retrieve benchmark performance stats via a market simulator
    benchmarkPortVal, benchmarkTradesDF, benchmarkHoldingsDF, benchmarkPricesDF = compute_portvals_single_stock(
        benchmarkOrders, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact,
        num_shares=num_shares)

    benchmarkPortVal2 = compute_portVals_from_trades(tradesDF=benchmarkTrades, pricesDF=benchmarkPricesDF,
                                                                   symbol=benchmarkSymbol, start_val=start_val,
                                                                   commission=commission, impact=impact)
    benchmarkNormedReturns = benchmarkPortVal2 / benchmarkPortVal2.iloc[0][0]
    benchmarkDates = benchmarkNormedReturns.index.values

    # Retrieve learner performance stats via a market simulator
    learnerPortVal, learnerTradesDF, learnerHoldingsDF, learnerPricesDF = compute_portvals_single_stock(
        learnerOrders, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact,
        num_shares=num_shares)

    learnerPortVal2 = compute_portVals_from_trades(tradesDF=learnerTrades, pricesDF=learnerPricesDF,
                                                                 symbol=benchmarkSymbol, start_val=start_val,
                                                                 commission=commission, impact=impact)

    learnerNormedReturns = learnerPortVal2 / learnerPortVal2.iloc[0][0]
    print "in sample learner portfolio cumulative daily returns: ", learnerNormedReturns
    learnerDates = learnerNormedReturns.index.values

    plt.plot(benchmarkDates, benchmarkNormedReturns, label="Benchmark Portfolio")
    plt.plot(learnerDates, learnerNormedReturns, label="Learned Portfolio")
    plt.xlabel("Dates")
    plt.ylabel("Cumulative return (%)")
    plt.legend(loc="best")
    plt.title("In-Sample Normed Returns for {}: Benchmark vs Learner".format(benchmarkSymbol))
    plt.savefig('experiment1insample.png')
    plt.switch_backend('Agg')



    #Out-of-sample or testing period
    # Perform similar steps as above except no training of the data
    start_date = dt.datetime(2010, 1, 1, 0, 0)
    end_date = dt.datetime(2011, 12, 31, 0, 0)

    # new benchmark, same criteria
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data([benchmarkSymbol], dates).dropna()
    print "Prices during out of sample period: ", prices_all
    indexDates = prices_all.index
    zeroes = [0.0] * len(prices_all)

    #Benchmark trades and orders
    benchmarkTrades = pd.DataFrame({"Date": indexDates, benchmarkSymbol: zeroes})
    benchmarkTrades = benchmarkTrades.set_index('Date')
    benchmarkTrades.iloc[0][0] = 1000  # set to buy LONG on day1
    benchmarkTrades.iloc[(len(prices_all) - 1)][0] = -1000  # set to sell all on the last day

    benchmarkOrders = pd.Series(index=indexDates, data=zeroes)
    benchmarkOrders.iloc[0] = 1.0  # set to buy LONG on day1
    benchmarkOrders.iloc[(len(prices_all) - 1)] = -1.0  # set to sell all on the last day

    #Learner trades and orders
    learnerOrders = stl.generateOrders(symbol=benchmarkSymbol, sd=start_date, ed=end_date)
    learnerTrades = stl.testPolicy(symbol=benchmarkSymbol, sd=start_date, ed=end_date)

    # Retrieve benchmark performance stats via a market simulator
    benchmarkPortVal, benchmarkTradesDF, benchmarkHoldingsDF, benchmarkPricesDF = compute_portvals_single_stock(
        benchmarkOrders, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact,
        num_shares=num_shares)

    benchmarkPortVal2 = compute_portVals_from_trades(tradesDF=benchmarkTrades, pricesDF=benchmarkPricesDF,
                                                                   symbol=benchmarkSymbol, start_val=start_val,
                                                                   commission=commission, impact=impact)

    benchmarkNormedReturns = benchmarkPortVal2 / benchmarkPortVal2.iloc[0][0]
    benchmarkDates = benchmarkNormedReturns.index.values

    # Retrieve learner performance stats via a market simulator
    learnerPortVal, learnerTradesDF, learnerHoldingsDF, learnerPricesDF = compute_portvals_single_stock(
        learnerOrders, symbol=benchmarkSymbol, start_val=start_val, commission=commission, impact=impact,
        num_shares=num_shares)

    learnerPortVal2 = compute_portVals_from_trades(tradesDF=learnerTrades, pricesDF=learnerPricesDF,
                                                                 symbol=benchmarkSymbol, start_val=start_val,
                                                                 commission=commission, impact=impact)
    learnerNormedReturns = learnerPortVal2 / learnerPortVal2.iloc[0][0]
    learnerDates = learnerNormedReturns.index.values

    plt.plot(benchmarkDates, benchmarkNormedReturns, label="Benchmark Portfolio")
    plt.plot(learnerDates, learnerNormedReturns, label="Learned Portfolio")
    plt.xlabel("Dates")
    plt.ylabel("Cumulative return (%)")
    plt.legend(loc="best")
    plt.title("Out-of-Sample Normed Returns for {}: Benchmark vs Learner".format(benchmarkSymbol))
    plt.savefig('experiment1outofsample.png')
    plt.switch_backend('Agg')




if __name__=="__main__":
    benchmark(start_val = 100000, benchmarkSymbol = "WMT", commission = 0.00,
              impact = 0.00, num_shares = 1000, sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31))




