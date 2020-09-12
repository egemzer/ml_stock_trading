"""
Test a learner.  (c) 2015 Tucker Balch
"""

import numpy as np
import math
import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import InsaneLearner as il
import BagLearner as bl
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if __name__=="__main__":
    if len(sys.argv) != 2:
        print "Usage: python testlearner.py <filename>"
        sys.exit(1)
    inf = open(sys.argv[1])
    data = np.genfromtxt(inf, delimiter=',')
    print data
    if sys.argv[1] == "Data/istanbul.csv":
        data = data[1:, 1:]
    else:
        #data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])
        data = data[:,:]


    # compute how much of the data is training and testing
    train_rows = int(0.6* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows, 0:-1]
    trainY = data[:train_rows, -1]
    testX = data[train_rows:, 0:-1]
    testY = data[train_rows:, -1]

    #print testX.shape
    #print testY.shape

    # # create a LinReg learner and train it
    # learner = lrl.LinRegLearner(verbose = True) # create a LinRegLearner
    # learner.addEvidence(trainX, trainY) # train it
    # print learner.author()

    # # create a DT learner and train it
    # learner = dt.DTLearner(verbose = False, leaf_size = 1) # create a DTLearner
    # learner.addEvidence(trainX, trainY) # train it
    # print learner.author()

    # # create an RT learner and train it
    # learner = rt.RTLearner(verbose = False) # create an RTLearner
    # learner.addEvidence(trainX, trainY) # train it
    # print learner.author()

    # # create a Bag learner and train it
    # learner = bl.BagLearner(learner = dt.DTLearner, kwargs = {"leaf_size":1}, bags = 20, boost = False, verbose = False) # create a BagLearner
    # learner.addEvidence(trainX, trainY) # train it
    # print learner.author()

    # # create an Insane learner and train it
    # learner = il.InsaneLearner(verbose = True) # create an InsaneLearner
    # learner.addEvidence(trainX, trainY) # train it
    # print learner.author()

    # evaluate in sample ONE TRIAL
    # predY = learner.query(trainX) # get the predictions
    # rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    # print "In sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=trainY)
    # print "corr: ", c[0,1]

    # # evaluate out of sample ONE TRIAL
    # predY = learner.query(testX) # get the predictions
    # rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    # print
    # print "Out of sample results"
    # print "RMSE: ", rmse
    # c = np.corrcoef(predY, y=testY)
    # print "corr: ", c[0,1]

    #evaluate in sample and out of sample MANY TRIALS with different leaf sizes and DTLearner
    leafSize = list(range(1,61))
    #print "leafSize:", leafSize
    RMSE_train_all = []
    RMSE_test_all = []
    for leaf in leafSize:
        # create a DT learner and train it
        learner = dt.DTLearner(verbose = False, leaf_size=leaf) # create a DTLearner
        learner.addEvidence(trainX, trainY) # train it

        predYtrain = learner.query(trainX) # get the predictions from training
        rmseTrain = math.sqrt(((trainY - predYtrain) ** 2).sum()/trainY.shape[0]) #training RMSE
        RMSE_train_all.append(rmseTrain)
        predYtest = learner.query(testX)  # get the predictions for testing
        rmseTest = math.sqrt(((testY - predYtest) ** 2).sum() / testY.shape[0])
        RMSE_test_all.append(rmseTest)

    plt.plot(leafSize, RMSE_train_all, label="In Sample RMSE -- training set")
    plt.plot(leafSize, RMSE_test_all, label="Out of Sample RMSE -- test set")
    plt.legend(loc="lower right")
    plt.title("Overfitting Analysis of DTLearner based on Leaf Size")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.savefig('DTresult.png')
    plt.switch_backend('Agg')


    #evaluate in sample and out of sample MANY TRIALS with different leaf sizes and RTLearner
    leafSize = list(range(1,61))
    #print "leafSize:", leafSize
    RMSE_train_all = []
    RMSE_test_all = []
    for leaf in leafSize:
        # create a DT learner and train it
        learner = rt.RTLearner(verbose = False, leaf_size=leaf) # create a RTLearner
        learner.addEvidence(trainX, trainY) # train it

        predYtrain = learner.query(trainX) # get the predictions from training
        rmseTrain = math.sqrt(((trainY - predYtrain) ** 2).sum()/trainY.shape[0]) #training RMSE
        RMSE_train_all.append(rmseTrain)
        predYtest = learner.query(testX)  # get the predictions for testing
        rmseTest = math.sqrt(((testY - predYtest) ** 2).sum() / testY.shape[0])
        RMSE_test_all.append(rmseTest)

    plt.plot(leafSize, RMSE_train_all, label="In Sample RMSE -- training set")
    plt.plot(leafSize, RMSE_test_all, label="Out of Sample RMSE -- test set")
    plt.legend(loc="lower right")
    plt.title("Overfitting Analysis of RTLearner based on Leaf Size")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.savefig('RTresult.png')
    plt.switch_backend('Agg')

    # #evaluate in sample and out of sample MANY TRIALS with different leaf sizes and DTLearner bagged 50 times
    # leafSize = list(range(1,31))
    # #print "leafSize:", leafSize
    # RMSE_train_all = []
    # RMSE_test_all = []
    # for leaf in leafSize:
    #     # create a BagLearner with 50 bags and train it
    #     learner = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": leaf}, bags=50, boost=False,
    #                             verbose=False)  # create a BagLearner
    #     learner.addEvidence(trainX, trainY) # train it
    #     predYtrain = learner.query(trainX) # get the predictions from training
    #     rmseTrain = math.sqrt(((trainY - predYtrain) ** 2).sum()/trainY.shape[0]) #training RMSE
    #     RMSE_train_all.append(rmseTrain)
    #     predYtest = learner.query(testX)  # get the predictions for testing
    #     rmseTest = math.sqrt(((testY - predYtest) ** 2).sum() / testY.shape[0])
    #     RMSE_test_all.append(rmseTest)
    #
    # plt.plot(leafSize, RMSE_train_all, label="In Sample RMSE -- training set")
    # plt.plot(leafSize, RMSE_test_all, label="Out of Sample RMSE -- test set")
    # plt.legend(loc="lower right")
    # plt.title("Overfitting Analysis of Bagged DTLearner (50bags) based on Leaf Size")
    # plt.xlabel("Leaf Size")
    # plt.ylabel("RMSE")
    # plt.savefig('baggedDTresult.png')
    # plt.switch_backend('Agg')


    # #evaluate in sample and out of sample MANY TRIALS with different leaf sizes and RTLearner bagged 50 times
    # leafSize = list(range(1,61))
    # #print "leafSize:", leafSize
    # RMSE_train_all = []
    # RMSE_test_all = []
    # for leaf in leafSize:
    #     # create a BagLearner with 20 bags and train it
    #     learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": leaf}, bags=20, boost=False,
    #                             verbose=False)  # create a BagLearner
    #     learner.addEvidence(trainX, trainY) # train it
    #     predYtrain = learner.query(trainX) # get the predictions from training
    #     rmseTrain = math.sqrt(((trainY - predYtrain) ** 2).sum()/trainY.shape[0]) #training RMSE
    #     RMSE_train_all.append(rmseTrain)
    #     predYtest = learner.query(testX)  # get the predictions for testing
    #     rmseTest = math.sqrt(((testY - predYtest) ** 2).sum() / testY.shape[0])
    #     RMSE_test_all.append(rmseTest)
    #
    # plt.plot(leafSize, RMSE_train_all, label="In Sample RMSE -- training set")
    # plt.plot(leafSize, RMSE_test_all, label="Out of Sample RMSE -- test set")
    # plt.legend(loc="lower right")
    # plt.title("Overfitting Analysis of Bagged RTLearner (20bags) based on Leaf Size")
    # plt.xlabel("Leaf Size")
    # plt.ylabel("RMSE")
    # plt.savefig('baggedRTresult.png')
    # plt.switch_backend('Agg')