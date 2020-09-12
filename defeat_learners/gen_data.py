"""
template for generating data to fool learners (c) 2016 Tucker Balch
"""

import numpy as np
import math
import DTLearner as dt
import LinRegLearner as lrl


# this function should return a dataset (X and Y) that will work
# better for linear regression than decision trees
def best4LinReg(seed=1489683273):
    """creates and returns a dataset better for linear regression learners than for decision tree learners
    Input: seed value, a random value. Different seeds should result in different datasets
    Returns X and Y: both numpy arrays, features (X, 2D) and values (Y, 1D) ideal for linear regression learning
    """
    np.random.seed(seed)  # seed a random number generator
    num_rows = np.random.randint(10, 500)  # X and Y can have 10 to 500 rows /samples
    num_features = np.random.randint(2, 15)  # X can have 2 to 15 features (columns)
    X = np.random.normal(size=(num_rows, num_features))
    Y = np.ones(num_rows)
    for feature in range(num_features):
        Y = Y + X[:, feature]  # Y is the sum of the X's for each sample, a linear equation

    return X, Y


def best4DT(seed=1489683273):
    """creates and returns a dataset better for decision tree learners than for linear regression learners
    Input: seed value, a random value. Different seeds should result in different datasets
    Returns X and Y: both numpy arrays, features (X, 2D) and values (Y, 1D) ideal for decision tree learning
    """
    np.random.seed(seed)  # seed a random number generator
    num_rows = np.random.randint(10, 500)  # X and Y can have 10 to 500 rows /samples
    num_features = np.random.randint(2, 15)  # X can have 2 to 15 features (columns)
    X = np.random.normal(size=(num_rows, num_features))
    Y = np.ones(num_rows)
    for feature in range(num_features):
        Y = X[:, feature] ** 3  # Y is the sum of the exponent of the X's (nonlinear)
    return X, Y


def author():
    return 'gth659q'  # my Georgia Tech username


# def test_code():
#     # create data best for LinRegLearner
#     print "check to make sure that the dimensions fit the project requirements"
#     lrlX, lrlY = best4LinReg(seed=5420)
#     print "lrlX, lrlY: ", lrlX.shape, lrlY.shape
#     # create data best for DTLearner
#     dtX, dtY = best4DT(seed=5430)
#     print "dtlX, dtY: ", dtX.shape, dtY.shape
#
#     # Create LinRegLearner and DTLearner from the data best for LinRegLearner
#     learner1 = dt.DTLearner(leaf_size=1, verbose=False)
#     learner2 = lrl.LinRegLearner(verbose=False)
#     # add data
#     learner1.addEvidence(lrlX, lrlY)
#     learner2.addEvidence(lrlX, lrlY)
#
#     # Create LinRegLearner and DTLearner from the data best for DTLearner
#     learner3 = dt.DTLearner(leaf_size=1, verbose=False)
#     learner4 = lrl.LinRegLearner(verbose=False)
#     # add data
#     learner3.addEvidence(dtX, dtY)
#     learner4.addEvidence(dtX, dtY)


# if __name__ == "__main__":
    # print "You've called the gen_data function"
    #test_code()
