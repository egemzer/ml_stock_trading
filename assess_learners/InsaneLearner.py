"""implement InsaneLearner to prove that BagLearner is working \
The code for InsaneLearner should be 20 lines or less. """

import BagLearner as bl
import LinRegLearner as lrl
import numpy as np

class InsaneLearner(object):

    def __init__(self, verbose = False, baglearner=bl.BagLearner, otherlearner=lrl.LinRegLearner, num_bag_learners = 20, kwargs={}): #constructor
        """Parameters:
                baglearner: A BagLearner
                otherlearner: A LinRegLearner, DTLearner, or RTLearner to be called by bag_learner
                num_bag_learners: The number of Bag learners to be trained
                verbose: always false, this learner doesn't print info
                kwargs: Keyword arguments to be passed on to the bag learner's constructor

        Returns: An instance of Insane Learner
        """
        self.verbose = verbose
        self.baglearner = baglearner
        bag_learners = [] #empty array to add each learner
        for i in range(num_bag_learners):
            bag_learners.append(self.baglearner(learner=otherlearner, **kwargs))
        self.bag_learners = bag_learners
        self.kwargs = kwargs
        self.num_bag_learners = num_bag_learners

    def author(self):
        return 'gth659q' # my Georgia Tech username


    def addEvidence(self, dataX, dataY): #train
        for bag_learner in self.bag_learners:
            bag_learner.addEvidence(dataX, dataY)


    def query(self, points):  #estimate a set of test points given the model.
        """Input: points: A ndarray of test queries
        Returns: est_vals: A 1D array of the estimated values"""

        est_vals = []  # instantiate the empty list
        for learner in self.bag_learners:
            est_vals.append(learner.query(points))
        est_vals = np.array(est_vals)
        return np.mean(est_vals, axis=0)