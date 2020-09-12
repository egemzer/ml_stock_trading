"""implement bootstrap aggregating as a python class. BagLearner should accept \
any learner (RTLearner, LinRegLearner, or even another BagLearner"""

import DTLearner as dt
import RTLearner as rt
import LinRegLearner as lrl
import numpy as np

class BagLearner(object):

    def __init__(self, learner, bags = 20, boost = False, verbose = False, kwargs = {}): #constructor
        """Parameters:
                learner: A LinRegLearner, DTLearner, or RTLearner (or even another BagLearner or InsaneLearner)
                bags: The number of learners to be trained using Bootstrap Aggregation (BAgging)
                boost: If true, boosting will be implemented **TODO erika**
                verbose: If True, information about the learner will be printed
                kwargs: Keyword arguments to be passed on to the learner's constructor, ie {"leaf_size":1}
        Returns: An instance of Bag Learner
        """
        self.verbose = verbose
        self.learner = learner
        self.learners = []  # create an empty array where we will store the learners
        self.bags = bags
        for i in range(self.bags):
            self.learners.append(learner(**kwargs)) # create as many learners as the # of bags, using the kwargs
        self.kwargs = kwargs
        self.boost = boost
        if verbose:
            self.get_info()


    def author(self):
        return 'gth659q' # my Georgia Tech username


    def addEvidence(self, dataX, dataY): #Add training data to learner
        """Inputs
            dataX: the X values of data to add, an ndarray (numpy)
            DataY: the Y training values, a 1D ndarray (numpy)
        Returns: updated individual learners in a BagLearner
        """
        num_samples = dataX.shape[0] #number of samples in the data set (number of rows)
        if self.boost==False:
            for learner in self.learners:
                a=np.random.randint(0, high=num_samples, size=num_samples) #randomly selects indices for each bag
                bagX=dataX[a] #randomly selected samples from dataX
                bagY=dataY[a] #corresponding Y values
                learner.addEvidence(bagX, bagY)

        if self.verbose:
            print self.learners

        return self.learners


    def query(self, points): #query the Y values by scanning the tree and average the Y values. Return this as estimate.
        """Input: points: A ndarray of test queries
        Returns: est_vals: A 1D array of the estimated values"""

        est_vals = [] #instantiate the empty list
        for learner in self.learners:
            est_vals.append(learner.query(points))
        est_vals = np.array(est_vals)
        return np.mean(est_vals, axis=0)


    def get_info(self):
        print "About this BagLearner:"
        learner_name = str(type(self.learners[0]))
        print ("This BagLearner is made up of {} {}:".
            format(self.bags, learner_name))

        print ("kwargs =", self.kwargs)
        print ("boost =", self.boost)

        # Information for each learner within BagLearner
        for i in range(1, self.bags + 1):
            print (learner_name, "#{}:".format(i))
            self.learners[i-1].get_info()

def test_code():
    # Some data to test the BagLearner from Professor's example
    x0 = np.array([0.885, 0.725, 0.560, 0.735, 0.610, 0.260, 0.500, 0.320])
    x1 = np.array([0.330, 0.390, 0.500, 0.570, 0.630, 0.630, 0.680, 0.780])
    x2 = np.array([9.100, 10.900, 9.400, 9.800, 8.400, 11.800, 10.500, 10.000])
    x = np.array([x0, x1, x2]).T

    y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

    # Create a BagLearner using DTLearner given training x and y
    bl = BagLearner(dt.DTLearner, verbose=False)
    bl.addEvidence(x, y)

    # Query with dummy data
    #print "DTLearner predicts: ", bl.query(np.array([[1, 2, 3], [0.2, 12, 12]]))

    # Create a BagLearner using RTLearner given training x and y
    bl2 = BagLearner(rt.RTLearner, verbose=False)
    bl2.addEvidence(x, y)

    # Query with dummy data
    #print "RTLearner predicts: ", bl2.query(np.array([[1, 2, 3], [0.2, 12, 12]]))

if __name__ == "__main__":
    test_code()