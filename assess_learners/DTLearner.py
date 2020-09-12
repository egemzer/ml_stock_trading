"""building a single decision tree learner class. \
do not generate statistics or charts"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

class DTLearner(object):
    """JR Quinlan approach, regression DT due to continuous data."""

    def __init__(self, leaf_size = 1, verbose = False): #constructor
        self.leaf_size = leaf_size
        self.tree = np.array([0,0,0,0])
        self.verbose = verbose
        pass

    def author(self):
        return 'gth659q' # my Georgia Tech username

    def __buildTree(self, dataX, dataY):
        """Builds a Decision Tree recursively. Selects the best feature to split on and the ideal split value.
        The best feature has the highest absolute correlation with dataY. If all features have the same absolute correlation,
        we choose the first feature. The splitting value is the median of the data in that feature.

        Inputs:
        dataX: A numpy ndarray of X values
        dataY: A numpy 1D array of Y values

        Returns: tree, a matrix / nparray. Columns are feature indices, and each row is a node. Column headers:
        (1)factor (featureID, for a leaf is -1), (2) split value, (3 and 4) starting rows, from the current
        root, for left and right subtrees.
        """

        # establish the number of samples (rows) and features (num columns) in dataX
        num_samples = dataX.shape[0]
        num_features = dataX.shape[1]
        #print ("num_samples: ", num_samples)
        #print ("num_features: ", num_features)
        leaf = np.array([-1, np.mean(dataY), np.nan, np.nan]) #leaf value is the mean of dataY for the given samples

        #if there is only one sample, return a leaf
        if num_samples == 1:
            return leaf

        #if there are <= leaf_size samples or all the data in dataY are the same value, return a leaf
        if num_samples <= self.leaf_size or np.all(dataY==dataY[0]):
            return leaf

        else: #we have the beginnings of a tree!
            avail_feats = list(range(num_features)) #list of all features available to be split

            # determine best feature i to split on based on highest absolute value correlation with Y
            x_corrs = [] #empty list to store correlation coefficients
            for col in range(num_features): #for each feature
                absCorr = abs(pearsonr(dataX[:, col], dataY)[0]) #use pearson, take absolute values since range is -1 to 1
                if np.isnan(absCorr):
                    absCorr = 0.0
                x_corrs.append((col, absCorr)) #add correlation to list

            x_corrs = sorted(x_corrs, key=lambda rec: rec[1], reverse=True) # sort list of correlations in descending order

            # Choose the best feature, if any, to split on by iterating over x_corrs
            x_corr_i = 0 #index for the feature correlation coeffs, which are already sorted
            while len(avail_feats) > 0:
                best_feat_i = x_corrs[x_corr_i][0] #get the feature index of the feature with the highest correlation

                # Split the data on the highest correlation feature (or first, if all equal)
                SplitVal = np.median(dataX[:, best_feat_i])

                # Index using SplitVal comparison
                left_index = dataX[:, best_feat_i] <= SplitVal
                #print ("left index: ", left_index)
                right_index = dataX[:, best_feat_i] > SplitVal
                #print ("right index: ", right_index)

                # If we can split the data in the index into two groups, break out of the loop to continue
                if len(np.unique(left_index)) != 1:
                    break

                #remove this feature from the available features
                avail_feats.remove(best_feat_i)
                x_corr_i += 1
                #print("Indexing x_corr_i: ", x_corr_i, "and removing this feature ", avail_feats)

            # If we complete the while loop and run out of features to split, return leaf
            if len(avail_feats) == 0:
                #print ("you've completed the while loop and run out of features to split!")
                return leaf

            # Build left and right branches
            lefttree = self.__buildTree(dataX[left_index], dataY[left_index])
            righttree = self.__buildTree(dataX[right_index], dataY[right_index])

            # Set the starting row for the right subtree of the current root
            if lefttree.ndim == 1:
                righttree_start = 2  # The right subtree starts 2 rows down
            elif lefttree.ndim > 1:
                righttree_start = lefttree.shape[0] + 1
            root = np.array([best_feat_i, SplitVal, 1, righttree_start]) #the root of this tree
            #print ("new stack: ", np.vstack((root, lefttree, righttree)))
            self.tree = np.vstack((root, lefttree, righttree)) #the tree, appending new nodes
            #print self.tree
            return self.tree


    def addEvidence(self, dataX, dataY): #Add training data to learner
        """Inputs
            dataX: the X values of data to add, an ndarray (numpy)
            DataY: the Y training values, a 1D ndarray (numpy)
        Returns: an updated decision tree matrix (ndarray)
        """
        newTree = self.__buildTree(dataX, dataY)

        if self.verbose:
            self.get_info()

    def __tree_search(self, point, row):
        """A helper function for query. It recursively searches the decision
        tree matrix and returns a predicted value for a data point
        Inputs:
            point: A 1D numpy array of a test query
            row: The row of the decision tree matrix to search
        Returns: prediction: The predicted Y value
        """

        # Get the feature on the row and its corresponding splitting value
        feat, SplitVal = self.tree[row, 0:2]
        #print("self.tree in treesearch: ", self.tree)
        #print ("row: ", row)
        #print ("Feat, SplitVal, point, row ", feat, SplitVal, point, row)
        #print ("point[int(feat)]: ", point[int(feat)])

        # If the factor of feature is -1, we have reached a leaf, return the SplitVal
        if feat == -1:
            return SplitVal

        # If the corresponding feature's value from point <= SplitVal, go to the left tree
        elif point[int(feat)] <= SplitVal:
            prediction = self.__tree_search(point, row + int(self.tree[row, 2]))

        # Otherwise, go to the right tree
        else:
            prediction = self.__tree_search(point, row + int(self.tree[row, 3]))
        return prediction

    def query(self, points): #estimate Y values for a set of test points given the model.
        """ Input: points: A ndarray of test queries
        Returns: est_vals: A 1D array of the estimated values
        """
        est_vals = []  # instantiate the empty list
        for point in points:
            est_vals.append(self.__tree_search(point, row=0))
        return np.asarray(est_vals)

    def get_info(self):
        print "about this DTLearner:"
        print "leaf_size =", self.leaf_size
        if not np.all(self.tree==self.tree[0]):
            print "tree shape =", self.tree.shape
            # Create a user-friendly view of the tree including the nodes
            tree = pd.DataFrame(self.tree, columns=["factor", "split_val", "left", "right"])
            tree.index.name = "node"
            print (tree)

def test_code():
    # Some data to test the DTLearner from Professor's example
    x0 = np.array([0.885, 0.725, 0.560, 0.735, 0.610, 0.260, 0.500, 0.320])
    x1 = np.array([0.330, 0.390, 0.500, 0.570, 0.630, 0.630, 0.680, 0.780])
    x2 = np.array([9.100, 10.900, 9.400, 9.800, 8.400, 11.800, 10.500, 10.000])
    x = np.array([x0, x1, x2]).T

    y = np.array([4.000, 5.000, 6.000, 5.000, 3.000, 8.000, 7.000, 6.000])

    # Create a tree learner from given train X and y
    dtl = DTLearner(leaf_size=1, verbose=True)

    #adding data
    dtl.addEvidence(x, y)

    # Query with dummy data
    #print ("dtl.tree in test_code: ", dtl.tree)
    dtl.query(np.array([[1, 2, 3], [0.2, 12, 12]]))


if __name__ == "__main__":
    test_code()