"""
TODO: add header
"""

from math import log
from collections import OrderedDict

class NaiveBayes:

    def __init__(self, part):
        """
        TODO: inside the constructor, set up all probabilities that will be
        necessary for classifying an example later on
        """

        self.K = part.K
        self.n = part.n
        self.F = part.F
        
        self.K_count = [0 for i in range(self.K)]
        self.f_count = [OrderedDict() for i in range(self.K)]

        # Construct feature counts with 0s (pain)
        for dict in self.f_count:
            for feature in list(self.F):
                dict[feature] = OrderedDict()
                for value in list(self.F[feature]):
                    dict[feature][value] = 0

        # Fill K_count and f_count variables
        for example in part.data:
            self.K_count[example.label] += 1
            for feature in example.features:
                self.f_count[example.label][feature][example.features[feature]] += 1
        
    def classify(self, x_test):
        """
        TODO: based on the dictionary of features x_test, return the most
        likely class (integer)
        """

        probs = []
        for k in range(self.K):
            # Theta_K
            sum = log(self.K_count[k] + 1) - log(self.n + self.K)
            for feature in list(x_test):
                # Prod of Theta_k_j_v
                sum += log(self.f_count[k][feature][x_test[feature]] + 1) - log(self.K_count[k] + len(self.F[feature]))
            probs.append(sum)

        return probs.index(max(probs))