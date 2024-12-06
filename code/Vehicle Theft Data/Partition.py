"""
"""

from math import log2

class Example:

    def __init__(self, features, label):
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {0, 1}

class Partition:

    def __init__(self, data, F, K):
        """Store information about a dataset"""
        # list of Examples
        self.data = data
        self.n = len(self.data)

        # dictionary. key=feature name: value=list of possible values
        self.F = F

        # number of classes
        self.K = K

    def best_feature(self):

        # Class probabilities
        num_pos = 0
        for example in self.data:
            if example.label == 1:
                num_pos += 1
        
        class_prob = [(self.n-num_pos)/self.n, num_pos/self.n]

        
        # Entropy Calculation
        H = sum([-prob*log2(prob) for prob in class_prob])        
        
        # Conditional Entropy Calculations (Using helper)
        con_H = {}
        for feature in self.F:
            con_H[feature] = self.feature_entropy(feature)
        
        # Convert to Gain
        gain = {}
        for feature in self.F:
            gain[feature] = H - con_H[feature]
        
        # Info printout
        print('\nInfo Gain:')
        for feature in gain:
            print(f'{feature}, {round(gain[feature], 6)}')
        print()

        # Return best feature from max gain
        best_f = max(gain, key=gain.get)
        return best_f

    # H(Y | X)
    def feature_entropy(self, feature):
        
        sum = 0
        for val in self.F[feature]:
            count = 0
            for example in self.data:
                if example.features[feature] == val:
                    count += 1
            sum += count/self.n * self.value_entropy(feature, val)

        return sum
    
    # H(Y | X = v)
    def value_entropy(self, feature, val):
        
        sum = 0
        for k in [-1, 1]:
            num = 0
            denom = 0
            for example in self.data:
                if example.features[feature] == val:
                    denom += 1
                    if example.label == k:
                        num += 1
            prob = 0
            if denom > 0:
                prob = num/denom
            if prob > 0:
                sum -= prob * log2(prob)
        
        return sum

        