"""
"""

import pandas as pd
import math
from Partition import Partition, Example
from NaiveBayes import NaiveBayes
import numpy as np
from tabulate import tabulate

def main(): 
    
    df = pd.read_csv('data/vehicle_theft_data.csv', header=0)
    
    features = ['year','manufacturer','make','make/model','production','rate','type']

    df = df[features]
    ave_rate = df['rate'].mean()
    prod_deciles = df.production.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    train = create_partition(df, ave_rate, prod_deciles)
    model = NaiveBayes(train)

    conf = np.zeros((2, 2))

    for example in train.data:
        real = example.label
        pred = model.classify(example.features)

        conf[real][pred] += 1
    
    print('\n\n\n\n\n\n\n')

    table = pd.DataFrame(conf.astype(int))
    print(tabulate(table, headers='keys', tablefmt='simple_grid'))
    print(f'\nAccuracy: {round(np.trace(conf)/np.sum(conf)*100, 3)}% ({int(np.trace(conf))}/{int(np.sum(conf))})\n')


def create_partition(df, ave_rate, prod_deciles):

    feature_list = list(df)
    feature_list.remove('rate')

    data = []

    F_dict = dict.fromkeys(feature_list)
    for f in F_dict:
        F_dict[f] = []

    for index, row in df.iterrows():
        features = dict.fromkeys(feature_list)
        for f in feature_list:
            
            if f != 'production':
                features[f] = row[f]

            if f == 'production':
                for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    if row[f] <= prod_deciles[i]:
                        features[f] = i
                        break

            if features[f] not in F_dict[f]:
                F_dict[f].append(features[f])

        label = 1 if row['rate'] >= ave_rate else 0

        example = Example(features, label)
        data.append(example)

    partition = Partition(data, F_dict, 2)
    return partition


if __name__ == '__main__':
    main()