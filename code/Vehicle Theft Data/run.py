"""
"""

import pandas as pd
import math
from Partition import Partition, Example
import numpy as np

def main(): 
    
    df = pd.read_csv('data/vehicle_theft_data.csv', header=0)
    
    features = ['year','manufacturer','make','make/model','production','rate','type']

    df = df[features]
    ave_rate = df['rate'].mean()
    prod_deciles = df.production.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    train = create_partition(df, ave_rate, prod_deciles)


def create_partition(df, ave_rate, prod_deciles):

    feature_list = list(df)
    feature_list.remove('rate')

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

    print(F_dict)


if __name__ == '__main__':
    main()