"""
"""

import pandas as pd
import math
from Partition import Partition, Example

def main(): 
    
    df = pd.read_csv('data/vehicle_theft_data.csv', header=0)
    
    features = ['year','manufacturer','make','make/model','production','rate','type']

    df = df[features]
    ave_rate = df['rate'].mean()

    train = create_partition(df, ave_rate)



def create_partition(df, ave_rate):

    feature_list = list(df)
    feature_list.remove('rate')

    for index, row in df.iterrows():
        features = dict.fromkeys(feature_list)
        for f in feature_list:
            features[f] = row[f]
        label = 1 if row['rate'] >= ave_rate else 0

        example = Example(features, label)

        print(label)


if __name__ == '__main__':
    main()