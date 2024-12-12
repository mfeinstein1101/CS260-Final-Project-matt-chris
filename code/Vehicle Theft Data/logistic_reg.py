import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():

    df = pd.read_csv('data/AI_vehicle_theft_data.csv', header=0)
    df = df[df['horsepower'] != 0]

    X = df[['year', 'production', 'horsepower', 'reliability', 'price']]
    
    mean_rate = df['rate'].mean()
    y = [1 if row['rate'] >= mean_rate else 0 for idx, row in df.iterrows()]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = LogisticRegression().fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    conf = np.zeros((2, 2))
    for i in range(len(y_test)):
        conf[y_test[i]][y_pred[i]] += 1
    
    print(conf)
    print(f'Accuracy: {clf.score(X_test, y_test)}')


if __name__ == '__main__':
    main()