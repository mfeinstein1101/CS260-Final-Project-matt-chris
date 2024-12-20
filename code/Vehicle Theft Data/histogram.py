import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    df = pd.read_csv('data/vehicle_theft_data.csv', header=0)
    plt.hist(df['rate'], density=True, bins=50)

    plt.xlabel('Theft Rate')
    plt.ylabel('Frequency')
    plt.title('Histogram of Theft Rates')
    
    plt.savefig('code/figures/histogram.pdf', format='pdf')
    plt.show()
    

if __name__ == '__main__':
    main()