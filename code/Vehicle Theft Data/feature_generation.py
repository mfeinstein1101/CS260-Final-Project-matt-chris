import pandas as pd
import os
from dotenv import load_dotenv

def main():
    df = pd.read_csv('data/vehicle_theft_data.csv', header=0)

    load_dotenv()
    print(os.environ['API_KEY'])


def row_string(df, idx):
    row = df.iloc[idx]
    return f'{row['year']} {row['make']} {row['make/model']}'

if __name__ == '__main__':
    main()