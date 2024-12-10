import pandas as pd

def main():
    df = pd.read_csv('data/vehicle_theft_data.csv', header=0)
    print(row_string(df, 1))


def row_string(df, idx):
    row = df.iloc[idx]
    return f'{row['year']} {row['make']} {row['make/model']}'

if __name__ == '__main__':
    main()