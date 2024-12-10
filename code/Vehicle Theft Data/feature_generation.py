import pandas as pd
import os
from dotenv import load_dotenv
import openai

def main():
    df = pd.read_csv('data/vehicle_theft_data.csv', header=0)

    load_dotenv()
    openai.api_key = os.environ['API_KEY']

    for i in range(0, 10):
        print(get_features(df, i))


def get_features(df, idx):
    row = df.iloc[idx]
    make, model, year = row['make'], row['make/model'], row['year']

    prompt = f'Generate the horsepower, reliability index, and and price of a {make} {model} from {year}. Format all of the results as integers separated by commas.'
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that provides vehicle specifications."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    main()