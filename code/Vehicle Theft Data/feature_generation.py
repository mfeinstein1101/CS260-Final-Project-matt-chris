import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ['API_KEY'])

def main():
    df = pd.read_csv('data/vehicle_theft_data.csv', header=0)

    for i in range(0, 10):
        print(get_features(df, i))


def get_features(df, idx):
    row = df.iloc[idx]
    make, model, year = row['make'], row['make/model'], row['year']

    prompt = f'Generate the horsepower, reliability index, and and price of a {make} {model} from {year}. Format all of the results as integers separated by commas.'
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that provides vehicle specifications."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    main()