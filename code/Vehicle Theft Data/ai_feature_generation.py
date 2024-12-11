# import pandas as pd
# import time
# import os
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()
# client = OpenAI(api_key=os.environ['API_KEY'])

# def main():
#     df = pd.read_csv('data/vehicle_theft_data.csv', header=0)

#     horsepower, reliability, price = [], [], []

#     for idx, row in df.iterrows():

#         hp, rel, pr = get_features(row)
        
#         horsepower.append(hp)
#         reliability.append(rel)
#         price.append(pr)

#         print(f'{idx}: {row['year']} {row['make']} {row['make/model']}: ({hp}, {rel}, {pr})')

#         time.sleep(0.2)
    
#     df['Horsepower'] = horsepower
#     df['Reliability'] = reliability
#     df['Price'] = price

#     df.to_csv('data/AI_vehicle_theft_data.csv', index=False)


# def get_features(row):
#     make, model, year = row['make'], row['make/model'], row['year']

#     prompt = (
#         f"Provide the horsepower, reliability index, and price for the {year} {make} {model} "
#         "as integers separated by a comma. For example: '250, 85, 17000'. If data is unavailable, return '0, 0, 0'."
#     )

#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You are an assistant that provides vehicle specifications."},
#                 {"role": "user", "content": prompt}
#             ]
#         )
#         content = response.choices[0].message.content
#         hp, rel, pr = map(int, content.split(','))
#         return hp, rel, pr
#     except Exception as e:
#         return 0, 0, 0

# if __name__ == '__main__':
#     main()