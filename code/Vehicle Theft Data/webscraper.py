# import requests
# from bs4 import BeautifulSoup
# import pandas as pd

# base_url = 'https://www.nhtsa.gov/vehicle-theft-data?field_manufacturer_target_id=All&field_theft_type_value=All&field_theftyear_value_1=All&order=field_theft_rate&sort=desc'

# data = []
# headers = ['year', 'manufacturer', 'make', 'make/model', 'thefts', 'production', 'rate', 'type']

# for page in range(0, 121):
#     print(f"Scraping page {page}...")
#     response = requests.get(f"{base_url}&page={page}")
#     soup = BeautifulSoup(response.text, 'html.parser')

#     table = soup.find('table', {'class': 'cols-8 table d8-port views-table'})
#     if table:
#         rows = table.find_all('tr')
#         for row in rows:
#             cells = [cell.text.strip() for cell in row.find_all('td')]
#             if cells:
#                 data.append(cells)
#     else:
#         print('No table found')

# if data:
#     df = pd.DataFrame(data, columns=headers)
#     # df.to_csv('vehicle_theft_data.csv', index=False)
#     print("Data has been saved to 'vehicle_theft_data.csv'.")