import requests
import pandas as pd
from factorlib.utils.system import get_data_dir
from tqdm import tqdm
import os

API_KEY = os.environ.get('API_KEY')

raw_data_dir = get_data_dir() / 'raw'
tickers = pd.read_csv(raw_data_dir / 'tickers_finance_statement.csv', index_col=0)

base_url = 'https://financialmodelingprep.com'
data_to_download = 'earnings-surprises'
version = 'v3'

all_fmp_data = pd.DataFrame()
for ticker in tqdm(tickers['ticker']):
    url = base_url + f'/api/{version}/{data_to_download}/{ticker}?period=quarter&apikey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        curr_ticker = pd.DataFrame(response.json())
        all_fmp_data = pd.concat([all_fmp_data, curr_ticker])
    else:
        print(f'Request failed with status code {response.status_code}')

filename = data_to_download.replace("-", "_")
all_fmp_data.to_csv(raw_data_dir / f'{filename}.csv')
