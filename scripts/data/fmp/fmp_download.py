import requests
import pandas as pd
from factorlib.utils.system import get_data_dir
from tqdm import tqdm
import os
from io import StringIO

API_KEY = os.environ.get('API_KEY')

raw_data_dir = get_data_dir() / 'raw'
tickers = pd.read_csv(raw_data_dir / 'tickers_finance_statement.csv', index_col=0)

base_url = 'https://financialmodelingprep.com'
data_to_download = 'historical-price-full'
version = 'v3'


all_fmp_data = pd.DataFrame()
for ticker in tqdm(tickers['ticker']):
    url = base_url + f'/api/{version}/{data_to_download}/{ticker}?datatype=csv&apikey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        # tic = response.json()['symbol']
        content = StringIO(response.content.decode())
        try:
            curr_ticker = pd.read_csv(content)
            tic = pd.DataFrame({'ticker': [ticker] * len(curr_ticker)})
            curr_ticker = pd.concat([tic, curr_ticker], axis=1)
            all_fmp_data = pd.concat([all_fmp_data, curr_ticker])
        except Exception:
            continue
    else:
        print(f'Request failed with status code {response.status_code}')

filename = data_to_download.replace("-", "_")
all_fmp_data.to_csv(raw_data_dir / f'{filename}.csv')
