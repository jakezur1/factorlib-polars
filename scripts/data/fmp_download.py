import requests
import pandas as pd
from factorlib.utils.system import get_data_dir
from tqdm import tqdm
import os

API_KEY = os.environ.get('API_KEY')
raw_data_dir = get_data_dir() / 'raw'
tickers = pd.read_csv(raw_data_dir / 'tickers_finance_statement.csv', index_col=0)
call_count = 0
limit = 400

base_url = 'https://financialmodelingprep.com'
all_financial_statements = pd.DataFrame()

for ticker in tqdm(tickers['ticker']):
    url = base_url + f'/api/v3/income-statement/{ticker}?period=quarter&data_type=csv&limit=400&apikey={API_KEY}'
    response = requests.get(url)
    if response.status_code == 200:
        curr_ticker = pd.DataFrame(response.json())
        all_financial_statements = pd.concat([all_financial_statements, curr_ticker])
    else:
        print(f'Request failed with status code {response.status_code}')

all_financial_statements.to_csv(raw_data_dir / 'financial_statements.csv')
