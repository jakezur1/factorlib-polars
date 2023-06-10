import json
from yahoo_fin import stock_info as si

from factorlib.utils.system import get_data_dir

sp500 = si.tickers_sp500()
nasdaq = si.tickers_nasdaq()

tickers = {'sp500': sp500, 'nasdaq': nasdaq}

raw_data_dir = get_data_dir() / 'raw'
with open(raw_data_dir / 'index_tickers.json', 'w') as file:
    json.dump(tickers, file)

