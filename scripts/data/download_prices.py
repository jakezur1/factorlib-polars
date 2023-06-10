import yfinance as yf
import pandas as pd

from factorlib.utils.system import get_data_dir

raw_data_dir = get_data_dir() / "raw"
training_tickers = pd.read_csv(raw_data_dir / 'tickers_to_train.csv')['ticker'].tolist()

stocks_data = yf.download(training_tickers, '1990-01-01', '2023-06-01')['Adj Close']
stocks_data.to_csv(raw_data_dir / 'trainable_ticker_prices.csv')
