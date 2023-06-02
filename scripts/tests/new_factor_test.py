import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta

from factorlib.factor import Factor
from factorlib.factor_model import FactorModel
from factorlib.utils.system import get_data_dir, get_results_dir

raw_data_dir = get_data_dir() / "raw"

print('Reading in Stock Data...')
stocks_data = pl.scan_csv(raw_data_dir / 'spy_data_daily.csv', try_parse_dates=True).rename(
    {'Date': 'date_index'}).collect(streaming=True)

tickers = stocks_data.columns[1:]
print("Universe of Tickers: ", len(tickers), " Total")
stocks_data = stocks_data.select(pl.col('date_index'), pl.all().exclude('date_index').cast(float))
returns_data = stocks_data.with_columns(pl.col('date_index'),
                                        pl.all().exclude('date_index').pct_change())

print("Loading Most Up-to-Date Model...")
model = FactorModel(tickers=tickers, interval='1d')
model.load(get_results_dir() / 'current_model.pkl')

print('Creating New Factor...')


stats = model.wfo(returns_data,
                  train_interval=relativedelta(years=5), anchored=False,  # interval parameters
                  start_date=datetime(2013, 1, 1), end_date=datetime(2019, 1, 1),
                  k_pct=0.2, long_pct=0.5)  # weight parameters

stats.print_statistics_report()
stats.get_html()
stats.save(get_results_dir() / 'wfo_stats.pkl')
model.save(get_results_dir() / 'new_factor_model.pkl')
