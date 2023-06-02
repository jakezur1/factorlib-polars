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

print('Creating General Features...')
# fff
general_data_dir = get_data_dir() / 'general'
fff_daily = (
    pl.scan_csv(general_data_dir / 'fff_daily.csv', try_parse_dates=True)
    .collect(streaming=True)
)

fff_factor = Factor(name='fff', data=fff_daily, current_interval='1d', general_factor=True, tickers=tickers)

print('Creating Fundamental Features...')
# fundamentals1_monthly
fundamental_data_dir = get_data_dir() / 'fundamental'
fundamentals1 = pl.scan_csv(fundamental_data_dir / 'fundamentals1_monthly.csv', try_parse_dates=True) \
    .collect(streaming=True)
fundamentals1_factor = Factor(name='fundamentals1', data=fundamentals1,
                              current_interval='1mo', desired_interval='1d')

# div_season
div_season = pl.scan_csv(fundamental_data_dir / 'div_season.csv', try_parse_dates=True) \
    .collect(streaming=True)
div_season_factor = Factor(name='div_season', data=div_season, current_interval='1mo', desired_interval='1d')

# ch_tax
ch_tax = pl.scan_csv(fundamental_data_dir / 'ch_tax.csv', try_parse_dates=True) \
    .collect(streaming=True)
ch_tax_factor = Factor(name='ch_tax', data=ch_tax, current_interval='1mo', desired_interval='1d')


print('Creating Momentum Features...')
# trend_factor
momentum_dir = get_data_dir() / 'momentum'
trend_factor_data = pl.scan_csv(momentum_dir / 'trend_factor.csv', try_parse_dates=True).collect(streaming=True)
trend_factor = Factor(name='trend_factor', data=trend_factor_data, current_interval='1mo', desired_interval='1d')

print('Creating Model and Adding Factors...')
model = FactorModel(tickers=tickers, interval='1d')

model.add_factor(fff_factor)
model.add_factor(fundamentals1_factor)
model.add_factor(div_season_factor)
model.add_factor(trend_factor)

stats = model.wfo(returns_data,
                  train_interval=relativedelta(years=5), anchored=False,  # interval parameters
                  start_date=datetime(2013, 1, 1), end_date=datetime(2019, 1, 1),
                  k_pct=0.2, long_pct=0.5)  # weight parameters

stats.print_statistics_report()
stats.get_html()
stats.save(get_results_dir() / 'wfo_stats.pkl')
model.save(get_results_dir() / 'current_model.pkl')
