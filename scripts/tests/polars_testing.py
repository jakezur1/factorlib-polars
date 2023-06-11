import json
import polars as pl
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

from factorlib.factor import Factor
from factorlib.factor_model import FactorModel
from factorlib.utils.system import get_data_dir, get_results_dir
from factorlib.transforms.expr_shortcuts import calculate_returns

raw_data_dir = get_data_dir() / "raw"

print('Reading in Stock Data...')
tradeable_tickers = pd.read_csv(raw_data_dir / 'tickers_to_trade.csv')['ticker'].tolist()
print("Universe of Tickers: ", len(tradeable_tickers), " Total")

returns_data = pl.read_csv(raw_data_dir / 'training_returns.csv', try_parse_dates=True)

print('Creating General Features...')
# fff
general_data_dir = get_data_dir() / 'general'
# fff_daily = (
#     pl.scan_csv(general_data_dir / 'fff_daily.csv', try_parse_dates=True)
#     .collect(streaming=True)
# )
#
# fff_factor = Factor(name='fff', data=fff_daily, current_interval='1d', general_factor=True, tickers=tickers)

print('Creating Fundamental Features...')
fundamental_data_dir = get_data_dir() / 'fundamental'

# industry relative p/e
ir_pe = pl.scan_csv(fundamental_data_dir / 'pe_analysis.csv', try_parse_dates=True).collect(streaming=True)
ir_pe_factor = Factor(name='ir_pe', data=ir_pe,
                      current_interval='1mo', desired_interval='1d')

# current ratio analysis
curr_ratio_analysis = pl.scan_csv(fundamental_data_dir / 'curr_ratio_analysis.csv', try_parse_dates=True).collect(
    streaming=True)
curr_ratio_factor = Factor(name='cr', data=curr_ratio_analysis,
                           current_interval='1mo', desired_interval='1d')

# div_season
# div_season = pl.scan_csv(fundamental_data_dir / 'div_season.csv', try_parse_dates=True) \
#     .collect(streaming=True)
# div_season_factor = Factor(name='div_season', data=div_season, current_interval='1mo', desired_interval='1d')

# ch_tax
# ch_tax = pl.scan_csv(fundamental _data_dir / 'ch_tax.csv', try_parse_dates=True) \
#     .collect(streaming=True)
# ch_tax_factor = Factor(name='ch_tax', data=ch_tax, current_interval='1mo', desired_interval='1d')
#
# asset_growth = pl.scan_csv(fundamental_data_dir / 'asset_growth.csv', try_parse_dates=True) \
#     .collect(streaming=True)
# asset_growth_factor = Factor(name='asset_growth', data=asset_growth, current_interval='1mo', desired_interval='1d')

print('Creating Momentum Features...')
momentum_dir = get_data_dir() / 'momentum'

# 5 and 21 rolling regression momentum
# regression_momentum = pl.scan_csv(momentum_dir / 'reg_momentum_5_21.csv', try_parse_dates=True).collect(streaming=True)
# regression_momentum_factor = Factor(name='5_21_mom', data=regression_momentum, current_interval='1d', desired_interval='1d')

# 5 and 21 rolling regression momentum ranked
# ranked_momentum = pl.scan_csv(momentum_dir / 'ranked_momentum_5_21.csv', try_parse_dates=True).collect(streaming=True)
# ranked_momentum_factor = Factor(name='5_21_mom_ranked', data=ranked_momentum, current_interval='1d', desired_interval='1d')

# trend_factor
# trend_factor_data = pl.scan_csv(momentum_dir / 'trend_factor.csv', try_parse_dates=True).collect(streaming=True)
# trend_factor = Factor(name='trend_factor', data=trend_factor_data, current_interval='1mo', desired_interval='1d')

# momentum_seasonality monthly/daily
# mom_season_short_monthly = pl.scan_csv(momentum_dir / 'mom_season_short_monthly.csv',
#                                        try_parse_dates=True).collect(streaming=True)
# mom_season_short_monthly_factor = Factor(name='mom_season_short_monthly', data=mom_season_short_monthly,
#                                          current_interval='1mo', desired_interval='1d')
# mom_season_short_daily = pl.scan_csv(momentum_dir / 'mom_season_short_daily.csv',
#                                      try_parse_dates=True).collect(streaming=True)
# mom_season_short_daily_factor = Factor(name='mom_season_short_daily', data=mom_season_short_daily,
#                                        current_interval='1d')

print('Creating Model and Adding Factors...')
model = FactorModel(tickers=tradeable_tickers, interval='1d')

# model.add_factor(fff_factor)
# model.add_factor(fundamentals1_factor)
# model.add_factor(div_season_factor)
model.add_factor(ir_pe_factor)
model.add_factor(curr_ratio_factor)
# model.add_factor(rsi_factor)
# model.add_factor(ranked_momentum_factor)
# model.add_factor(regression_momentum_factor)
# model.add_factor(mom_season_short_monthly_factor)
# model.add_factor(mom_season_short_daily_factor)
# model.add_factor(asset_growth_factor)

stats = model.wfo(returns_data,
                  train_interval=relativedelta(years=5), anchored=False,  # interval parameters
                  start_date=datetime(2013, 1, 1), end_date=datetime(2019, 1, 1),
                  k_pct=0.2, long_pct=0.5,  # weight parameters,
                  # reg_alpha=0.5, reg_lambda=0.5,  # regularization parameters
                  )

stats.print_statistics_report()
stats.get_html()
stats.save(get_results_dir() / 'wfo_stats.pkl')
model.save(get_results_dir() / 'current_model.pkl')
