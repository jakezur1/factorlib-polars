import json
import polars as pl
import pandas as pd
import pickle as pkl
from datetime import datetime
from dateutil.relativedelta import relativedelta

from factorlib.factor import Factor
from factorlib.factor_model import FactorModel
from factorlib.utils.system import get_data_dir, get_results_dir
from factorlib.transforms.expr_shortcuts import calculate_returns

MODEL_INTERVAL = '1d'

raw_data_dir = get_data_dir() / "raw"

print('Reading in Stock Data...')
# tradeable_tickers = pd.read_csv(raw_data_dir / 'tickers_to_trade.csv')['ticker'].tolist()
tradeable_tickers = ["OPRA", "SMCI", "LMB", "MLTX", "YPF", "CABA", "WEAV", "ELF", "EDN", "ACLS", "INTT", "ETNB", "CIR",
                     "RCL", "NVDA", "DAKT", "TCMD", "DMAC", "IMVT", "MMMB", "ENIC", "WFRD", "IPDN", "STRL", "RMBS",
                     "MOD", "NGL", "TDW", "TAYD", "VIST", "EXTR", "SYM", "CCL", "CMT", "CBAY", "TGLS", "BELFB", "VECT",
                     "AEHR", "CUK", "UFPT", "AUGX", "ISEE", "TAST", "COCO", "VRT", "BWMN", "ONCY", "BLDR", "ODC",
                     "ATEC", "NVTS", "RMTI", "AVDL", "IRS", "DFH", "CVRX", "PEN", "TGS", "GRBK", "PLPC", "SKYW", "USAP",
                     "ACVA", "RETA", "BTBT", "TROO", "POWL", "PPSI", "FTI", "DO", "SGML", "GGAL", "PCYG", "NETI",
                     "TRHC", "ARDX", "STVN", "NFLX", "INTA", "MORF", "RXST", "HGBL", "GE", "BZH", "BBAR", "PESI", "RIG",
                     "NU", "TK", "JBL", "ERO", "SMHI", "IRON", "EVLV", "GENI", "ELTK", "ENVX", "META", "NCLH"]
print("Universe of Tickers: ", len(tradeable_tickers), " Total")

returns_data = pl.scan_csv(raw_data_dir / 'small_universe_returns.csv', try_parse_dates=True).collect(streaming=True)
print('Creating Price Features...')
price_data_dir = get_data_dir() / 'price'

# technicals
technicals = (
    pl.scan_csv(price_data_dir / 'technicals.csv', try_parse_dates=True)
    .collect(streaming=True)
)

technicals_factor = Factor(name='techs', data=technicals, current_interval='1d')

# candle sticks
candle_sticks = (
    pl.scan_csv(price_data_dir / 'candle_sticks.csv', try_parse_dates=True)
    .collect(streaming=True)
)

candle_sticks_factor = Factor(name='c_sticks', data=candle_sticks, current_interval='1d')

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

categorical_fundamentals = pl.scan_csv(fundamental_data_dir / 'subindustry_fundamentals.csv',
                                       try_parse_dates=True).collect(streaming=True)
cat_fundamentals_factor = Factor(name='cat_fund', data=categorical_fundamentals,
                                 current_interval=MODEL_INTERVAL)

non_cat_fundamentals = pl.scan_csv(fundamental_data_dir / 'non_cat_fundamentals.csv', try_parse_dates=True).collect(
    streaming=True)
non_cat_funds_factor = Factor(name='non_cat_fund', data=non_cat_fundamentals,
                              current_interval=MODEL_INTERVAL)
#
# p/e analysis
# pe_analysis = pl.scan_csv(fundamental_data_dir / 'pe_analysis.csv', try_parse_dates=True).collect(streaming=True)
# pe_factor = Factor(name='pe', data=pe_analysis,
#                    current_interval='1mo', desired_interval=MODEL_INTERVAL)
#
# s/p analysis
# sp_analysis = pl.scan_csv(fundamental_data_dir / 'sp_analysis.csv', try_parse_dates=True).collect(streaming=True)
# sp_factor = Factor(name='sp', data=sp_analysis,
#                    current_interval='1mo', desired_interval=MODEL_INTERVAL)
#
#
# current ratio analysis
# curr_ratio_analysis = pl.scan_csv(fundamental_data_dir / 'curr_ratio_analysis.csv', try_parse_dates=True).collect(
#     streaming=True)
# curr_ratio_factor = Factor(name='cr', data=curr_ratio_analysis,
#                            current_interval='1mo', desired_interval=MODEL_INTERVAL)
#
# earnings surprises
# earnings_surprises = pl.scan_csv(fundamental_data_dir / 'earnings_surprises.csv', try_parse_dates=True).collect(
#     streaming=True)
# earnings_surprises_factor = Factor(name='earn_surp', data=earnings_surprises,
#                                    current_interval='1mo', desired_interval=MODEL_INTERVAL)
# div_season
# div_season = pl.scan_csv(fundamental_data_dir / 'div_season.csv', try_parse_dates=True) \
#     .collect(streaming=True)
# div_season_factor = Factor(name='div_season', data=div_season, current_interval='1mo', desired_interval='1d')

# ch_tax
# ch_tax = pl.scan_csv(fundamental_data_dir / 'ch_tax.csv', try_parse_dates=True).collect(streaming=True)
# ch_tax_factor = Factor(name='ch_tax', data=ch_tax, current_interval='1mo', desired_interval='1d')
#
# asset_growth = pl.scan_csv(fundamental_data_dir / 'asset_growth.csv', try_parse_dates=True) \
#     .collect(streaming=True)
# asset_growth_factor = Factor(name='asset_growth', data=asset_growth, current_interval='1mo', desired_interval='1d')

print('Creating Statistical Features...')
statistical_dir = get_data_dir() / 'statistical'

# egarch variance
# egarch_variance = pl.scan_csv(statistical_dir / 'egarch_variance.csv', try_parse_dates=True).collect(
#     streaming=True)
# egarch_variance_factor = Factor(name='egarch', data=egarch_variance,
#                                 current_interval='1d', desired_interval=MODEL_INTERVAL)

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
model = FactorModel(tickers=tradeable_tickers, interval=MODEL_INTERVAL)

model.add_factor(technicals_factor)
model.add_factor(candle_sticks_factor)
model.add_factor(cat_fundamentals_factor)
model.add_factor(non_cat_funds_factor)
# model.add_factor(fff_factor)
# model.add_factor(fundamentals1_factor)
# model.add_factor(div_season_factor)
# model.add_factor(pe_factor)
# model.add_factor(sp_factor)
# model.add_factor(curr_ratio_factor)
# model.add_factor(earnings_surprises_factor)
# model.add_factor(egarch_variance_factor)
# model.add_factor(ch_tax_factor)
# model.add_factor(rsi_factor)
# model.add_factor(ranked_momentum_factor)
# model.add_factor(regression_momentum_factor)
# model.add_factor(mom_season_short_monthly_factor)
# model.add_factor(mom_season_short_daily_factor)
# model.add_factor(asset_growth_factor)

with open(raw_data_dir / 'sp500_candidates.pkl', 'rb') as p:
    candidates = pkl.load(p)
stats = model.wfo(returns_data,
                  train_interval=relativedelta(years=5), anchored=False,  # interval parameters
                  start_date=datetime(2017, 1, 1), end_date=datetime(2023, 1, 1),
                  # candidates=candidates,  # list of candidates to consider each year
                  k_pct=0.2, long_pct=0.5,  # weight parameters,
                  reg_alpha=0.5, reg_lambda=0.5,  # regularization parameters
                  )

stats.print_statistics_report()
stats.get_html()
stats.save(get_results_dir() / 'wfo_stats.pkl')
model.save(get_results_dir() / 'current_model.pkl')
