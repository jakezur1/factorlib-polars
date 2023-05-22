import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from factorlib.utils.datetime_maps import polars_to_pandas, pl_time_intervals


def resample(data: pl.DataFrame, interval: str, melted=True):
    """
    Resample a polars dataframe to the given `interval`.

    """
    resampling_technique = _up_or_downsample(data=data, model_interval=interval)
    if resampling_technique == 'downsample':
        if melted:  # include 'by' parameter with ticker column
            data = (
                data.sort('date_index')
                .set_sorted('date_index')
                .groupby_dynamic('date_index', every=interval, by='ticker')
                .agg(pl.all().exclude('date_index').first())
            )
        else:
            data = (
                data.sort('date_index')
                .set_sorted('date_index')
                .groupby_dynamic('date_index', every=interval)
                .agg(pl.all().exclude('date_index').first())
            )
    elif resampling_technique == 'upsample':
        if melted:
            data = (
                data.sort('date_index')
                .set_sorted('date_index')
                .upsample(time_column='date_index', every=interval, by='ticker')
                .select(pl.all().forward_fill())
            )
        else:
            data = (
                data.sort('date_index')
                .set_sorted('date_index')
                .upsample(time_column='date_index', every=interval)
                .select(pl.all().forward_fill())
            )
    else:  # intervals are equal
        data = data
    data.replace('date_index', data.select(pl.col('date_index').cast(pl.Datetime)).to_series())
    return data


def offset_datetime(date: datetime, interval: str, sign=1):
    interval = polars_to_pandas[interval]
    if interval == 'D':
        date += sign * pd.DateOffset(days=1)
    elif interval == 'W':
        date += sign * pd.DateOffset(days=7)
    elif interval == 'MS':
        date += sign * pd.DateOffset(months=1)
    elif interval == 'Y':
        date += sign * pd.DateOffset(years=1)
    return date


def shift_by_time_step(time: str, returns: pl.DataFrame):
    value = time.split('t+')
    assert (len(value) > 0), 'pred_time must be formatted as `t+{time_steps_ahead}`, where time_steps_ahead is a ' \
                             'positive integer representing the number of time steps to predict ahead'
    shift = value[1]
    returns = (
        returns.lazy()
        .select(
            pl.col('date_index'),
            pl.all().exclude('date_index').shift(-1 * int(shift))
        )
        .collect(streaming=True)  # shift returns back
    )
    return returns


def align_by_date_index(df1: pd.DataFrame, df2: pd.DataFrame):
    df1_dates = df1.select(pl.col('date_index'))
    df2_dates = df2.select(pl.col('date_index'))
    if df1_dates.item(0, 0) > df2_dates.item(0, 0):
        start = df1_dates.item(0, 0)
    else:
        start = df2_dates.item(0, 0)

    if df1_dates.item(-1, 0) < df2_dates.item(-1, 0):
        end = df1_dates.item(-1, 0)
    else:
        end = df2_dates.item(-1, 0)

    df1 = (
        df1.lazy()
        .with_columns(pl.col("date_index"))
        .filter(pl.col("date_index").is_between(start, end))
        .collect()
    )
    df2 = (
        df2.lazy()
        .with_columns(pl.col("date_index"))
        .filter(pl.col("date_index").is_between(start, end))
        .collect()
    )

    return df1, df2


def clean_data(X: pl.DataFrame, y: pl.DataFrame, col_thresh=0.5):
    X = (
        X.lazy()
        .join(y.lazy(), on=['date_index', 'ticker'], how='outer')
        .collect(streaming=True)
    )
    X = X.drop_nulls(subset=['returns'])  # only look for NaNs in returns, otherwise keep NaNs
    y = (
        X.lazy().select(
            pl.col('date_index'),
            pl.col('ticker'),
            pl.col('returns')
        )
        .collect(streaming=True)
    )
    X = (
        X.lazy()
        .select(
            pl.all().exclude('returns')
        )
        .collect(streaming=True)
    )
    inf_dict = {
        np.inf: 0,
        -np.inf: 0
    }
    X = (
        X.lazy()
        .select(
            pl.col('date_index'),
            pl.col('ticker'),
            pl.all().exclude(['date_index', 'ticker']).map_dict(inf_dict, default=pl.first())
        )
        .collect(streaming=True)
    )
    # X.replace([np.inf, -np.inf], 0, inplace=True)  make different in polars
    return X, y


def get_start_convention(date: datetime, interval: str):
    interval = polars_to_pandas[interval]
    temp_df = pd.DataFrame(index=[date])
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df.index = temp_df.index.tz_localize(None)
    temp_df = temp_df.resample(interval, convention='start').ffill()
    end_convention = temp_df.index[0]
    return end_convention


def _up_or_downsample(data: pl.DataFrame, model_interval: str):
    unique_dates = (
        data.lazy()
        .select(
            pl.col('date_index').unique()
        )
        .sort('date_index')
        .set_sorted('date_index')
        .collect(streaming=True)
    )
    first_date = unique_dates.item(0, 0)
    second_date = unique_dates.item(1, 0)
    num_min = (second_date - first_date).total_seconds() / 60

    if num_min < pl_time_intervals[model_interval]:
        return 'downsample'
    elif num_min > pl_time_intervals[model_interval]:
        return 'upsample'
    else:
        return 'don\'t waste time!'
