import numpy as np
import pandas as pd
import polars as pl
from datetime import datetime
from factorlib.utils.datetime_maps import polars_to_pandas, pl_time_intervals
import polars.selectors as cs


def resample(data: pl.DataFrame, current_interval: str, desired_interval: str, melted=True):
    """
    Resample a polars dataframe to the given `interval`.

    :param data: The polars dataframe to resample
    :param current_interval: The current interval of the data before resampling.
    :param desired_interval: The interval with which to resample the dataframe. See /datetime_maps/polars_datetimes.json
                             for valid string datetime intervals.
    :param melted: Whether the dataframe has been melted. If melted=True, the dataframe has already been melted and the
                   dataframe has duplicate dates in `date_index`, but unique combinations of dates and tickers in
                   `date_index` and `ticker`

    :return: The resampled dataframe.
    """

    resampling_technique = _up_or_down_sample(data=data, input_interval=current_interval,
                                              output_interval=desired_interval)
    if resampling_technique == 'downsample':
        if melted:  # include 'by' parameter with ticker column
            data = (
                data.lazy()
                .sort('date_index')
                .set_sorted('date_index')
                .groupby_dynamic('date_index', every=desired_interval, by='ticker')
                .agg(pl.all().exclude('date_index').first())
                .collect(streaming=True)
            )
            if desired_interval == '1d':
                data = _account_for_bdays(data)
        else:
            data = (
                data.lazy()
                .sort('date_index')
                .set_sorted('date_index')
                .groupby_dynamic('date_index', every=desired_interval)
                .agg(pl.all().exclude('date_index').first())
                .collect(streaming=True)
            )
            if desired_interval == '1d':
                data = _account_for_bdays(data)
    elif resampling_technique == 'upsample':
        if melted:
            data = (
                data.sort('date_index')
                .set_sorted('date_index')
                .upsample(time_column='date_index', every=desired_interval, by='ticker', maintain_order=True)
                .select(pl.all().forward_fill())
            )
            if desired_interval == '1d':
                data = _account_for_bdays(data)
        else:
            data = (
                data.sort('date_index')
                .set_sorted('date_index')
                .upsample(time_column='date_index', every=desired_interval)
                .select(pl.all().forward_fill())
            )
            if desired_interval == '1d':
                data = _account_for_bdays(data)
    data.replace('date_index', data.select(pl.col('date_index').cast(pl.Datetime)).to_series())
    return data


def offset_datetime(date: datetime, interval: str, sign=1):
    """
    Offset a datetime object in python by the given `interval` in the direction of `sign`.

    :param date: The datetime to offset.
    :param interval: The string interval with which to offset `date`.
    :param sign: The direction to offset `date`. If sign=1, offset the datetime in the future direction. If sign=-1,
                 offset the datetime back in time.

    :return: The offset datetime.
    """

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
    """
    Shift a polars dataframe by N number of time steps, denoted by `time` as 't+{N}'.

    :param time: A string representation of the time step to shift to. `time` should be formatted as 't+{N}' where `N`
                 is the number of time steps to shift. time='t+1' will shift `returns` one time step ahead. N < 0 will
                 shift returns back in time (N < 0 is not helpful in prediction, will ensure look ahead bias,
                 and should not be used).
    :param returns: The polars dataframe to shift by `time`.

    :return: The shifted polars dataframe.
    """

    value = time.split('t+')
    assert (len(value) > 0), '`time` must be formatted as `t+{N}`, where N is a positive integer representing ' \
                             'the number of time steps to predict ahead'
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


def align_by_date_index(df1: pl.DataFrame, df2: pl.DataFrame):
    """
    Given two polars dataframes that both have a date/datetime column named `date_index`, align the polars dataframes
    such that they only contain data for which they both have dates.

    Example:
        df1 contains data from August 1 - September 1
        df2 contains data from August 15 - September 15

        * alignment occurs *

        df1 contains data from August 15 - September 1
        df2 contains data from August 15 - September 1

    :param df1: A dataframe containing a `date_index` column.
    :param df2: A dataframe containing a `date_index` column.

    :return: The aligned dataframes.
    """
    df1 = df1.sort('date_index')
    df2 = df2.sort('date_index')
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
    """
    Given an X and a y of training data, clean the data such that every value of y exists. X may still have null or nan
    values, but y will be continuous.

    :param X: The features of the training data.
    :param y: The target to train on.
    :param col_thresh: TODO: Deprecated.

    :return: The cleaned X and y.
    """

    X = (
        X.join(y, on=['date_index', 'ticker'], how='inner')
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
        np.inf: 0.0,
        -np.inf: 0.0
    }
    string_and_categorical = X.select(cs.string(include_categorical=True)).columns
    string_only = X.select(cs.string(include_categorical=False)).columns
    categorical_columns = [column for column in string_and_categorical if column not in string_only]
    exclude_columns = categorical_columns + ['date_index', 'ticker']
    X = (
        X.lazy()
        .select(
            pl.col('date_index'),
            pl.col('ticker'),
            pl.col(categorical_columns),
            pl.all().exclude(exclude_columns).map_dict(inf_dict, default=pl.first())
        )
        .collect(streaming=True)
    )
    return X, y


def get_start_convention(date: datetime, interval: str):
    """
    Give a datetime and a string interval. Resample the datetime to be formatted as start convention.
    Example:
        date=datetime(2015, 1, 15)
        interval='1mo'

        * get_start_convention(date=date, interval=interval) *

        output: datetime(2015, 1, 1)

    :param date: The datetime with which to find the start convention.
    :param interval: The pandas string interval to determine the start convention. See the values of
                     /datetime_maps/polars_to_pandas.json, or use intervals from polars_datetimes.json and pass those
                     intervals to polars_to_pandas.json to get valid pandas intervals.

    :return: A datetime following the start convention of `interval`.
    """

    interval = polars_to_pandas[interval]
    temp_df = pd.DataFrame(index=[date])
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df.index = temp_df.index.tz_localize(None)
    temp_df = temp_df.resample(interval, convention='start').ffill()
    end_convention = temp_df.index[0]
    return end_convention


def _up_or_down_sample(data: pl.DataFrame, input_interval, output_interval: str):
    """
    An internal helper function used in resample(data: pl.DataFrame, current_interval: str, desired_interval: str,
    melted=True) to determine if the data requires sampling or down sampling depending on the `resampling_interval`
    and the 'date_index' column of `data`.
    """

    if pl_time_intervals[input_interval] < pl_time_intervals[output_interval]:
        return 'downsample'
    elif pl_time_intervals[input_interval] > pl_time_intervals[output_interval]:
        return 'upsample'
    else:
        return 'Don\'t waste your time'


def _account_for_bdays(data: pl.DataFrame):
    start_date = data.select(pl.col('date_index').min()).item()
    end_date = data.select(pl.col('date_index').max()).item()
    business_days = pd.bdate_range(start=start_date, end=end_date)
    business_days = business_days.tolist()
    business_days = pd.to_datetime(business_days)
    data = data.filter(pl.col('date_index').is_in(business_days))
    return data
