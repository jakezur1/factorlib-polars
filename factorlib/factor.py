import polars as pl
import pandas as pd
import numpy as np
import warnings

from .utils.helpers import resample
from .utils.warnings import ParameterOverride


class Factor:
    """
    This class represents a factor for the FactorModel class. It is responsible for formatting user data, and
    transforming it for use in a `FactorModel`.
    """

    def __init__(self, name: str = None, data: pl.DataFrame = None,
                 current_interval: str = '1d', general_factor: bool = False,
                 desired_interval: str = None, tickers: [str] = None,
                 transforms=None):
        """
        :param name: The name of the factor. If multiple factors are included in this object, name it by a
                     general category that separates this dataset from other factor objects.
        :param data: The polars dataframe that will serve as the data for this factor. The dataframe must have a date
                     column called `date_index` which must have type date or datetime. The dataframe must also contain
                     a ticker column called `ticker`, or be a `general_factor`.
        :param current_interval: The current frequency of the time series. This is the interval between each row of the
                                 date column.
        :param desired_interval: The interval that the factor will be resampled to. This is parameter exists so that
                                 factorlib can ensure that your factors have the same frequency that you wish to
                                 trade on in backtesting.
        :param general_factor: True if this factor is a general factor for all tickers. If so, this factor must have
                               a ticker column.
        :param tickers: A list of tickers. This will only be used if the factor is a `general_factor`. If so, it will
                        create a tickers column by performing cartesian multiplication between the dates and the
                        given list of tickers.
        :param transforms: A list of functions or functors that will perform transforms on the data. These functions
                           must only take in a polars dataframe or series. If the desired transform require more
                           parameters than just the data to operate on, create a functor class and pass function
                           parameters as member variables of the functor class. See factorlib.transforms for examples.
        """

        assert (data.columns.__contains__('date_index')), 'Your factor must contain a date column called called ' \
                                                          '`date_index` that will serve as the index of the ' \
                                                          'dataframe.'

        if general_factor:
            assert (tickers is not None), 'All general factors mut be supplied with the `tickers` parameter. ' \
                                          'Ideally, this should be fully comprehensive list of all tickers that ' \
                                          'you plan to use in the factor model to which this factor will be added.'

        if tickers is not None and general_factor is False:
            warnings.warn(f'You have passed a `tickers` list for the factor named {name}, but this factor is '
                          'not a `general_factor` and so `tickers` will not being used.', category=ParameterOverride)

        if transforms is None:
            transforms = []

        self.name = name
        self.interval = current_interval
        data = data.lazy().sort(by='date_index').set_sorted('date_index').collect(streaming=True)
        try:
            data = data.with_columns(
                pl.all().exclude(['date_index', 'ticker']).cast(pl.Float64)
            )
            data = data.with_columns(
                pl.col('date_index').cast(pl.Datetime)
            )
        except Exception:
            pass

        if not general_factor:
            data = data.drop_nulls(subset='date_index')

        if desired_interval is not None:
            self.data = resample(data=data, current_interval=self.interval, desired_interval=desired_interval,
                                 melted=(not general_factor))
            self.interval = desired_interval
        else:
            self.data = data

        self.transforms = transforms

        self.data.sort('date_index')
        dates = data.select(pl.col('date_index').drop_nulls()).to_series().to_list()
        self.start = dates[0]
        self.end = dates[-1]

        if general_factor:
            tickers_df = pl.DataFrame({'ticker': tickers})
            self.data = (
                self.data.lazy()
                .select(
                    pl.col(['date_index']),
                    pl.all().exclude(['date_index', 'ticker']).suffix('_' + self.name)
                )
                .join(tickers_df.lazy(), how='cross')
                .collect(streaming=True)
            )
        for transform in transforms:
            self.data = transform(self.data)
