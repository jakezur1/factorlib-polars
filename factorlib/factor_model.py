from typing import Literal
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import polars as pl
import pickle
import time
import shap
import re
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.ensemble import *
from xgboost import XGBRegressor

import factorlib
from .factor import Factor
from .utils.helpers import resample, shift_by_time_step, align_by_date_index, offset_datetime, clean_data, \
    get_start_convention
from .utils.system import silence_warnings
from .utils.datetime_maps import pl_time_intervals, polars_to_pandas

silence_warnings()

shap.initjs()

ModelType = Literal['hgb', 'gbr', 'adaboost', 'rf', 'et', 'linear', 'voting', 'xgb']


class FactorModel:
    """
    This class represents an interface for an alpha factor model system. It allows users to add factors to the model
    and perform a walk-forward optimization for backtesting.
    """

    def __init__(self, tickers: [str] = None, interval: str = '1d'):
        """
        :param tickers: The list of tickers that the model will choose to trade from. You can add factors to this model
                        even if that factor has more or less tickers than this parameter.
        :param interval: The interval that all data will be resampled on. All factors will be resampled to this
                         interval before being added to the model.
        """

        assert tickers is not None, 'You must provide a valid list of tickers. It cannot be None.'

        self.tickers = tickers
        self.interval = interval
        self.factors = pl.DataFrame()
        self.earliest_start = None
        self.latest_end = None
        self.model = None

    def add_factor(self, factor: Factor | list[Factor], replace: bool = False) -> None:
        """
        Adds a factor to the model's factors dataframe.

        :param factor: A Factor object or list of Factor objects that represents the factor(s) to add to the model.
        :param replace: If replace=True, the factor being added will replace the existing factor with the same name.
                        This param is most commonly used when loading a past model that has been saved, and modifying a
                        factor that had been previously added.
        """

        if type(factor) == Factor:
            factor = [factor]

        for curr_factor in factor:
            curr_factor.data = resample(data=curr_factor.data, current_interval=curr_factor.interval,
                                        desired_interval=self.interval)

            if self.factors.is_empty():
                self.factors = curr_factor.data
            else:
                self.factors = (
                    self.factors.lazy()
                    .join(curr_factor.data.lazy(), on=['date_index', 'ticker'], how='outer')
                    .collect(streaming=True)
                )

            if replace:
                regex = re.compile(".*_right")
                cols_to_keep = list(filter(regex.match, self.factors.columns))
                cols_to_exclude = [col[:-6] for col in cols_to_keep]
                self.factors = (
                    self.factors.lazy()
                    .select(
                        pl.all().exclude(cols_to_exclude),
                    )
                    .collect(streaming=True)
                )
                rename_map = dict(zip(cols_to_keep, cols_to_exclude))
                self.factors.rename(rename_map)

            self.factors = self.factors.sort(['date_index', 'ticker'])

            if self.earliest_start is None:
                self.earliest_start = curr_factor.start
            else:
                if self.earliest_start < curr_factor.start:
                    self.earliest_start = curr_factor.start
            if self.latest_end is None:
                self.latest_end = curr_factor.end
            else:
                if self.latest_end > curr_factor.end:
                    self.latest_end = curr_factor.end

    def predict(self, factors: pl.DataFrame) -> np.ndarray:
        """Given a polars dataframe of factors, predict the next intervals returns."""
        return self.model.predict(factors)

    def wfo(self, returns: pl.DataFrame,
            train_interval: timedelta | relativedelta,
            start_date: datetime = None,
            end_date: datetime = None,
            anchored=True,
            k_pct=0.2,
            long_pct=0.5,
            long_only=False,
            short_only=False,
            pred_time='t+1',
            train_freq=None,
            candidates=None, **kwargs):
        """
        Perform walk-forward optimization for backtesting.

        Required: The interval of returns must match the interval of the model.

        :param returns: A polars dataframe of all the returns for all of this model's `tickers` at each interval
        :param train_interval: The total interval with which to train the model. If anchor=True, this is the initial
                               training interval on the first iteration. If anchor=False, this is the rolling window
                               with which to train each month.
        :param start_date: The date to begin backtesting. This date must be greater than the latest start date of
                           all factors in this model.
        :param end_date: The date to end backtesting. This date must be less than the earliest end date of all factors
                         in this model.
        :param anchored: If anchored=True, the starting day of training does not change as the backtesting proceeds.
                         This means the training interval is getting longer as the model continues to trade.
                         If anchored=False, the training interval moves through the data set as the model trades,
                         and the interval of training is constant and equal to `train_interval`.
        :param k_pct: The top and/or bottom k percent of stocks to trade.
        :param long_pct: The percentage of stocks to long. This is only applicable if `long_only` and `short_only`
                         are both False.
        :param long_only: Equivalent to setting long_pct=1.0.

        :param short_only: Equivalent to setting long_pct=0.0.
        :param pred_time: Number of time steps ahead to predict. Formatted as `t+{time steps}`.
        :param train_freq: The frequency with which to retrain the interval.
        :param candidates: TODO: Implement candidates as a dict that holes yearly SP500 candidates.
        :param kwargs: Additional key word arguments to be given to the XGBoostRegressor. These can be regularization
                       parameters, training parameters, or any other parameter of XGBoostRegressor.
        :return: A statistics object containing all the information gathered while backtesting. See Statistics
                 in statistics.py for more details.
        """

        assert (self.interval == '1d' or self.interval == '1w' or self.interval == '1mo' or self.interval == '1y'), \
            'Walk forward optimization currently only supports daily, weekly, monthly, or yearly intervals'

        if train_freq is not None:
            print('the train_freq parameter does not have stable implementation yet. '
                  'Defaulting to monthly (`M`) training.')
            train_freq = 'M'

        assert (not (long_only and short_only)), '`long_only` and `short_only` cannot both be True'

        if start_date is not None:
            assert (start_date > self.earliest_start), 'The model\'s start_date must have data for all factors.'
        else:
            start_date = self.earliest_start

        if end_date is not None:
            assert (end_date < self.latest_end), '`end_date` must be before latest end date'
        else:
            end_date = self.latest_end

        print('Starting Walk-Forward Optimization from', start_date, 'to', end_date, 'with a',
              train_interval.years, 'year training interval')

        # cast returns date_index to datetime
        returns = pl.from_pandas(
            returns.to_pandas().set_index('date_index').resample(polars_to_pandas[self.interval],
                                                                 convention='start').asfreq().fillna(0).reset_index())

        returns.replace('date_index', returns.select(pl.col('date_index').cast(pl.Datetime)).to_series())
        # shift returns back by 'time' time steps
        shifted_returns = shift_by_time_step(pred_time, returns)

        # align factor dates to be at the latest first date and earliest last date
        _, shifted_returns = align_by_date_index(self.factors, shifted_returns)

        # stack the returns and sort on date_index and ticker
        start = time.time()
        melted_returns = (
            shifted_returns.lazy()
            .melt(id_vars=['date_index'])
            .select(
                pl.col('date_index'),
                pl.col('variable').alias('ticker'),
                pl.col('value').alias('returns')
            )
            .sort(by=['date_index', 'ticker'])
            .collect(streaming=True)
        )

        # sort factors on date_index and ticker
        self.factors = self.factors.lazy().sort(by=['date_index', 'ticker']).collect(streaming=True)
        # set the frequency of training
        frequency = None
        if train_freq is None:
            if pl_time_intervals[self.interval] <= pl_time_intervals['1mo']:
                frequency = '1mo'
            else:
                frequency = self.interval
        else:
            frequency = train_freq

        training_start = start_date
        training_end = start_date + train_interval
        assert training_end < end_date, 'Training interval exceeds the total amount of data provided. Choose a ' \
                                        'shorter `train_interval` or an earlier start date.'
        self.model = XGBRegressor(n_jobs=-1, tree_method='hist', random_state=42, **kwargs)

        # perform walk forest optimization on factors data and record expected returns
        # at each time step

        # initialize statistics data
        expected_returns = pd.DataFrame()
        expected_returns_index = []
        training_spearman = pd.Series(dtype=object)

        # using for loop for tqdm progress bar
        loop_start = training_end
        loop_end = offset_datetime(end_date, interval=frequency, sign=-1)
        loop_range = pd.date_range(loop_start, loop_end, freq=polars_to_pandas[frequency])

        shap_values = []

        for index, date in enumerate(tqdm(loop_range)):
            start = time.time()
            X_train = (
                self.factors.lazy()
                .filter(
                    pl.col('date_index').is_between(training_start, training_end, closed="left")
                )
                .collect(streaming=True)
            )
            y_train = (
                melted_returns.lazy()
                .filter(
                    pl.col('date_index').is_between(training_start, training_end, closed="left")
                )
                .collect(streaming=True)
            )
            X_train, y_train = pl.align_frames(X_train, y_train, on=['date_index', 'ticker'])
            X_train, y_train = clean_data(X_train, y_train)
            X_train, y_train = pl.align_frames(X_train, y_train, on=['date_index', 'ticker'])
            start = time.time()

            X_train_unindexed = X_train.drop(['date_index', 'ticker'])
            y_train_unindexed = y_train.drop(['date_index', 'ticker'])
            self.model.fit(X_train_unindexed, y_train_unindexed)

            if index == (len(loop_range) - 1):
                explainer = shap.Explainer(self.model)
                shap_values = explainer(X_train_unindexed)

            # print('Took', time.time() - start, 'seconds to fit model')

            if index != 0:
                training_predictions = self.predict(X_train_unindexed)
                spear_index = X_train.select(
                    pl.col('date_index')
                ).sort('date_index').to_series()
                training_predictions = pd.DataFrame(training_predictions,
                                                    index=spear_index)
                training_predictions['ticker'] = pd.Series(X_train.select(pl.col('ticker')).to_series().to_list(),
                                                           index=spear_index)
                training_predictions.rename(columns={'0': 'returns'})
                training_predictions = training_predictions.set_index([training_predictions.index, 'ticker'])
                training_predictions = training_predictions.unstack(level=1).droplevel(0, axis=1)
                returns_for_spearman = (
                    returns.lazy()
                    .filter(pl.col('date_index').is_between(training_predictions.index[0],
                                                            training_predictions.index[-1]))
                    .collect(streaming=True, no_optimization=True)
                )
                returns_for_spearman = returns_for_spearman.to_pandas().set_index('date_index')
                spearman = returns_for_spearman.corrwith(training_predictions, method='spearman', axis=1).mean()
                spearman = pd.Series(spearman, index=[training_predictions.index[-1]])
                training_spearman = pd.concat([training_spearman, spearman])

            # get predictions
            # this is our OOS sample test (that's one timestep ahead)
            start = time.time()
            pred_start = training_end
            pred_start = get_start_convention(pred_start, interval=self.interval)
            pred_end = offset_datetime(training_end, interval=frequency)

            curr_predictions = pd.DataFrame()
            for ticker in self.tickers:
                try:
                    indexed_prediction_data = (
                        self.factors.lazy()
                        .filter(
                            ((pl.col('date_index').is_between(pred_start, pred_end, closed="left")) &
                             (pl.col('ticker') == str(ticker)))
                        )
                        .sort('date_index')
                        .collect(streaming=True)
                    )
                    prediction_data = indexed_prediction_data.drop(['date_index', 'ticker'])

                    if len(curr_predictions) != len(prediction_data):
                        pass
                    curr_predictions[ticker] = self.model.predict(prediction_data).flatten()
                    curr_index = (
                        indexed_prediction_data.lazy()
                        .select(
                            pl.col('date_index').unique()
                        )
                        .collect(streaming=True)
                        .to_series()
                        .to_list()
                    )
                except pl.exceptions.ColumnNotFoundError:
                    curr_predictions[ticker] = [np.nan] * len(pd.date_range(pred_start, pred_end,
                                                                            freq=polars_to_pandas[self.interval]))
                    curr_index = (
                        indexed_prediction_data.lazy()
                        .select(
                            pl.col('date_index').unique()
                        )
                        .collect(streaming=True)
                        .to_series()
                        .to_list()
                    )

            expected_returns = pd.concat([expected_returns, curr_predictions], axis=0)
            expected_returns_index.extend(curr_index)

            # calculate new intervals to train
            if not anchored:
                training_start = offset_datetime(training_start, interval=frequency)
                training_start = get_start_convention(training_start, self.interval)

            training_end = offset_datetime(training_end, interval=frequency)
            training_end = get_start_convention(training_end, self.interval)

        expected_returns_index = np.array(expected_returns_index, dtype='datetime64[D]')
        expected_returns_index = np.unique(expected_returns_index)
        expected_returns.index = expected_returns_index
        expected_returns = expected_returns.resample(polars_to_pandas[self.interval],
                                                     convention='start').asfreq().fillna(0)

        print('Expected returns: ')
        print(f'{expected_returns}\n')

        # get positions
        positions = expected_returns.apply(self._get_positions, axis=1,
                                           k_pct=k_pct, long_pct=long_pct,
                                           long_only=long_only, short_only=short_only)

        positions = positions.resample(polars_to_pandas[self.interval], convention='start').asfreq().fillna(0)
        positions = positions.shift(-1).dropna()
        positions = pl.from_pandas(positions)
        positions = (
            positions.lazy()
            .with_columns(
                pl.Series('date_index', expected_returns_index.tolist()[:-1]).cast(pl.Datetime)
            )
            .collect(streaming=True)
        )

        # align positions and returns
        positions, returns = align_by_date_index(positions, returns)
        # calculate back tested returns
        returns_index = returns.select(pl.col('date_index'))
        positions = positions.drop('date_index')
        returns_unindexed = returns.drop('date_index')

        returns_per_stock = returns_unindexed * positions
        portfolio_returns = returns_per_stock.sum(axis=1)
        portfolio_returns = portfolio_returns.rename('returns')
        returns_index = returns_index.to_pandas().set_index('date_index') \
            .resample(polars_to_pandas[self.interval], convention='start').asfreq().fillna(0).reset_index()
        returns_index = pl.from_pandas(returns_index).to_series()
        portfolio_returns = portfolio_returns.to_frame().with_columns(returns_index)

        portfolio_returns = portfolio_returns.with_columns(returns_index)
        portfolio_returns = portfolio_returns.to_pandas()
        portfolio_returns = portfolio_returns.set_index('date_index')
        returns = returns.to_pandas()
        returns = returns.set_index('date_index')
        positions = positions.with_columns(returns_index)
        positions = positions.to_pandas().set_index('date_index')

        # importing here to avoid circular import
        from .statistics import Statistics
        return Statistics(portfolio_returns, self, predicted_returns=expected_returns, stock_returns=returns,
                          position_weights=positions, training_spearman=training_spearman, shap_values=shap_values)

    def save(self, path: str | Path) -> None:
        """
        Save the current FactorModel object to disk at the given path.
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str | Path) -> None:
        """
        Load a FactorModel that has been previously saved to disk.
        """
        with open(path, 'rb') as f:
            loaded_model = pickle.load(f)
        self.__dict__.update(loaded_model.__dict__)

    def _get_positions(self, row: pd.Series,
                       k_pct: float = 0.2,
                       long_pct: float = 0.5,
                       long_only: bool = False,
                       short_only: bool = False) -> pd.Series:
        """
        Given a row of returns and a percentage of stocks to long and short, return a row of positions of equal
        long and short positions, with weights equal to long_pct and 1 - long_pct respectively.

        :param row: Intended to be used with pandas' .apply function. The first argument is reserved for the series
                    that will be operated on.
        :param k_pct: The top and/or bottom k percent of stocks to long and/or short.
        :param long_pct: The percentage of stocks to long. This is only applicable if `long_only` and `short_only`
                         are both False.
        :param long_only: Equivalent to setting long_pct=1.0.
        :param short_only: Equivalent to setting long_pct=0.0.

        :return: A pandas series of N tickers representing the position weight of each ticker
        """

        num_na = int(row.isna().sum())
        indices = np.argsort(row)[:-num_na]  # sorted in ascending order
        if num_na == 0:
            indices = np.argsort(row)  # sorted in ascending order
        k = int(np.floor(len(indices) * k_pct))
        bottomk = indices[:k]
        topk = indices[-k + 1:]
        positions = [0] * len(row)

        if long_only:
            long_pct = 1.0
        elif short_only:
            long_pct = 0.0

        for i in topk:
            positions[i] = round((1 / k) * long_pct, 3)
        for i in bottomk:
            positions[i] = round((-1 / k) * (1 - long_pct), 3)
        return pd.Series(positions, index=self.tickers)

    def _get_model(self, model, **kwargs):
        if model == 'hgbm':
            self.model = HistGradientBoostingRegressor(**kwargs)
        elif model == 'gbr':
            self.model = GradientBoostingRegressor(**kwargs)
        elif model == 'adaboost':
            self.model = AdaBoostRegressor(**kwargs)
        elif model == 'rf':
            self.model = RandomForestRegressor(**kwargs)
        elif model == 'et':
            self.model = ExtraTreesRegressor(**kwargs)
        elif model == 'xgb':
            self.model = XGBRegressor(**kwargs)
        return self.model
