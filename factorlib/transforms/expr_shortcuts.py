import polars as pl


def account_for_split_factor(closes_col: str, split_col: str) -> pl.Expr:
    """
    Given a polars dataframe with a columns named `closes_col` and `split_col`, return a polars expression
    that returns a new column of closes_col * split_col. This should account for the changes in price when stocks
    split over time. The new column will be called `{closes_col}_with_splits`.

    Note: Intended to use on data from Wharton Research Data Services (WRDS).

    :param closes_col: The name of the column in the dataframe that contains the close prices.
    :param split_col: The name of the column in the dataframe that contains the split factors over time.

    :return: The polars expression to compute the true price accounting for split factors.
    """

    return (pl.col(closes_col) * pl.col(split_col)).alias(f'{closes_col}_with_splits')


def calculate_returns(s: pl.Series) -> pl.Series:
    return s / s.shift() - 1
