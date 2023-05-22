import polars as pl


def pct_change(exclude_col=[]) -> pl.Expr:
    """
    The polars expression to compute the percent change of a dataframe.

    :param exclude_col: A list of columns to exclude in the calculation. These columns will still be returned in the
                        computation, but as they were originally.

    :return: A polars expression to compute the percent change.
    """
    return pl.col(exclude_col), pl.all().exclude(exclude_col).map(lambda s: ((s / s.shift(1)) - 1))
