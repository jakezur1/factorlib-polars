import polars as pl


def pct_change(exclude_col=[]) -> pl.Expr:
    return pl.col(exclude_col), pl.all().exclude(exclude_col).map(lambda s: ((s / s.shift(1)) - 1))
