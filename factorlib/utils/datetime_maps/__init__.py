import json
import os
from factorlib.utils.system import get_datetime_maps_dir


POLARS_DATETIMES_JSON = get_datetime_maps_dir() / 'polars_datetimes.json'
POLARS_TO_PANDAS_JSON = get_datetime_maps_dir() / 'polars_to_pandas.json'
TIMEDELTAS_JSON = get_datetime_maps_dir() / 'time_delta_intervals.json'

with open(POLARS_DATETIMES_JSON) as polars_datetimes:
    pl_time_intervals = json.load(polars_datetimes)

with open(POLARS_TO_PANDAS_JSON) as p_to_p:
    polars_to_pandas = json.load(p_to_p)

with open(TIMEDELTAS_JSON) as time_deltas:
    time_delta_intervals = json.load(time_deltas)
