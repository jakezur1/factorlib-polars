import warnings
import inspect
import pandas as pd
from pathlib import Path
import sys


def is_frozen():
    return getattr(sys, 'frozen', False)


def get_root_dir() -> Path:
    if is_frozen():
        # we are running in a bundle
        return Path(sys.executable).parent

    # we are running in a normal Python environment
    return Path(__file__).resolve().parent.parent.parent


def get_datetime_maps_dir() -> Path:
    return get_root_dir() / 'factorlib' / 'utils' / 'datetime_maps'


def get_results_dir() -> Path:
    return get_root_dir() / 'results'


def get_data_dir() -> Path:
    return get_root_dir() / 'data'


def silence_warnings():
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    warnings.filterwarnings(action='ignore', message='An input array is constant; '
                                                     'the correlation coefficient is not defined.')
    warnings.filterwarnings(action='ignore', message='ntree_limit is deprecated, use `iteration_range` or '
                                                     'model slicing instead.')


def _get_defining_class(meth) -> any:
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
    if inspect.isfunction(meth):
        return getattr(inspect.getmodule(meth),
                       meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0],
                       None)
    return None
