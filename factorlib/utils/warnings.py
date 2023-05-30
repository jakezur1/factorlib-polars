import warnings


class ParameterOverride(UserWarning):
    """
    A warning class for when parameters are passed to a factor-lib function but are never used, or when a parameter is
    passed but overridden by a second parameter.
    """
    pass
