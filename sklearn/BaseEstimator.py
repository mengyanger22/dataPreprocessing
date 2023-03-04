class BaseEstimator:
    @classmethod
    def _get_param_names(cls):
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []
        