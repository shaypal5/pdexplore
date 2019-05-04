"""Base classes for pdexplore."""

import abc
import inspect

from .util import (
    # custom_print as print,
    comment,
    # bold,
)


def precondition(fail_msg=None):
    if fail_msg is None:
        fail_msg = "Unspecified"

    def _prec_decorator(func):
        func.precondition = True
        func.fail_msg = fail_msg
        return func
    # _prec_decorator.precondition = True
    return _prec_decorator


class Exploration(abc.ABC):
    """An exploration method to be applied to a pandas series or dataframe.

    Parameters
    ----------
    """

    @property
    def name(self):
        return str(type(self))

    def _preconditions_hold(self, srs):  # pylint: disable=R0201,W0613
        """Returns True if this method can be applied to a given series."""
        methods = inspect.getmembers(self, predicate=inspect.ismethod)
        for method_tup in methods:
            method_name, method = method_tup
            if hasattr(method, 'precondition'):
                if not method(srs):
                    comment(f"{self.name} skipped. Reason: {method.fail_msg}.")
                    return False
        return True

    @abc.abstractmethod
    def _explore(self, srs):  # pylint: disable=R0201,W0613
        """Explores the given series with this exploration method."""
        raise NotImplementedError

    def apply(self, srs):
        """Applies this method if all preconditions hold."""
        if self._preconditions_hold(srs):
            return self._explore(srs)
        else:
            pass
            # print(f"{self.name} skipped as not all preconditions hold.")

    @staticmethod
    def save(srs, key, val):
        try:
            srs.pdexplore[key] = val
        except AttributeError:
            srs.pdexplore = {}
            srs.pdexplore[key] = val


class SeriesExploration(Exploration):
    pass


class DataframeExploration(Exploration):
    pass
