"""The pdexplore package."""

from .core import (  # noqa: F401
    explore,
    explore_series,
)


from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
