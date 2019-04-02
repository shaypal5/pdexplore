"""Utility functions for pdexplore."""

import time
import logging


def nice_time_str():
    """Returns current time as a nice string."""
    gt = time.gmtime(time.time())
    return (f'{gt.tm_year}_{gt.tm_mon:02}_{gt.tm_mday:02}_'
            f'{gt.tm_hour:02}:{gt.tm_min:02}:{gt.tm_sec:02}')
