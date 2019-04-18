"""Utility functions for pdexplore."""

import os
import time

import colored
from colored import stylize


def nice_time_str():
    """Returns current time as a nice string."""
    gt = time.gmtime(time.time())
    return (f'{gt.tm_year}-{gt.tm_mon:02}-{gt.tm_mday:02}_'
            f'{gt.tm_hour:02}-{gt.tm_min:02}-{gt.tm_sec:02}')


def get_output_fpath(output_fpath, label=None):
    if os.path.isdir(output_fpath):
        nice_time = nice_time_str()
        fname = f'pdexplore_{label}_{nice_time}.txt'
        return os.path.join(output_fpath, fname)
    elif os.path.isfile(output_fpath):
        return output_fpath
    else:
        raise ValueError(
            f"Bad value for provided output path: {output_fpath}")


PRINT_TO_SCREEN = True
OUTPUT_F = None


def set_printing_to_screen(val):
    global PRINT_TO_SCREEN
    PRINT_TO_SCREEN = val


def set_output_f(f_obj):
    global OUTPUT_F
    OUTPUT_F = f_obj


def cstr(s, color='black'):
    return f"<text style=color:{color}>{s}</text>"


def custom_print(string, color=None, attr=None):
    if PRINT_TO_SCREEN:
        if color and attr:
            print(stylize(string, colored.fg(color), colored.attr(attr)))
        elif color:
            print(stylize(string, colored.fg(color)))
        elif attr:
            print(stylize(string, colored.attr(attr)))
        else:
            print(string)
    if OUTPUT_F:
        OUTPUT_F.write(string+'\n')


def comment(string):
    custom_print(string, color='grey_50')


def bold(string):
    custom_print(string, attr='bold')
