"""Core functionalities for pdexplore."""

from .numeric import run_numeric_exploration_pipeline
from .general import general_exploration
from .util import (
    get_output_fpath,
    OUTPUT_F,
    set_printing_to_screen,
    set_output_f,
    custom_print as print,
    # comment,
    # bold,
)


def _explore_series(series, label=None):
    general_exploration(series, label=label)
    run_numeric_exploration_pipeline(series)
    # _explore_numeric_series(series, count)


def explore_series(series, label=None, output_path=None, silent=False):
    """Perform basic data exploration of a series and prints the results.

    Explorations performed:

    Parameters
    ----------
    series : pandas.Series, numpy.Ndarray
        The pandas series or numpy array to explore.
    label : str, optional
        The label of the explored series.
    output_path : str, optional
        A path for an output file, where exploration output is written to in
        addition to standard output. If a path to a file is given, it is
        created (if missing); if a path to a directory is given, an aptly named
        file is created in it. If no path is given, output is only writted to
        the standard output (usually the screen).
    silent : bool, optional
        If set to True, no output is printed to screen. Defaults to False.
    """
    set_printing_to_screen(not silent)
    if output_path is not None:
        output_path = get_output_fpath(output_path, laebl=label)
        with open(output_path, 'wt+') as out_f:
            set_output_f(out_f)
            _explore_series(
                df=series,
                label=label,
            )
            set_output_f(None)
    else:
        if OUTPUT_F is None and silent:
            raise ValueError(
                "No output path is given but function is set to"
                " silent. That's silly. Terminating."
            )
        _explore_series(
            df=series,
            label=label,
        )


def _explore_df(df, skip_lbl=None, skip_cond=None):
    if skip_lbl is None:
        skip_lbl = []
    if skip_cond is None:
        skip_cond = []
    print("Starting to explore a dataframe with pdexplore.")
    print(f"The dataframe contains {len(df.columns)} columns.")
    print(f"The dataframe contains {len(df)} rows.")
    skipped = 0
    if callable(skip_cond):
        skip_cond = [skip_cond]
    for col_lbl in df.columns:
        col = df[col_lbl]
        try:
            if col_lbl in skip_lbl:
                print(f"Skipping exploration of column {col_lbl}!")
                skipped += 1
                continue
        except TypeError:
            pass
        try:
            if not any([cond(col) for cond in skip_cond]):
                _explore_series(
                    series=col,
                    label=col_lbl,
                )
            else:
                print(f"Skipping exploration of column {col_lbl}!")
                skipped += 1
        except TypeError:
            _explore_series(
                series=col,
                label=col_lbl,
            )
    if skipped > 0:
        print(f"Explortaion of {skipped} columns was skipped.")


def explore(df, output_path=None, skip_lbl=None, skip_cond=None, silent=False):
    """Perform basic data exploration of a dataframe and prints the results.

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas dataframe to explore.
    output_path : str, optional
        A path for an output file, where exploration output is written to in
        addition to standard output. If a path to a file is given, it is
        created (if missing); if a path to a directory is given, an aptly named
        file is created in it. If no path is given, output is only writted to
        the standard output (usually the screen).
    skip_lbl : list of str, optional
        A list of labels of columns to skip exploration for.
    skip_cond : callable or array of callables, optional
        If given, then for each series each condition callable is called with
        the series as the sole argument, and if any condition callable returns
        True, series exploration is skipped.
    silent : bool, optional
        If set to True, no output is printed to screen. Defaults to False.
    """
    set_printing_to_screen(not silent)
    if output_path is not None:
        output_path = get_output_fpath(output_path)
        with open(output_path, 'wt+') as out_f:
            set_output_f(out_f)
            _explore_df(df=df, skip_lbl=skip_lbl, skip_cond=skip_cond)
    else:
        if OUTPUT_F is None and silent:
            raise ValueError(
                "No output path is given but function is set to"
                " silent. That's silly. Terminating."
            )
        _explore_df(df, skip_lbl=skip_lbl, skip_cond=skip_cond)
