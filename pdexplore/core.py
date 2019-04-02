"""Core functionalities for pdexplore."""


from .util import (
    get_output_fpath,
)


def _explore_series(series, label=None):
    pass


def explore_series(series, label=None, output_path=None):
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
    """

    pass



def explore(df, output_path=None):
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
    """
    if output_path is not None:
        output_path = get_output_fpath(output_path)
    with open(output_path, 'wt+') as
    for col_lbl in df.columns:
        explore_series(df[col_lbl])
