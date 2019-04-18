"""Core functionalities for pdexplore."""

import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.robust import mad

from .util import (
    get_output_fpath,
    OUTPUT_F,
    set_printing_to_screen,
    set_output_f,
    custom_print as print,
    comment,
    bold,
)


DEF_NORMALITY_ALPHA = 0.05


def _explore_numeric_series(series, count, alpha=None):
    if alpha is None:
        alpha = DEF_NORMALITY_ALPHA
    if not np.issubdtype(series.dtype, np.number):
        return
    print("\n--- Starting numeric data exploration ---")
    print(f"Data mean is {series.mean():,.2f}, std is {series.std():,.2f}")
    print("It's also usefull to examine the two corresponding robust stats:")
    print((f"Data median is {series.median():,.2f}, median absolute deviation "
           f"(MAD) is {mad(series):,.2f}."))

    bold("\nStaring to test for Normal (Gaussian) data distribution.")
    print(f"Using a significance level (α) of {alpha*100}%.")
    if len(series) > 5000:
        comment((
            "SciPy implmentation for the Shapiro-Wilk test for normality "
            "does not implement parameter censoring, so for N > 5000 the W "
            "test statistic is accurate but the p-value may not be. Thus, "
            "and since N>5000, skipping the Shapiro-Wilk test."))
    else:
        print("Performing the Shapiro-Wilk test for normality...")
        print("Null hypothesis (H0): The data comes from a normal dist.")
        shap_stat, shap_pval = sp.stats.shapiro(series)
        print(f"Test statistic: {shap_stat:.3f} p-value: {shap_pval:.3f}")
        if shap_pval < alpha:
            print(("The p-value is smaller than the set α; the null hypothesis"
                   " can be rejected: the data isn't  normally distributed."))
        else:
            print(("The p-value is larger than the set α; the null hypothesis"
                   " cannot be rejected: the data is  normally distributed."))
    print("Performing the D’Agostino’s K^2 test for normality...")
    dag_stat, dag_pval = sp.stats.normaltest(series)
    print(f"Test statistic: {dag_stat:.3f} p-value: {dag_pval:.3f}")
    if dag_pval < alpha:
        print(("The p-value is smaller than the set α; the null hypothesis"
               " can be rejected: the data isn't  normally distributed."))
    else:
        print(("The p-value is larger than the set α; the null hypothesis"
               " cannot be rejected: the data is  normally distributed."))


def _val_counts_plot(vcounts, count, label="series"):
    if len(vcounts) < 2:
        print("One unique value. Skipping value counts plot.")
        return
    if vcounts.iloc[0] < 3:
        print(("Most frequent value appears less than 3 time. Skipping value"
               " counts plot."))
        return
    vcounts_p = vcounts.reset_index()
    if len(vcounts) > 10:
        vcounts_p = vcounts_p[0:10]
    # building informative index
    new_index = []
    for x in vcounts_p['index']:
        newi = str(x)
        if len(newi) > 20:
            newi = f"{newi[:18]}..."
        newi = f"{newi} ({vcounts[x]*100/count:.2f}%)"
        new_index.append(newi)
    vcounts_p['index'] = new_index
    sns.barplot(
        data=vcounts_p, x=vcounts_p.columns[1], y='index',
        palette='Spectral').set_title(f"10 most frequent values of {label}")
    plt.show()


def _explore_series(series, label=None):
    bold(f"\n=================== {label} =================================")
    print(f"Starting to explore series {label} with pdexplore.")
    print(f"dtype: {series.dtype}")
    count = len(series)
    unique_vals = series.unique()
    print(f"{len(unique_vals):,} unique values over {count:,} entries.")
    count_na = sum(series.isna())
    print(f"{count_na*100/count:.2f}% missing values ({count_na:,}).")
    vcounts = series.value_counts()
    _val_counts_plot(vcounts, count, label=label)
    _explore_numeric_series(series, count)


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


def _explore_df(df):
    print("Starting to explore a dataframe with pdexplore.")
    print(f"The dataframe contains {len(df.columns)} columns.")
    print(f"The dataframe contains {len(df)} rows.")
    for col_lbl in df.columns:
        _explore_series(
            series=df[col_lbl],
            label=col_lbl,
        )


def explore(df, output_path=None, silent=False):
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
    silent : bool, optional
        If set to True, no output is printed to screen. Defaults to False.
    """
    set_printing_to_screen(not silent)
    if output_path is not None:
        output_path = get_output_fpath(output_path)
        with open(output_path, 'wt+') as out_f:
            set_output_f(out_f)
            _explore_df(df)
    else:
        if OUTPUT_F is None and silent:
            raise ValueError(
                "No output path is given but function is set to"
                " silent. That's silly. Terminating."
            )
        _explore_df(df)
