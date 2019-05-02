"""Core functionalities for pdexplore."""

import abc
import inspect

import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.robust import mad
from scipy.stats import (
    skew,
    skewtest,
)

from .util import (
    get_output_fpath,
    OUTPUT_F,
    set_printing_to_screen,
    set_output_f,
    custom_print as print,
    comment,
    bold,
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
                    print(f"{self.name} skipped. Reason: {method.fail_msg}.")
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

    @classmethod
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


DEF_NORMALITY_ALPHA = 0.05


def _explore_numeric_series(series, count, alpha=None):
    nona = series.dropna()
    if alpha is None:
        alpha = DEF_NORMALITY_ALPHA
    if not np.issubdtype(series.dtype, np.number):
        return
    if len(nona) < 4:
        print("Skipping numeric exploration as N < 4.")
        return
    print("\n--- Starting numeric data exploration ---")
    comment((f"Using an α (alpha) value of {alpha*100}% for all tests. This "
             "will be used as the significance level of every statistical test"
             " conducted; finally, it will be re-interpreted as the control "
             "level for the false discovery rate (FDR) of the set of conducted"
             " tests."))
    print(f"Data mean is {series.mean():,.2f}, std is {series.std():,.2f}")
    print((
        "It's also usefull to examine the two corresponding outlier-robust "
        "stats:"))
    print((f"Median={series.median():,.2f}, min={series.min()}, "
           f"max={series.max()}, median absolute deviation "
           f"(MAD)={mad(nona):,.2f}."))
    print((f"Data skewness is {skew(nona):,.2f}. For normally distributed "
           "data, the skewness should be about 0. A skewness value > 0 means "
           "that there is more weight in the left tail of the distribution."))
    if len(nona) < 8:
        comment("Can't perform skewness test, as N<8.")
    else:
        print((
            "Performing skewness test. H0 is that the skewness of the "
            "population that the sample was drawn from is the same as that of "
            "a corresponding normal distribution."))
        skew_zscore, skew_pval = skewtest(nona)
        print((
            f"Skew test z-score is {skew_zscore:,.2f}, p-value is "
            f"{skew_pval:,.2f}"))
        if skew_pval < alpha:
            print(("The p-value is smaller than the set α; the null hypothesis"
                   " can be rejected: data skewness is not normal-like."))
        else:
            print(("The p-value is larger than the set α; the null hypothesis"
                   " cannot be rejected: data skewness is normal-like."))

    bold("\nStaring to test for Normal (Gaussian) data distribution.")
    if len(nona) > 5000:
        comment((
            "SciPy implmentation for the Shapiro-Wilk test for normality "
            "does not implement parameter censoring, so for N > 5000 the W "
            "test statistic is accurate but the p-value may not be. Thus, "
            "and since N>5000, skipping the Shapiro-Wilk test."))
    else:
        print("Performing the Shapiro-Wilk test for normality...")
        print("Null hypothesis (H0): The data comes from a normal dist.")
        shap_stat, shap_pval = sp.stats.shapiro(nona)
        print(f"Test statistic: {shap_stat:.3f} p-value: {shap_pval:.3f}")
        if shap_pval < alpha:
            print(("The p-value is smaller than the set α; the null hypothesis"
                   " can be rejected: the data isn't normally distributed."))
        else:
            print(("The p-value is larger than the set α; the null hypothesis"
                   " cannot be rejected: the data is normally distributed."))
    if len(nona) < 8:
        comment("Can't perform D’Agostino’s K^2 test for normality, as N<8.")
    else:
        print("Performing the D’Agostino’s K^2 test for normality...")
        dag_stat, dag_pval = sp.stats.normaltest(nona)
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


def _explore_df(df, skip_lbl=None, skip_cond=None):
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
