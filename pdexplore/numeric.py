"""Numeric explorations for pandas series."""

import numpy as np
import scipy as sp
from statsmodels.robust import mad
from scipy.stats import (
    skew,
    skewtest,
)

from .base import (
    SeriesExploration,
    precondition,
)
from .util import (
    custom_print as print,
    warning,
)


DEF_ALPHA = 0.05


class _BaseNumericExploration(SeriesExploration):

    @precondition(fail_msg="dtype is non-numeric")
    def is_of_numeric_dtype(self, srs):
        return np.issubdtype(srs.dtype, np.number)


class BasicNumericExploration(_BaseNumericExploration):

    @property
    def name(self):
        return "Basic numeric exploration"

    @precondition(fail_msg="Less than four non-null values")
    def has_atleast_four_non_null_values(self, srs):
        nona = srs.dropna()
        BasicNumericExploration.save(srs, 'nona', nona)
        return len(nona) >= 4

    def _explore(self, srs):
        nona = srs.pdexplore['nona']
        print("\n--- Starting numeric data exploration ---")
        print(f"Data min={srs.min():,.2f}, max={srs.max():,.2f}.")
        print(f"Data mean is {srs.mean():,.2f}, std is {srs.std():,.2f}")
        print((
            "It's also usefull to examine the two corresponding outlier-robust"
            " stats:"))
        print((
            f"Median={srs.median():,.2f}, median absolute deviation "
            f"(MAD)={mad(nona):,.2f}."))
        skwns = skew(nona)
        print((
            f"Data skewness is {skwns:,.2f}. For normally distributed "
            "data, the skewness should be about 0. A skewness value > 0 means "
            "that there is more weight in the left tail of the distribution."))
        BasicNumericExploration.save(srs, 'skewness', skwns)
        if 'vcounts' not in srs.pdexplore:
            vcounts = srs.value_counts()
            BasicNumericExploration.save(srs, 'vcounts', vcounts)


class SkewnessTest(_BaseNumericExploration):

    def __init__(self, alpha=None):
        if alpha is None:
            alpha = DEF_ALPHA
        self.alpha = alpha

    @property
    def name(self):
        return "Skewness test"

    @precondition(fail_msg="Less than eight non-null values")
    def has_atleast_eight_non_null_values(self, srs):
        return len(srs.pdexplore['nona']) >= 8

    def _explore(self, srs):
        print((
            f"Performing skewness test with α={self.alpha}. H0 is that the "
            "skewness of the population that the sample was drawn from is the"
            "same as that of a corresponding normal distribution."))
        skew_zscore, skew_pval = skewtest(srs.pdexplore['nona'])
        BasicNumericExploration.save(srs, 'skew_zscore', skew_zscore)
        BasicNumericExploration.save(srs, 'skew_pval', skew_pval)
        print((
            f"Skew test z-score is {skew_zscore:,.2f}, p-value is "
            f"{skew_pval:,.2f}"))
        if skew_pval < self.alpha:
            print(("The p-value is smaller than the set α; the null hypothesis"
                   " can be rejected: data skewness is not normal-like."))
            BasicNumericExploration.save(srs, 'normal_skewness', False)
        else:
            print(("The p-value is larger than the set α; the null hypothesis"
                   " cannot be rejected: data skewness is normal-like."))
            BasicNumericExploration.save(srs, 'normal_skewness', True)


class ShapiroWilkNormalityTest(_BaseNumericExploration):

    def __init__(self, alpha=None):
        if alpha is None:
            alpha = DEF_ALPHA
        self.alpha = alpha

    @property
    def name(self):
        return "Shapiro-Wilk normality test"

    @precondition(fail_msg="Less than three non-null values")
    def has_atleast_four_non_null_values(self, srs):
        return len(srs.pdexplore['nona']) >= 4

    @precondition(fail_msg="More than 5000 non-null values")
    def has_atmost_5000_non_null_values(self, srs):
        return len(srs.pdexplore['nona']) <= 5000

    def _explore(self, srs):
        print("Performing the Shapiro-Wilk test for normality...")
        print("Null hypothesis (H0): The data comes from a normal dist.")
        shap_stat, shap_pval = sp.stats.shapiro(srs.pdexplore['nona'])
        BasicNumericExploration.save(srs, 'shapiro_stat', shap_stat)
        BasicNumericExploration.save(srs, 'shapiro_pval', shap_pval)
        print(f"Test statistic: {shap_stat:.3f} p-value: {shap_pval:.3f}")
        if shap_pval < self.alpha:
            print((
                "The p-value is smaller than the set α; the null hypothesis"
                " - that the data is normally distributed - can be rejected."))
            BasicNumericExploration.save(srs, 'shapiro_normal', False)
        else:
            print(("The p-value is larger than the set α; the null hypothesis"
                   " - that the data is normally distributed - cannot be "
                   "rejected."))
            BasicNumericExploration.save(srs, 'shapiro_normal', True)


class DagostinoNormalityTest(_BaseNumericExploration):

    def __init__(self, alpha=None):
        if alpha is None:
            alpha = DEF_ALPHA
        self.alpha = alpha

    @property
    def name(self):
        return "D’Agostino’s K^2 normality test"

    @precondition(fail_msg="Less than eight non-null values")
    def has_atleast_eight_non_null_values(self, srs):
        return len(srs.pdexplore['nona']) >= 8

    def _explore(self, srs):
        print("Performing the D’Agostino’s K^2 test for normality...")
        print("Null hypothesis (H0): The data comes from a normal dist.")
        dag_stat, dag_pval = sp.stats.normaltest(srs.pdexplore['nona'])
        BasicNumericExploration.save(srs, 'dag_stat', dag_stat)
        BasicNumericExploration.save(srs, 'dag_pval', dag_pval)
        print(f"Test statistic: {dag_stat:.3f} p-value: {dag_pval:.3f}")
        if dag_pval < self.alpha:
            print((
                "The p-value is smaller than the set α; the null hypothesis"
                " - that the data is normally distributed - can be rejected."))
            BasicNumericExploration.save(srs, 'dagostino_normal', False)
        else:
            print(("The p-value is larger than the set α; the null hypothesis"
                   " - that the data is normally distributed - cannot be "
                   "rejected."))
            BasicNumericExploration.save(srs, 'dagostino_normal', True)


SUSPICIOUS_COMPUTING_INT_DICT = [
    # 8-bits
    {
        "number": -128,
        "equivalent": "-(2^7)",
        "location": "lowest",
        "sign": "a signed",
        "bits": 8,
        "minmax": "min",
    },
    {
        "number": 127,
        "equivalent": "2^7-1",
        "location": "highest",
        "sign": "a signed",
        "bits": 8,
        "minmax": "max",
    },
    {
        "number": 255,
        "equivalent": "2^8-1",
        "location": "highest",
        "sign": "an unsigned",
        "bits": 8,
        "minmax": "max",
    },
    # 16-bits
    {
        "number": -32768,
        "equivalent": "-(2^15)",
        "location": "lowest",
        "sign": "a signed",
        "bits": 16,
        "minmax": "min",
    },
    {
        "number": 32767,
        "equivalent": "2^15-1",
        "location": "highest",
        "sign": "a signed",
        "bits": 16,
        "minmax": "max",
    },
    {
        "number": 65535,
        "equivalent": "2^16-1",
        "location": "highest",
        "sign": "an unsigned",
        "bits": 16,
        "minmax": "max",
    },
    # 32-bits
    {
        "number": -2147483648,
        "equivalent": "-(2^31)",
        "location": "lowest",
        "sign": "a signed",
        "bits": 32,
        "minmax": "min",
    },
    {
        "number": 2147483647,
        "equivalent": "2^31-1",
        "location": "highest",
        "sign": "a signed",
        "bits": 32,
        "minmax": "max",
    },
    {
        "number": 4294967295,
        "equivalent": "2^32-1",
        "location": "highest",
        "sign": "an unsigned",
        "bits": 32,
        "minmax": "max",
    },
    # 64-bits
    {
        "number": -pow(2, 63),
        "equivalent": "-(2^63)",
        "location": "lowest",
        "sign": "a signed",
        "bits": 64,
        "minmax": "min",
    },
    {
        "number": pow(2, 63) - 1,
        "equivalent": "2^63-1",
        "location": "highest",
        "sign": "a signed",
        "bits": 64,
        "minmax": "max",
    },
    {
        "number": pow(2, 64) - 1,
        "equivalent": "2^64-1",
        "location": "highest",
        "sign": "an unsigned",
        "bits": 64,
        "minmax": "max",
    },
    # 128-bits
    {
        "number": -pow(2, 127),
        "equivalent": "-(2^127)",
        "location": "lowest",
        "sign": "a signed",
        "bits": 128,
        "minmax": "min",
    },
    {
        "number": pow(2, 127) - 1,
        "equivalent": "2^127-1",
        "location": "highest",
        "sign": "a signed",
        "bits": 128,
        "minmax": "max",
    },
    {
        "number": pow(2, 128) - 1,
        "equivalent": "2^128-1",
        "location": "highest",
        "sign": "an unsigned",
        "bits": 128,
        "minmax": "max",
    },
]


class SuspiciousNumbersCheck(_BaseNumericExploration):

    @property
    def name(self):
        return "Suspicious numers check"

    def _explore(self, srs):
        vcounts = srs.pdexplore['vcounts']
        for x in SUSPICIOUS_COMPUTING_INT_DICT:
            n = x['number']
            if n in vcounts:
                warning(
                    f"{n:,} found {vcounts[n]} times. It is suspicious, as it"
                    f" is exactly {x['equivalent']}; i.e. the {x['location']} "
                    f"number that can be represented by {x['sign']} "
                    f"{x['bits']}-bit binary number. It is therefore the "
                    f"{x['minmax']} value for variables declared as integers "
                    "in many programming language. The appearance of the "
                    "number may reflect an error, overflow condition or "
                    "missing value.")
            if len(srs) == n:
                warning(
                    f"Series length is {n:,}. It is suspicious, as it"
                    f" is exactly {x['equivalent']}; i.e. the {x['location']} "
                    f"number that can be represented by {x['sign']} "
                    f"{x['bits']}-bit binary number. It is therefore the "
                    f"{x['minmax']} value for variables declared as integers "
                    "in many programming language. In the case of series "
                    "length, it can imply the dataset itself was trimmed or "
                    "sliced.")
        for n in [99, 999, 9999, 99999, 999999, 9999999]:
            if n in vcounts:
                print(
                    f"{n:,} found {vcounts[n]} times. This might be suspicious"
                    ", as all-9 numbers are often used as sentinel values.")


DEFAULT_NUMERIC_EXPLORATIONS_ORDER = [
    BasicNumericExploration,
    SkewnessTest,
    ShapiroWilkNormalityTest,
    DagostinoNormalityTest,
    SuspiciousNumbersCheck,
]


def run_numeric_exploration_pipeline(srs):
    stages = []
    for clas in DEFAULT_NUMERIC_EXPLORATIONS_ORDER:
        stages.append(clas())
    for stage in stages:
        stage.apply(srs)
