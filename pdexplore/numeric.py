"""Numeric explorations for pandas series."""

import numpy as np
import scipy as sp
from statsmodels.robust import mad
from scipy.stats import (
    skew,
    skewtest,
)

from .core import (
    SeriesExploration,
    precondition,
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
        return len(srs.dropna()) >= 4

    def _explore(self, srs):
        nona = srs.dropna()
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
        self.save(srs, 'nona', nona)
        self.save(srs, 'skewness', skwns)


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
        return len(srs.dropna()) >= 8

    def _explore(self, srs):
        print((
            f"Performing skewness test with α={self.alpha}. H0 is that the "
            "skewness of the population that the sample was drawn from is the"
            "same as that of a corresponding normal distribution."))
        skew_zscore, skew_pval = skewtest(srs.nona)
        self.save(srs, 'skew_zscore', skew_zscore)
        self.save(srs, 'skew_pval', skew_pval)
        print((
            f"Skew test z-score is {skew_zscore:,.2f}, p-value is "
            f"{skew_pval:,.2f}"))
        if skew_pval < self.alpha:
            print(("The p-value is smaller than the set α; the null hypothesis"
                   " can be rejected: data skewness is not normal-like."))
            self.save(srs, 'normal_skewness', False)
        else:
            print(("The p-value is larger than the set α; the null hypothesis"
                   " cannot be rejected: data skewness is normal-like."))
            self.save(srs, 'normal_skewness', True)


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
        return len(srs.dropna()) >= 4

    @precondition(fail_msg="More than 5000 non-null values")
    def has_atmost_5000_non_null_values(self, srs):
        return len(srs.dropna()) <= 5000

    def _explore(self, srs):
        print("Performing the Shapiro-Wilk test for normality...")
        print("Null hypothesis (H0): The data comes from a normal dist.")
        shap_stat, shap_pval = sp.stats.shapiro(srs.pdexplore['nona'])
        self.save(srs, 'shapiro_stat', shap_stat)
        self.save(srs, 'shapiro_pval', shap_pval)
        print(f"Test statistic: {shap_stat:.3f} p-value: {shap_pval:.3f}")
        if shap_pval < self.alpha:
            print(("The p-value is smaller than the set α; the null hypothesis"
                   " can be rejected: the data isn't normally distributed."))
            self.save(srs, 'shapiro_normal', False)
        else:
            print(("The p-value is larger than the set α; the null hypothesis"
                   " cannot be rejected: the data is normally distributed."))
            self.save(srs, 'shapiro_normal', True)


DEFAULT_NUMERIC_EXPLORATIONS_ORDER = [
    BasicNumericExploration,
    SkewnessTest,
]
