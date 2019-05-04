"""General data explorations."""

import seaborn as sns
from matplotlib import pyplot as plt

from .base import (
    SeriesExploration,
    precondition,
)
from .util import (
    custom_print as print,
    # comment,
    bold,
)


class BasicExploration(SeriesExploration):

    def __init__(self, series_label=None):
        self.lbl = series_label

    @property
    def name(self):
        return "Basic exploration"

    def _explore(self, srs):
        bold(f"\n=================== {self.lbl} =============================")
        print(f"Starting to explore series {self.lbl} with pdexplore.")
        print(f"dtype: {srs.dtype}")
        count = len(srs)
        unique_vals = srs.unique()
        print(f"{len(unique_vals):,} unique values over {count:,} entries.")
        count_na = sum(srs.isna())
        print(f"{count_na*100/count:.2f}% missing values ({count_na:,}).")
        vcounts = srs.value_counts()
        BasicExploration.save(srs, 'vcounts', vcounts)


class ValueCountsPlot(BasicExploration):

    @property
    def name(self):
        return "Value counts plot"

    @precondition(fail_msg="Less than two uniqe values")
    def has_atleast_two_unique_values(self, srs):
        vcounts = srs.pdexplore['vcounts']
        return len(vcounts) > 1

    @precondition(fail_msg="Low frequency of most-frequent value")
    def high_enough_frequency_of_most_frequent_value(self, srs):
        vcounts = srs.pdexplore['vcounts']
        return vcounts.iloc[0] > 2

    def _explore(self, srs):
        vcounts = srs.pdexplore['vcounts']
        count = len(srs)
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
            palette='Spectral'
        ).set_title(f"10 most frequent values of {self.lbl}")
        plt.show()


GENERAL_EXPLORATIONS_ORDER = [
    BasicExploration,
    ValueCountsPlot,
]


def general_exploration(series, label=None):
    stages = [
        clas(series_label=label)
        for clas in GENERAL_EXPLORATIONS_ORDER
    ]
    for stage in stages:
        stage.apply(series)
