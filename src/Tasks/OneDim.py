"""Class to process one dimensional tasks."""

from .Task import Task
# from ..metrics import Metric

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from checklist.perturb import Perturb
from functools import reduce
from progress.bar import ShadyBar


class OneDim(Task):
    """Base class for all tasks using one axis of deterioration."""

    __slots__ = [
        "texts", "results", "dmgd_texts",
        "combined_results", "step_arr",
        "path", "name", "df", "descr"]

    def __init__(self, params: dict):
        """Initialize."""
        super(OneDim, self).__init__(params=params)

        self.set_steps(params['steps'])

    def set_steps(self, steps: dict) -> Task:
        """Set amount of levels of challenges."""
        step_size: float = 1 / steps['steps']
        self.step_arr = np.flip(
            1 - np.concatenate(
                (
                    np.arange(0, 1, step=step_size),
                    np.array([1])
                )
            ))
        return self

    def __eval(
            self,
            reference: list,
            candidate: list,
            metrics: list) -> dict:
        """Evaluate ref and cand.

        Params
        ------
        reference : list
            list of reference sentences
        candidate : list
            list of candidate sentences
        metrics : list
            list of metrics to be computed

        Returns
        -------
        dictionary of computed values
        """
        for m in metrics:
            if m.id and candidate == reference:
                yield m.get_id(cand=candidate, ref=reference)
            else:
                yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics: list) -> None:
        """Evaluate sentence pairs.

        Params
        ------
        metrics : list
            list of metrics to be evaluated
        """
        if len(metrics) == 0:
            return

        bar: ShadyBar = ShadyBar(
            message="Evaluating " + self.name,
            max=len(self.step_arr) * len(self.texts))

        # iterate over different levels
        for i, _ in enumerate(self.step_arr):
            step_results: list = []
            # iterate over sentences
            for j, (sentences, _) in enumerate(self.texts):
                if i == 0 or len(self.dmgd_texts[i][1][j]) == 0:
                    reference: list = sentences
                    candidate: list = self.dmgd_texts[i][0][j]
                else:
                    indices: np.ndarray = np.asarray(self.dmgd_texts[i][1][j])
                    reference: list = np.asarray(sentences)[indices]
                    candidate: list = np.asarray(
                        self.dmgd_texts[i][0][j])[indices]

                # check if any candidate sentence is
                # empty and keep non-empty ones
                ref_checked: list = []
                cand_checked: list = []

                for ref, cand in zip(reference, candidate):
                    if len(cand) != 0:
                        ref_checked.append(ref)
                        cand_checked.append(cand)
                    else:
                        continue

                reference = ref_checked
                candidate = cand_checked

                # if self.step_arr[i] == 0:
                #     assert candidate == reference

                # compute results
                step_results.append(
                    [*(res for res in self.__eval(
                        reference,
                        candidate,
                        metrics))])
                bar.next()
            self.results.append(step_results)
        bar.finish()

    def combine_results(self, metrics: list) -> None:
        """Combine results.

        Params
        ------
        metrics : list
            list of metrics
        """
        # list of results with each step
        for run in self.results:
            # dict for each metric and each submetric
            acc = dict(
                zip(
                    [metric.name for metric in metrics],
                    [dict(zip(
                        metric.submetrics,
                        [[] for _ in metric.submetrics]))
                        for metric in metrics]))
            # each text
            for result in run:
                # each submetric
                for i, metric in enumerate(metrics):
                    for j, submetric in enumerate(metric.submetrics):
                        # append result to result list for metric.name and
                        # it's specific submetric
                        acc[metric.name][submetric] += result[i][j]
            self.combined_results.append(acc)

    def create_table(self, metrics: list) -> None:
        """Create Table.

        Params
        ------
        metrics : list
            list of metrics
        """
        data: list = []
        for i, step in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in \
                            self.combined_results[i][metric.name][submetric]:
                        scatter_struc: dict = {
                            'metric': metric.name,
                            'submetric': submetric,
                            'degree': float(step),
                            'value': float(value)}
                        data.append(scatter_struc)

        self.df = pd.DataFrame(
            data=data,
            columns=['metric', 'submetric', 'degree', 'value'])

    def plot(self, ax: any, metric: any, submetric: str, **kwargs) -> None:
        """Plot Data.

        UNUSED
        """
        palette: list = [metric.color[submetric]]

        result = self.df[self.df['submetric'] == submetric]

        sns.set_theme(style="ticks", palette="pastel")
        sns.boxplot(
            x="degree",
            y="value",
            hue="submetric",
            palette=palette,
            data=result,
            ax=ax)
        ax.set(ylim=metric.limits)
        ax.set_aspect('equal')
        ax.set_aspect('auto')
        ax.set_ylabel("Results")
        ax.set_xlabel("Degree of deterioration.", fontsize=10)
        # ax.legend(bbox_to_anchor=(0,0), loc="lower left")
        ax.legend().set_visible(False)
