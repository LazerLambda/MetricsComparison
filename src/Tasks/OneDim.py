from .Task import Task

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from checklist.perturb import Perturb

class OneDim(Task):



    def evaluate(self, metrics : list) -> None:
        self.evaluate_1d(metrics)

    def combine_results(self, metrics : list) -> None:
        self.combine_results_1d(metrics)

    def create_table(self, metrics : list) -> None:
        self.create_table_1d(metrics)

    def get_results(self) -> None:
        self.get_results_1d()

    # TODO annotate
    def plot(self, ax, title : str) -> None:
        self.plot_1d(ax, title)