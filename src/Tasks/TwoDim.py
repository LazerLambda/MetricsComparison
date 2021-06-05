from .Task import Task

from functools import reduce

import matplotlib.pyplot as plt

import copy
import math
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sns



class TwoDim(Task):

    #TODO slots
    #TODO subclass for drop
    #TODO description
    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df_sct", "descr"]

    # overwrite
    def set_steps(self, steps : dict) -> Task:
        step_txt : float = 1 / steps['txt']
        step_snt : float = 1 / steps['snt']
        self.step_arr.append(np.flip(1 - np.concatenate( (np.arange(0,1, step=step_txt), np.array([1])))))
        self.step_arr.append(np.flip(1 - np.concatenate( (np.arange(0,1, step=step_snt), np.array([1])))))
        return self

    def evaluate(self, metrics : list) -> None:
        self.evaluate_2d(metrics)

    def combine_results(self, metrics : list) -> None:
        self.combine_results_2d(metrics)

    def create_table(self, metrics : list) -> None:
        self.create_table_2d(metrics)

    def get_results(self) -> None:
        self.get_results_2d()

    # TODO annotate
    def plot(self, ax, title : str, metrics : list) -> None:
        self.plot_2d(ax, title, metrics)