"""Base Class for all Tasks."""

import copy
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import random
import spacy

from functools import reduce
from progress.bar import ShadyBar
from typing import IO


class Task():
    """Base Class for all Tasks."""

    __slots__ = [
        "texts", "results", "dmgd_texts",
        "combined_results", "step_arr", "path",
        "name", "df", "descr"]

    def __init__(self, params: dict):
        """Initialize."""
        self.results: list = []
        self.dmgd_texts: list = []
        self.combined_results: list = []
        self.step_arr: list = []
        self.texts: list = params['texts']
        self.path: str = params['path']
        self.name: str =\
            (
                "Add a description instance to the"
                "__init__ method of the derived class.")
        self.df: pd.DataFrame = pd.DataFrame()
        self.descr: str =\
            (
                "Add a description instance to the"
                "__init__ method of the derived class.")

        random.seed(params['seed'])

    def set_steps(self, steps: dict):
        """Set steps.

        Params
        ------
        steps : dict
            dictionary containing information to
            different levels of modification.
        """
        return self

    def perturbate(self) -> None:
        """Perturbate sentences."""
        pass

    def evaluate(self, metrics: list) -> None:
        """Create Table.

        Evaluate samples.

        Params
        ------
        metrics : list
            list of metrics
        """
        pass

    def combine_results(self, metrics: list) -> None:
        """Combine results.

        Combine results into one custom data type.

        Params
        ------
        metrics : list
            list of metrics
        """
        pass

    def plot(
            self,
            ax: any,
            metric: any,
            submetric: str,
            **kwargs) -> None:
        """Plot.

        UNUSED.
        """
        pass

    # OBSOLETE
    def dump(
            self,
            data: any,
            descr: str) -> None:
        """Dump.

        UNUSED.
        """
        f_name: str = "." + self.name + "_" + descr + "_data.p"
        path: str = os.path.join(self.path, f_name)
        print(path)

        f: IO = open(path, 'wb')
        pickle.dump(data, f)
        f.close()
