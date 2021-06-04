from __future__ import annotations


import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import spacy


class Task():

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path"]

    def __init__(self, data: list, nlp: spacy.lang, path : str = ""):

        self.texts: list = []
        self.results : list = []
        self.combined_results : list = []
        self.step_arr : list = []
        self.dmgd_texts : list = []
        self.path : str = path

        for text in data:
            sentences: list = nltk.sent_tokenize(text)
            doc: list = list(nlp.pipe(sentences))
            self.texts.append((sentences, doc))

    def set_steps(self, steps : int) -> Task:
        step_size : float = 1 / steps
        self.step_arr = np.flip(1 - np.concatenate( (np.arange(0,1, step=step_size), np.array([1])) ) )
        return self

    def perturbate(self, params: dict) -> None:
        pass

    def evaluate(self, metrics: list) -> None:
        pass

    def combine_results(self, metrics : list) -> None:
        pass

    def plot(self, fig: plt.figure) -> None:
        pass
