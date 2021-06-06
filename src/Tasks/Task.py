from __future__ import annotations


import copy
import math
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import spacy

from functools import reduce

class Task():

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "df_sct", "step_arr", "path", "name"]

    def __init__(self, data: list, nlp: spacy.lang, path : str = ""):

        self.texts: list = []
        self.results : list = []
        self.combined_results : list = []
        self.step_arr : list = []
        self.dmgd_texts : list = []
        self.path : str = path
        self.df_sct : pd.DataFrame = None

        for text in data:
            sentences: list = nltk.sent_tokenize(text)
            doc: list = list(nlp.pipe(sentences))
            self.texts.append((sentences, doc))

    def set_steps(self, steps : int) -> Task:
        step_size : float = 1 / steps
        self.step_arr = np.flip(1 - np.concatenate( (np.arange(0,1, step=step_size), np.array([1])) ) )
        return self

    def perturbate_1d(self, f : callable) -> None:
        # [(degree of deterioration, deteriorated text, indices)]

        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                sample : int = int(math.floor(step * len(sentences)))

                for i in range(sample):
                    
                    new_sentence = sentences[i]
                    new_sentence, success = f(sentence=sentences[i], doc=doc[i])

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence
                    
                
                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)

            self.dmgd_texts.append(ret_tuple)

    def perturbate_2d(self, f : callable) -> None:
    
        for step_txt in self.step_arr[0]:
            ret_txt : list = []
            for step_snt in self.step_arr[1]:
                ret_tuple_snt : tuple = ([], [])
                for _, (sentences, doc) in enumerate(self.texts):
                    sample : int = int(math.floor(step_txt * len(sentences)))

                    sentences : list = copy.deepcopy(sentences)
                    indices : list = []

                    if step_txt == 0.0 or step_snt == 0.0:
                        ret_tuple_snt[0].append([])
                        ret_tuple_snt[1].append([])
                        continue

                    for i in range(sample):

                        new_sentence = sentences[i]
                        new_sentence, success = f(sentence=new_sentence, doc=doc[i], step_snt=step_snt)
                        if success:
                            indices.append(i)
                            sentences[i] = new_sentence

                    ret_tuple_snt[0].append(sentences)
                    ret_tuple_snt[1].append(indices)
                ret_txt.append(ret_tuple_snt)
            self.dmgd_texts.append(ret_txt)

    def perturbate(self, params: dict) -> None:
        pass

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        print("her\n")
        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics: list) -> None:
        pass

    def combine_results(self, metrics : list) -> None:
        pass

    def plot(self, fig: plt.figure) -> None:
        pass
