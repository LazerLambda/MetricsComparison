

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

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):

        
        self.results: list = []
        self.dmgd_texts: list = []
        self.combined_results: list = []
        self.step_arr: list = []
        self.texts: list = params['texts']
        self.path: str = params['path']
        self.name: str = "Add a description instance to the __init__ method of the derived class."
        self.df: pd.DataFrame = pd.DataFrame()
        self.descr: str = "Add a description instance to the __init__ method of the derived class."
        
        random.seed(params['seed'])


        # data : list = params['data']
        # nlp : spacy.lang = params['nlp']

        # for text in data:
        #     sentences: list = nltk.sent_tokenize(text)
        #     doc: list = list(nlp.pipe(sentences))
        #     self.texts.append((sentences, doc))

    def set_steps(self, steps : dict):
        return self

    def perturbate_1d(self, f : callable) -> None:
        # [(degree of deterioration, deteriorated text, indices)]

        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr) * len(self.texts))

        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                sample : int = int(math.floor(step * len(sentences)))

                for i in range(sample):
                    
                    if len(doc[i]) < 2:
                        continue

                    new_sentence = sentences[i]
                    new_sentence, success = f(sentence=sentences[i], doc=doc[i])

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence
                    
                
                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)
                bar.next()

            self.dmgd_texts.append(ret_tuple)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()

    def perturbate_2d(self, f : callable) -> None:

        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr[0]) * len(self.step_arr[1]) * len(self.texts))
    
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
                        bar.next()
                        continue

                    for i in range(sample):

                        if len(doc[i]) < 2:
                            continue

                        new_sentence = sentences[i]
                        new_sentence, success = f(sentence=new_sentence, doc=doc[i], step_snt=step_snt)
                        if success:
                            indices.append(i)
                            sentences[i] = new_sentence

                    ret_tuple_snt[0].append(sentences)
                    ret_tuple_snt[1].append(indices)
                    bar.next()
                ret_txt.append(ret_tuple_snt)
            self.dmgd_texts.append(ret_txt)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()

    def perturbate(self) -> None:
        pass

    def evaluate(self, metrics: list) -> None:
        pass

    def combine_results(self, metrics : list) -> None:
        pass

    def plot(self, ax : any, metric : any, submetric : str, **kwargs) -> None:
        pass


        # OBSOLETE
    def dump(self, data : any, descr : str) -> None:

        f_name : str = "." + self.name + "_" + descr + "_data.p"
        path : str = os.path.join(self.path, f_name)
        print(path)
        
        f : IO = open(path, 'wb')
        pickle.dump(data, f)
        f.close()
