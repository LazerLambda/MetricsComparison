from .OneDim import OneDim
from .Negate import Negate
from .SwapWordsOneDim import SwapWordsOneDim
from .DropWordsOneDim import DropWordsOneDim
from .POSDrop import POSDrop

import copy
import math
import numpy as np
import pandas as pd
import random
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb


class Mix(OneDim):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr", "nlp"]

    def __init__(self, params : dict):
        super(Mix, self).__init__(params=params)
        self.name : str = "mix"
        self.descr : str = "Different deteriorations applied sequentally"
        self.nlp = spacy.load('en_core_web_sm')
        self.step_arr : list = [
            'Original',
            'Neg + Swap',
            "Neg - POS 'ADJ'",
            "Neg - POS 'DET'",
            "Neg - POS 'VERB'",
            "Neg - POS 'NOUN'",
            'Neg + Drop'] 




    # override
    def perturbate(self) -> None:
        # [(degree of deterioration, deteriorated text, indices)]


        self.step_arr : list = [
            'Original',
            'Neg + Swap',
            "Neg - POS 'ADJ'",
            "Neg - POS 'DET'",
            "Neg - POS 'VERB'",
            "Neg - POS 'NOUN'",
            'Neg + Drop']  
        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr) * len(self.texts))


        for i, action in enumerate(self.step_arr):
            ret_tuple : tuple = ([], []) 
            for j, (sentences, doc) in enumerate(self.texts):
                
                indices : list = []
                sentences = copy.deepcopy(sentences)

                # identity category
                if action == 'Original':
                    ret_tuple[0].append(sentences)
                    ret_tuple[1].append(indices)
                    bar.next()
                    continue 
                
                # TODO annotate
                # doc = list(self.nlp.pipe(sentences))

                for k in range(len(sentences)):
                    
                    if len(doc[k]) < 2:
                        continue

                    # 'Neg + Swap'
                    if action == 'Neg + Swap':
                        new_sentence, success = Negate.negate(sentence=sentences[k], doc=doc[k])
                        if success:
                            new_sentence_2, success = SwapWordsOneDim.swap_pairs(sentence=new_sentence, doc=self.nlp(new_sentence), step=1)
                            if success:
                                indices.append(k)
                                sentences[k] = new_sentence_2
                    # 'Negated + POS Dropped'
                    if action == "Neg - POS 'ADJ'":
                        new_sentence, success = Negate.negate(sentence=sentences[k], doc=doc[k])
                        if success:
                            new_sentence_2, success = POSDrop.drop_single_pos(sentence=new_sentence, doc=self.nlp(new_sentence), pos='ADJ')
                            if success:
                                indices.append(k)
                                sentences[k] = new_sentence_2
                    if action == "Neg - POS 'DET'":
                        new_sentence, success = Negate.negate(sentence=sentences[k], doc=doc[k])
                        if success:
                            new_sentence_2, success = POSDrop.drop_single_pos(sentence=new_sentence, doc=self.nlp(new_sentence), pos='DET')
                            if success:
                                indices.append(k)
                                sentences[k] = new_sentence_2
                    if action == "Neg - POS 'VERB'":
                        new_sentence, success = Negate.negate(sentence=sentences[k], doc=doc[k])
                        if success:
                            new_sentence_2, success = POSDrop.drop_single_pos(sentence=new_sentence, doc=self.nlp(new_sentence), pos='VERB')
                            if success:
                                indices.append(k)
                                sentences[k] = new_sentence_2
                    if action == "Neg - POS 'NOUN'":
                        new_sentence, success = Negate.negate(sentence=sentences[k], doc=doc[k])
                        if success:
                            new_sentence_2, success = POSDrop.drop_single_pos(sentence=new_sentence, doc=self.nlp(new_sentence), pos='NOUN')
                            if success:
                                indices.append(k)
                                sentences[k] = new_sentence_2
                    # 'Neg + Drop'
                    if action == 'Neg + Drop':
                        new_sentence, success = Negate.negate(sentence=sentences[k], doc=doc[k])
                        if success:
                            new_sentence_2, success = DropWordsOneDim.drop_single(sentence=new_sentence, doc=self.nlp(new_sentence), step=1)
                            if success:
                                indices.append(k)
                                sentences[k] = new_sentence_2

                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)
                bar.next()

            self.dmgd_texts.append(ret_tuple)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()

    # override
    def create_table(self, metrics : list) -> None:
        data : list = []
        for i, step in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in self.combined_results[i][metric.name][submetric]:
                        # 'degree' is a string here
                        scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree' : str(step), 'value' : float(value)}
                        data.append(scatter_struc)
        
        self.df = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree', 'value'])
        