from .OneDim import OneDim

import copy
import math
import numpy as np
import pandas as pd
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb

class Repetition(OneDim):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(Repetition, self).__init__(params=params)
        self.name = "repetition"
        self.descr = "Word repetitions added to sentences"

    @staticmethod
    def create_repetitions(\
            sentence : str,\
            doc : spacy.tokens.doc.Doc,\
            step_snt : float,\
            phraseLength : int = 4) -> bool:

        for i in reversed(range(0, len(doc))):

            # find phrase at the end of the sentence without punctuation in it incrementally
            token_slice = doc[(i - phraseLength):i]
            for j in reversed(range(phraseLength)):
                j += 1
                token_slice = doc[(i - j):i]
                if not True in [token.pos_ == 'PUNCT' for token in token_slice]:
                    phraseLength = j
                    # break
                    token_slice = doc[(i - phraseLength):i]

                    acc : list = []
                    for k in range(i - phraseLength):
                        acc.append(doc[k])

                    n_times : int = math.floor(step_snt * len(doc))

                    if n_times == 0:
                        return sentence, True

                    acc += [token for token in token_slice] * n_times + [token for token in doc[i:len(doc)]]
                    
                    sent : str = ""

                    for i in range(len(acc)):

                        # TODO annotate
                        token = acc[i]

                        word = ""
                        if i == 0 or token.pos_ == "PUNCT":
                            word = token.text
                        else:
                            word = " " + token.text
                        sent += word

                    if len(sent) == 0:
                        print("Sentence empty! Repetition.")
                        return sent, False

                    return sent, True
        
        return None, False


    def perturbate(self) -> None:
        # [(degree of deterioration, deteriorated text, indices)]

        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr) * len(self.texts))


        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):
                
                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                for i in range(len(sentences)):
                    
                    if len(doc[i]) < 2:
                        continue

                    new_sentence = sentences[i]
                    new_sentence, success = self.create_repetitions(sentence=sentences[i], doc=doc[i], step_snt=step)

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence
                    
                
                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)
                bar.next()

            self.dmgd_texts.append(ret_tuple)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        for m in metrics:
            if m.id and candidate == reference:
                yield m.get_id(cand=candidate, ref=reference)
            else:
                yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:
        super(Repetition, self).evaluate(metrics)
        self.step_arr = [ "Rep. len.: " + str(step) + " * len(sents)" for step in self.step_arr]

    def create_table(self, metrics : list) -> None:
        data : list = []
        for i, step in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in self.combined_results[i][metric.name][submetric]:
                        scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree' : str(step), 'value' : float(value)}
                        data.append(scatter_struc)
        
        self.df = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree', 'value'])