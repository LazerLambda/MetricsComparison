from .OneDim import OneDim

import copy
import math
import numpy as np
import pandas as pd
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb

class Repetitions2(OneDim):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(Repetitions2, self).__init__(params=params)
        self.name = "negation"
        self.descr = "Negated sentences in the text."

    @staticmethod
    def create_repetitions(\
            sentence : list,\
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

                # sample : int = int(math.floor(step * len(sentences)))

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

        self.step_arr = [ "Rep. len.: " + str(step) + " * len(sents)" for step in self.step_arr]

    def __eval(self, reference : list , candidate : list, metrics : list) -> dict:
        for m in metrics:
            yield m.compute(cand=candidate, ref=reference)

    def evaluate(self, metrics : list) -> None:
        if len(metrics) == 0:
            return

        bar : ShadyBar = ShadyBar(message="Evaluating " + self.name, max=len(self.step_arr) * len(self.texts))
        for i, _ in enumerate(self.step_arr):
            step_results : list = []
            for j, (sentences, _) in enumerate(self.texts):
                reference : list = sentences
                candidate : list = self.dmgd_texts[i][0][j]

                # TODO into method
                # drop emtpy sentences
                ref_checked : list = []
                cand_checked : list = []

                # TODO
                for ref, cand in zip(reference, candidate):
                    if len(cand) != 0:
                        ref_checked.append(ref)
                        cand_checked.append(cand)
                    else:
                        continue
                
                reference = ref_checked
                candidate = cand_checked

                if self.step_arr[i] == 0:
                    assert candidate == reference
                    step_results.append([m.get_id() for m in metrics])
                else:
                    step_results.append([*(res for res in self.__eval(reference, candidate, metrics))])
                bar.next()
            self.results.append(step_results)
        bar.finish()

    def create_table(self, metrics : list) -> None:
        data : list = []
        for i, step in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in self.combined_results[i][metric.name][submetric]:
                        scatter_struc : dict = {'metric': metric.name, 'submetric': submetric, 'degree' : str(step), 'value' : float(value)}
                        data.append(scatter_struc)
        
        self.df = pd.DataFrame(data=data, columns=['metric', 'submetric', 'degree', 'value'])