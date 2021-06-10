from .TwoDim import TwoDim

import copy
import math
import numpy as np
import pandas as pd
import random
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb



class DropAndSwap(TwoDim):

    __slots__ = ["texts", "results", "dmgd_texts", "combined_results", "step_arr", "path", "name", "df", "descr"]

    def __init__(self, params : dict):
        super(DropAndSwap, self).__init__(params=params)
        self.name = "drop_and_swap"
        self.descr = "Dropped and swapped words"

    @staticmethod
    def swap_pairs(sentence : str, doc : spacy.tokens.doc.Doc, step_snt : float) -> tuple:

        candidates : list = []
        candidates_text : list = []

        for i in range(len(doc)):

            lower_text = doc[i].text.lower()

            # TODO maybe exclude first token ?

            if doc[i].pos_ != "PUNCT" and not lower_text in candidates_text:
                candidates.append(i)
                candidates_text.append(lower_text)
            else:
                continue

        if len(candidates) < 3:
            return sentence, False

        step_snt = 0.999999 if step_snt == 1.0 else step_snt
        upper : int = math.floor(step_snt * len(candidates))
        if upper % 2 == 1:
            upper += 1

        assert int(upper) <= len(candidates)

        sample : list = random.sample(candidates, upper)
        sample = [(sample[2 * i], sample[2 * i + 1]) for i in range(upper // 2)]

        tmp_l : list = []
        for i in range(len(doc)):
            tmp_l.append(doc[i])

        for i, j in sample:
            # TODO annotate
            tmp_t : any = tmp_l[i]
            tmp_l[i] = tmp_l[j]
            tmp_l[j] = tmp_t

        sent = ""
        for i, token in enumerate(tmp_l):

            word = ""
            if i == 0:
                word = token.text
            else:
                if token.pos_ == "PUNCT":
                    word = token.text
                else:
                    word = " " + token.text
            sent += word

        if len(sent) == 0:
            print("Sentence empty! Word swap")
            return sent, False

        return sent, True

    @staticmethod
    def drop_single(sentence : str, doc : spacy.tokens.doc.Doc, step_snt : float) -> tuple:
        # TODO add upper bound for dropping

        bound : float = 1 - 1 / len(doc)
        if len(doc) == 0:
            return sentence, False

        candidates : list = []
        for i in range(len(doc)):
            if doc[i].pos_ != "PUNCT":
                candidates.append(i)
            else:
                continue

        # one word must be in the sentence at least
        if step_snt > bound:
            step_snt = bound

        prop : float = int(math.floor(step_snt * len(candidates)))
        drop_list : list = random.sample(candidates, k=prop)

        sent : str = ""

        for i in range(len(doc)):

            # exclude words to be dropped
            if i in drop_list:
                continue

            # TODO annotate
            token = doc[i]

            word = ""
            if i == 0:
                word = token.text
            else:
                if token.pos_ == "PUNCT":
                    word = token.text
                else:
                    word = " " + token.text
            sent += word

        if len(sent) == 0:
            print("Sentence empty! Word drop")
            return sent, False

        return sent, True


    def perturbate(self) -> None:

        bar : ShadyBar = ShadyBar(message="Perturbating " + self.name + " ", max=len(self.step_arr[0]) * len(self.step_arr[1]) * len(self.texts))
    
        for step_txt in self.step_arr[0]:
            ret_txt : list = []
            for step_snt in self.step_arr[1]:
                ret_tuple_snt : tuple = ([], [])
                for _, (sentences, doc) in enumerate(self.texts):
                    # sample : int = int(math.floor(step_txt * len(sentences)))

                    sentences : list = copy.deepcopy(sentences)
                    indices : list = []

                    if step_txt == 0.0 or step_snt == 0.0:
                        ret_tuple_snt[0].append([])
                        ret_tuple_snt[1].append([])
                        bar.next()
                        continue

                    for i in range(len(sentences)):

                        if len(doc[i]) < 2:
                            continue

                        new_sentence = sentences[i]
                        new_sentence, success = self.drop_single(sentence=new_sentence, doc=doc[i], step_snt=step_txt)
                        if success:
                            new_sentence_swapped, success = self.swap_pairs(sentence=new_sentence, doc=doc[i], step_snt=step_snt)
                            if success:
                                sentences[i] = new_sentence_swapped
                            else:
                                print("Sentence:\n'%s'\ncan't be deteriorated further." % new_sentence)
                                sentences[i] = new_sentence
                            indices.append(i)

                    ret_tuple_snt[0].append(sentences)
                    ret_tuple_snt[1].append(indices)
                    bar.next()
                ret_txt.append(ret_tuple_snt)
            self.dmgd_texts.append(ret_txt)

        # self.dump(self.dmgd_texts, "dmgd")
        bar.finish()