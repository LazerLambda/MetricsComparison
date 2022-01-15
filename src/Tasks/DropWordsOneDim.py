"""Word Drop Module."""
from .OneDim import OneDim

import copy
import math
import numpy as np
import pandas as pd
import random
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb


class DropWordsOneDim(OneDim):
    """Class for word drop task."""

    __slots__ = [
        "texts", "results", "dmgd_texts",
        "combined_results", "step_arr", "path",
        "name", "df", "descr"]

    def __init__(self, params: dict):
        """Initialize."""
        super(DropWordsOneDim, self).__init__(params=params)
        self.name = "dropped_words"
        self.descr = "Dropped words"

    @staticmethod
    def drop_single(
            sentence: str,
            doc: spacy.tokens.doc.Doc,
            step: float) -> tuple:
        """Drop words.

        Params
        ------
        sentence : str
            sentence
        doc : spacy.tokens.doc.Doc
            spacy document containing linguistic information
            about the sentence
        step : float
            fraction of how many tokens are to be dropped

        Returns
        -------
        tuple
            Tuple including sentence and success of modification
        """
        bound: float = 1 - 1 / len(doc)
        if len(doc) == 0:
            return sentence, False

        if step == 0:
            return sentence, True

        candidates: list = []
        for i in range(len(doc)):
            if doc[i].pos_ != "PUNCT":
                candidates.append(i)
            else:
                continue

        # one word must be in the sentence at least
        if step > bound:
            step = bound

        prop: float = int(math.floor(step * len(candidates)))
        drop_list: list = random.sample(candidates, k=prop)

        sent: str = ""

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

        if len(sent.strip()) == 0:
            print("Sentence empty! Word drop")
            return sent, False

        return sent, True

    def perturbate(self) -> None:
        """Perturbate sentences."""
        bar: ShadyBar = ShadyBar(
            message="Perturbating " + self.name + " ",
            max=len(self.step_arr) * len(self.texts))

        for step in self.step_arr:
            ret_tuple: tuple = ([], [])
            for _, (sentences, doc) in enumerate(self.texts):

                sentences: list = copy.deepcopy(sentences)
                indices: list = []

                for i in range(len(sentences)):

                    if len(doc[i]) < 2:
                        continue

                    new_sentence = sentences[i]
                    new_sentence, success = self.drop_single(
                        sentence=sentences[i],
                        doc=doc[i],
                        step=step)

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence

                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)
                bar.next()

            self.dmgd_texts.append(ret_tuple)

        bar.finish()
