"""POS-Drop Module."""
from .OneDim import OneDim
from .Task import Task

import copy
import math
import numpy as np
import pandas as pd
import spacy

from progress.bar import ShadyBar
from checklist.perturb import Perturb


class POSDrop(OneDim):
    """Class for POS drop task."""

    __slots__ = [
        "texts", "results", "dmgd_texts",
        "combined_results", "step_arr",
        "path", "name", "df", "descr"]

    def __init__(self, params: dict):
        """Initialize."""
        super(POSDrop, self).__init__(params=params)
        self.name: str = "posdrop"
        self.descr: str = "DROP of words with specific POS tag"

        assert 'pos list' in params.keys()
        assert isinstance(params['pos list'], list)
        self.step_arr: list = ['Original'] + params['pos list']

    def set_steps(self, steps: dict) -> Task:
        """Set steps.

        UNUSED.
        """
        pass

    @staticmethod
    def drop_single_pos(
            sentence: str,
            doc: spacy.tokens.doc.Doc,
            pos: str) -> tuple:
        """Drop POS from sentence.

        Params
        ------
        sentence : str
            sentence
        doc : spacy.tokens.doc.Doc
            spacy document containing linguistic information
            about the sentence
        pos : str
            string describing the pos to be dropped

        Returns
        -------
        tuple
            Tuple including sentence and success of modification
        """
        candidates: list = []

        for i in range(len(doc)):

            if doc[i].pos_ == pos:
                candidates.append(i)
            else:
                continue

        if len(candidates) == 0:
            return sentence, False

        diff: int = 0
        for i in candidates:
            bounds: tuple = (
                doc[i].idx - diff,
                doc[i].idx + len(doc[i].text) - diff)
            sentence: str = sentence[0:bounds[0]] + sentence[(bounds[1] + 1)::]
            diff += len(doc[i].text) + 1

        if len(sentence.strip()) == 0:
            return sentence, False

        return sentence, True

    def perturbate(self) -> None:
        """Perturbate sentences."""
        # [(degree of deterioration, deteriorated text, indices)]

        bar: ShadyBar = ShadyBar(
            message="Perturbating " + self.name + " ",
            max=len(self.step_arr) * len(self.texts))

        for pos in self.step_arr:
            ret_tuple: tuple = ([], [])
            for _, (sentences, doc) in enumerate(self.texts):

                sentences: list = copy.deepcopy(sentences)
                indices: list = []

                if pos == 'Original':
                    ret_tuple[0].append(sentences)
                    ret_tuple[1].append(indices)
                    bar.next()
                    continue

                for i in range(len(sentences)):

                    if len(doc[i]) < 2:
                        continue

                    new_sentence = sentences[i]
                    new_sentence, success = self.drop_single_pos(
                        sentence=sentences[i],
                        doc=doc[i],
                        pos=pos)

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence

                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)
                bar.next()

            self.dmgd_texts.append(ret_tuple)

        bar.finish()

    # override
    def create_table(self, metrics: list) -> None:
        """Create Table.

        Customized method.

        Params
        ------
        metrics : list
            list of metrics
        """
        data: list = []
        for i, pos in enumerate(self.step_arr):
            for metric in metrics:
                for submetric in metric.submetrics:
                    for value in \
                            self.combined_results[i][metric.name][submetric]:
                        # degree is a string here
                        scatter_struc: dict = {
                            'metric': metric.name,
                            'submetric': submetric,
                            'degree': str(pos),
                            'value': float(value)}
                        data.append(scatter_struc)

        self.df = pd.DataFrame(
            data=data,
            columns=['metric', 'submetric', 'degree', 'value'])
