"""BERTScore IDF Metric Module."""

from ..Metric import Metric
from ...Tasks.Task import Task

from datasets import load_metric

import spacy


class BLEUScoreMetric(Metric):
    """BLEU Metric Class.

    Based on Metric class.
    """

    limits: tuple = (0, 1.05)

    def __init__(self):
        """Initialize."""
        super(BLEUScoreMetric, self).__init__()

        self.name: str = "BLEU"
        self.description: str = "Bilingual evaluation understudy"
        self.submetrics: list = ["BLEU"]
        self.id: bool = True

        self.bleu_hggfc = load_metric("bleu")
        self.nlp = spacy.load("en_core_web_sm")

    def get_id(self, ref: list, cand: list):
        """Get id value.

        Params
        ------
        ref : list
            list of reference sentences
        cand : list
            list of candidate sentences

        Returns
        -------
        iterable
            list of id values
        """
        assert len(ref) == len(cand)
        return [[1]]

    def compute(self, ref: list, cand: list):
        """Compute BLEU.

        Params
        ------
        ref : list
            list of reference sentences
        cand : list
            list of candidate sentences

        Returns
        -------
        iterable
            list of computed values.
        """
        # For this experiment, only one reference sample is used
        assert len(ref) == len(cand)
        cand, ref =\
            [[token.text for token in self.nlp(str(sent))]
                for sent in cand],\
            [[[token.text for token in self.nlp(str(sent))]]
                for sent in ref]
        score = self.bleu_hggfc.compute(predictions=cand, references=ref)
        return [[score['bleu']]]
