"""MoverScore Metric.

23. Nov. 2021

Philipp Koch
"""
from ..Metric import Metric

from moverscore_v2 import get_idf_dict, word_mover_score

import numpy as np
import torch

class MoverScoreMetric(Metric):
    """Class to organize Mover-ScoreMetric."""

    def __init__(self):
        """Initialize class and parent class."""
        super(MoverScoreMetric, self).__init__()

        # Properties
        self.name: str = "MoverScore"
        self.description: str = "MoverScore DistilBERT"
        self.submetrics: list = ["MoverScore"]
        self.id: bool = False

    def set_exp(self, exp):
        """Set experiment to class attribute."""
        super(MoverScoreMetric, self).set_exp(exp)

    def get_id(self, ref: list, cand: list):
        """Throw exception if WMD is not computed for id."""
        raise Exception(
            "ERROR:\n\t'-> MoverScore must be computed.")

    def compute(self, ref: list, cand: list) -> list:
        """Compute Word Mover Score.

        IDF-Scores are computed on each sentence, since the corpus
        consists of one sentence each.

        Returns
        -------
        scores : list
            list of list with the computed scores for each sentence.
        """
        super(MoverScoreMetric, self).check_input(ref, cand)

        idf_dict_ref: dict = get_idf_dict(ref)
        idf_dict_cnd: dict = get_idf_dict(cand)

        scores: list = word_mover_score(
            ref,
            cand,
            idf_dict_ref,
            idf_dict_cnd,
            n_gram=1,
            remove_subwords=False)
        torch.cuda.empty_cache()
        return [scores]
