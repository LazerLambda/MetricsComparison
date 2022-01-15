"""BERTScore IDF Metric Module."""

from ..Metric import Metric
from bert_score import BERTScorer

from itertools import chain

import torch


class BERTScoreIDFMetric(Metric):
    """BERTScore IDF Metric Class.

    Based on Metric class.
    """

    limits: tuple = (0, 1.05)

    def __init__(self):
        """Initialize."""
        super(BERTScoreIDFMetric, self).__init__()

        # Properties
        self.name: str = "BERTScore (idf)"
        self.description: str = "BERTScore with idf-weighting"
        self.submetrics: list = ["R", "P", "F1"]
        self.id: bool = True

        self.scorer_bertscore: BERTScorer = None

    def set_exp(self, exp):
        """Set experiment.

        Params
        ------
        exp : Experiment
            Experiment to be set as a class var
        """
        super(BERTScoreIDFMetric, self).set_exp(exp)
        self.init_idf(
            [sentences for sentences, _ in self.experiment.texts])

    def init_idf(self, ref: list):
        """Initialize idf-weights.

        Reference corpus is passed as list of lists of strings
        from which the idf-weigthts are computed.

        Params
        ------
        ref : list
            list of reference sentences
        """
        assert len(ref) != 0,\
            "Reference list is empty."
        assert isinstance(ref, list),\
            f"Passed value: {ref} is not of type 'list'"
        assert isinstance(ref[0], list),\
            f"First item: {ref[0]} of ref is not of type list"

        ref: list = list(chain.from_iterable(ref))
        self.scorer_bertscore: BERTScorer = BERTScorer(
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            idf=True,
            idf_sents=ref,
            batch_size=1,
            device='cpu',
            rescale_with_baseline=True,
            use_fast_tokenizer=False)

    def get_id(self, ref: list, cand: list) -> list:
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
        return ([1] * len(ref), [1] * len(ref), [1] * len(ref))

    def compute(self, ref: list, cand: list) -> list:
        """Compute BERTScore IDF.

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
        assert len(ref) == len(cand)
        result = self.scorer_bertscore.score(cand, ref)
        torch.cuda.empty_cache()
        return result
