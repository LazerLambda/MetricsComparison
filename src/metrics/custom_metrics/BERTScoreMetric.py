"""BERTScore Metric Module."""

from ..Metric import Metric
from bert_score import BERTScorer

import torch


class BERTScoreMetric(Metric):
    """BERTScore Metric Class.

    Based on Metric class.
    """

    limits: tuple = (0, 1.05)

    def __init__(self):
        """Initialize."""
        super(BERTScoreMetric, self).__init__()

        # Properties
        self.name: str = "BERTScore"
        self.description: str = "BERTScore without idf-weighting"
        self.submetrics: list = ["R", "P", "F1"]
        self.id: bool = True

        self.scorer_bertscore: BERTScorer = BERTScorer(
            lang="en",
            model_type="microsoft/deberta-xlarge-mnli",
            rescale_with_baseline=True,
            batch_size=16,
            device='cpu',
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
        """Compute BERTScore.

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
