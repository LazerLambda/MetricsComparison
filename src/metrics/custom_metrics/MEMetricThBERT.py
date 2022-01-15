"""Mark-Evaluate Petersen Metric Module."""

from ..Metric import Metric

from ..ME.markevaluate.MarkEvaluate import MarkEvaluate as ME
from ...Tasks.Task import Task


class MEMetricThBERT(Metric):
    """Mark-Evaluate Petersen Class.

    Based on Metric class.
    """

    limits: tuple = (0, 1.05)

    def __init__(self):
        """Initialize."""
        super(MEMetricThBERT, self).__init__()

        # Properties
        self.name: str = "Mark-Evaluate (Theorem based, BERT)"
        self.description: str = "ME-Petersen, ME*-Schnabel, ME*-CAPTURE"
        self.submetrics: list = ['Petersen']
        self.id: bool = True

        self.ME_scorer: ME = ME(
            sent_transf=False,
            sntnc_lvl=True,
            orig=False)

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
        return [[1] * len(cand)]

    def compute(self, ref: list, cand: list):
        """Compute Mark-Evaluate Petersen.

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
        super(MEMetricThBERT, self).check_input(ref, cand)
        score: dict = self.ME_scorer.estimate(cand=cand, ref=ref)
        return [score['Petersen']]
