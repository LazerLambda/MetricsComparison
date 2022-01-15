"""BLEURT Metric Module."""

from ..Metric import Metric
from ...Tasks.Task import Task
from bleurt import score

import os


class BLEURTRec(Metric):
    """BERTScore Metric Class.

    Based on Metric class.
    """

    limits: tuple = (-1.5, 1.5)

    def __init__(self):
        """Initialize."""
        super(BLEURTRec, self).__init__()

        # Properties
        self.name: str = "BLEURT-Base-128"
        self.description: str =\
            "BLEURT-Base-128, pre-trained, finetuned on WMT"
        self.submetrics: str = ["BLEURT"]
        self.id: bool = False

        path: str = "src/metrics/bleurt/BLEURT-20"

        # path from parent folder of src
        self.scorer_bleurt: score.BleurtScorer = score.BleurtScorer(
            checkpoint=path)

    def get_id(self, ref: list, cand: list):
        """Get id value.

        Params
        ------
        ref : list
            list of reference sentences
        cand : list
            list of candidate sentences

        Raises
        ------
        Exception
            BLEURT must be computed.
        """
        raise Exception(
            "ERROR:\n\t'-> BLEURT has to be computed.")

    def compute(self, ref: list, cand: list) -> list:
        """Compute BLEURT.

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
        super(BLEURTRec, self).check_input(ref, cand)
        return [self.scorer_bleurt.score(references=ref, candidates=cand)]
