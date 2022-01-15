"""Mark-Evaluate Petersen Metric Module."""

from ..Metric import Metric
from ...Tasks.Task import Task
from nubia_score import Nubia


class NUBIAMetric(Metric):
    """NUBIA Class.

    Based on Metric class.
    """

    def __init__(self):
        """Initialize."""
        super(NUBIAMetric, self).__init__()

        # Properties
        self.name: str = "NUBIA"
        self.description: str = "NeUral Based Interchangeability Assessor"
        self.submetrics: list = ["Nubia"]
        self.id: bool = False

        self.nubia: any = Nubia()

    def compute(self, ref: list, cand: list) -> list:
        """Compute NUBIA.

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
        super(NUBIAMetric, self).check_input(ref, cand)
        score: list =\
            [[self.nubia.score(ref, cand)] for ref, cand in zip(ref, cand)]
        return score

    def get_id(self, ref: str, cand: list) -> list:
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
        return [[1] * len(ref)]
