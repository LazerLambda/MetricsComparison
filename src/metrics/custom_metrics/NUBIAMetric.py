from ..Metric import Metric
from ...Tasks.Task import Task
from nubia_score import Nubia


class NUBIAMetric(Metric):

    def __init__(self):
        super(NUBIAMetric, self).__init__()

        # Properties
        self.name: str = "NUBIA"
        self.description: str = "NeUral Based Interchangeability Assessor"
        self.submetrics: list = ["Nubia"]
        self.id: bool = False

        self.nubia: any = Nubia()

    def compute(self, ref: list, cand: list) -> list:
        super(NUBIAMetric, self).check_input(ref, cand)
        score: list = [[self.nubia.score(ref, cand)] for ref, cand in zip(ref, cand)]
        return score
            

    def get_id(self, ref: str, cand: list) -> list:
        return [[1] * len(ref)]
        