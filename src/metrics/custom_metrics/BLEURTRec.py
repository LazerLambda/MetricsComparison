from ..Metric import Metric
from ...Tasks.Task import Task
from bleurt import score

import os

class BLEURTRec(Metric):

    limits : tuple = (-1.5,1.5)

    def __init__(self):
        
        super(BLEURTRec, self).__init__()
        
        # Properties
        self.name: str = "BLEURT-Base-128"
        self.description: str = "BLEURT-Base-128, pre-trained, finetuned on WMT"
        self.submetrics: str = ["BLEURT"]
        self.id : bool = False

        # path : str = "src/bleurt/bleurt/test_checkpoint" + os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        path: str = "src/metrics/bleurt/BLEURT-20"

        # path from parent folder of src
        self.scorer_bleurt: score.BleurtScorer = score.BleurtScorer(
            checkpoint=path)

    def get_id(self, ref :list, cand : list):
        raise Exception(
            "ERROR:\n\t'-> BLEURT has to be computed.")

    def compute(self, ref : list, cand : list):
        super(BLEURTRec, self).check_input(ref, cand)
        return [self.scorer_bleurt.score(references=ref, candidates=cand)]
        
