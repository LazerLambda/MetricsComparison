from ..Metric import Metric

from ..ME.markevaluate.MarkEvaluate import MarkEvaluate as ME
from ...Tasks.Task import Task

class MEMetricThSBERT(Metric):
    
    limits : tuple = (0,1.05)

    def __init__(self):
        
        super(MEMetricThSBERT, self).__init__()

        # Properties
        self.name: str = "Mark-Evaluate (Theorem based, SBERT)"
        self.description: str = "ME-Petersen, ME*-Schnabel, ME*-CAPTURE"
        self.submetrics : list = ['Petersen', 'Schnabel_qul', 'Schnabel_div', 'CAPTURE']
        self.id : bool = True

        self.ME_scorer : ME = ME(sent_transf=True, orig=False)

    def get_id(self, ref :list, cand : list):
        """Return id value for each word and each metric."""
        return ([1], [1], [1], [1])

    def compute(self, ref : list, cand : list):
        super(MEMetricThSBERT, self).check_input(ref, cand)
        score : dict = self.ME_scorer.estimate(cand=cand, ref=ref)
        return ([score['Petersen']], [score['Schnabel_qul']], [score['Schnabel_div']], [score['CAPTURE']])
