from ..Metric import Metric

from ..ME.markevaluate.MarkEvaluate import MarkEvaluate as ME
from ...Tasks.Task import Task

class MEMetricOrgBERT(Metric):
    
    limits : tuple = (0,1.05)

    def __init__(self):
        
        super(MEMetricOrgBERT, self).__init__()

        # Properties
        self.name: str = "Mark-Evaluate (Original, BERT)"
        self.description: str = "ME-Petersen, ME-Schnabel, ME-CAPTURE"
        self.submetrics : list = ['Petersen', 'Schnabel_qul', 'Schnabel_div', 'CAPTURE']
        self.id : bool = False

        self.ME_scorer : ME = ME(sent_transf=False, sntnc_lvl=True, orig=True)

    def get_id(self, ref :list, cand : list):
        return ([1] * len(cand), [1] * len(cand), [1] * len(cand), [1] * len(cand))

    def compute(self, ref : list, cand : list):
        super(MEMetricOrgBERT, self).check_input(ref, cand)
        score : dict = self.ME_scorer.estimate(cand=cand, ref=ref)
        return (score['Petersen'], score['Schnabel_qul'], score['Schnabel_div'], score['CAPTURE'])
