from .Metric import Metric

from ..ME.markevaluate.MarkEvaluate import MarkEvaluate as ME
from ..Tasks.Task import Task
from ..Tasks.OneDim import OneDim
from ..Tasks.TwoDim import TwoDim

import seaborn as sns

class MEMetricThBERT(Metric):
    
    limits : tuple = (0,1.05)

    def __init__(self):
        
        super(MEMetricThBERT, self).__init__()

        # Properties
        self.name: str = "Mark-Evaluate (Theorem based, BERT)"
        self.description: str = "ME-Petersen, ME*-Schnabel, ME*-CAPTURE"
        self.submetrics : list = ['Petersen', 'Schnabel_qul', 'Schnabel_div', 'CAPTURE']
        self.id : bool = True

        self.ME_scorer : ME = ME(sent_transf=False, sntnc_lvl=True, orig=False)

        palette = sns.color_palette(None, 4)

        self.color : dict = {
            'Petersen' : palette[0],
            'Schnabel_qul' : palette[1],
            'Schnabel_div' : palette[2],
            'CAPTURE' : palette[3]
        }

    def get_id(self, ref :list, cand : list):
        return ([1] * len(cand), [1] * len(cand), [1] * len(cand), [1] * len(cand))

    def compute(self, ref : list, cand : list):
        super(MEMetricThBERT, self).check_input(ref, cand)
        score : dict = self.ME_scorer.estimate(cand=cand, ref=ref)
        return (score['Petersen'], score['Schnabel_qul'], score['Schnabel_div'], score['CAPTURE'])

    @staticmethod
    def concat(acc, elem):
        pass

    def get_vis_info(self, t : Task) -> dict():
        if isinstance(t, OneDim):
            return dict()

        if isinstance(t, TwoDim):
            return {
                'color': None,
                'vmin' : 0,
                'vmax' : 1
            }
        
        return None
