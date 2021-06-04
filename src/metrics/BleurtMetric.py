from .Metric import Metric
from bleurt import score

import os

class BleurtMetric(Metric):

    def __init__(self, name : str, description : str, submetric : list):
        
        
        # path : str = "src/bleurt/bleurt/test_checkpoint" + os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        path : str = "src/bleurt/bleurt/test_checkpoint"

        # path from parent folder of src
        self.scorer_bleurt: score.BleurtScorer = score.BleurtScorer(
            checkpoint=path)

        super(BleurtMetric, self).__init__(name, description, submetric)

    def compute(self, ref : list, cand : list):
        super(BleurtMetric, self).check_input(ref, cand)
        return [self.scorer_bleurt.score(references=ref, candidates=cand)]

    @staticmethod
    def concat(acc, elem):
        for e in elem:
            if isinstance(e, dict):
                if 'BLEURT' in e.keys():
                    return {
                        'BLEURT': acc['BLEURT'] + e['BLEURT']
                    }

    def init_dict(self):
        return dict(zip(self._submetrics, [[] for _ in self._submetrics]))