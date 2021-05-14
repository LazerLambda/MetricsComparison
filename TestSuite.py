import numpy as np


from markevaluate import MarkEvaluate as me
from bert_score import BERTScorer
from bleurt import score
from GRUEN import Main
from nubia.nubia_score import Nubia
from Metrics import Metrics
from Visualize import Visualize
from DamageData import DamageData


class TestSuite:

    def __init__(self, cand : list, ref : list) -> None:
        self.cand : list = cand
        self.ref : list = ref
        self.results = list()


    def run_test(self) -> None:
        # TODO corrupt data

        data = DamageData()
        data.load()

        length_data : int = np.shape(data.data_set)['train'][0]
        
        for _ in range(length_data):
            result : dict = dict()

            result['ME'] = Metrics.comp_ME(self.cand, self.ref)
            result['BERTScore'] = Metrics.comp_BERTScore(self.cand, self.ref)
            result['BLEURT'] = Metrics.comp_BLEURT(self.cand, self.ref)
            # result['GRUEN'] = Metrics.comp_GRUEN(self.cand, self.ref)
            # result['NUBIA'] = Metrics.comp_NUBIA(self.cand, self.ref)





    def visualize(self) -> None:
        pass


cand , ref = ["BLEURT is a trained metric, that is, it is a regression model trained on ratings data.",\
    # "the thing goes skrrrrya pam pam pam pam pam",\
    "and a kut tun tun boom yo"],\
    ["asd",\
    # "The thing goes crazy.",\
    "And makes some noise."]


if __name__ == "__main__":
    testSuite = TestSuite(cand, ref)
    print(testSuite.results)
    testSuite.run_test()
    print(testSuite.results) 