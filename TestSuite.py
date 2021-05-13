from markevaluate import MarkEvaluate as me
from bert_score import BERTScorer
from bleurt import score
from GRUEN import Main
from nubia.nubia_score import Nubia
from Metrics import Metrics
from Visualize import Visualize


class TestSuite:

    def __init__(self, cand : list, ref : list) -> None:
        self.cand : list = cand
        self.ref : list = ref
        self.results = dict()


    def run_test(self) -> None:
        # TODO corrupt data
        Metrics.comp_ME(self.cand, self.ref, self.results)
        Metrics.comp_BERTScore(self.cand, self.ref, self.results)
        Metrics.comp_BLEURT(self.cand, self.ref, self.results)
        Metrics.comp_GRUEN(self.cand, self.ref, self.results)
        Metrics.comp_NUBIA(self.cand, self.ref, self.results)





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