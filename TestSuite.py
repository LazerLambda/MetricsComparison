from markevaluate import MarkEvaluate as me
from bert_score import BERTScorer
from bleurt import score
from GRUEN import Main
from nubia.nubia_score import Nubia
from Metrics import Metrics


class TestSuite:

    def __init__(self, cand : list, ref : list) -> None:
        self.cand : list = cand
        self.ref : list = ref
        self.results = dict()



    # def comp_ME(self) -> None:
    #     me_main : me = me.MarkEvaluate(self.cand, self.ref)
    #     self.results["ME"] = me_main.estimate()



    # def comp_BERTScore(self) -> None:
    #     scorer_bertscore : BERTScorer = BERTScorer(lang="en")
    #     scores_bertscore : tuple = scorer_bertscore.score(self.cand, self.ref) # actual type in tuple is tensor
    #     self.results["BERTScore"] = {"P": scores_bertscore[0],\
    #         "R": scores_bertscore[1],\
    #         "F1": scores_bertscore[2]}



    # def comp_BLEURT(self) -> None:
    #     scorer_bleurt : score.BleurtScorer = score.BleurtScorer(checkpoint="bleurt/bleurt/test_checkpoint") # checkpoint
    #     self.results["BLEURT"] = scorer_bleurt.score(references=self.ref, candidates=self.cand)



    # def comp_GRUEN(self) -> None:
    #     scores_gruen = Main.get_gruen(self.cand)
    #     self.results["GRUEN"] = {"GRUEN": scores_gruen}



    # def comp_NUBIA(self) -> None:
    #     n = Nubia()
    #     self.results["NUBIA"] = list(
    #         map(
    #             lambda candref : n.score(candref[0], candref[1], verbose=False, get_features=True),
    #             zip(self.cand, self.ref)
    #             )
    #         )

    def run_test(self) -> None:
        Metrics.comp_ME(self.cand, self.ref, self.results)
        Metrics.comp_BERTScore(self.cand, self.ref, self.results)
        Metrics.comp_BLEURT(self.cand, self.ref, self.results)
        Metrics.comp_GRUEN(self.cand, self.ref, self.results)
        Metrics.comp_NUBIA(self.cand, self.ref, self.results)





    def visualize(self) -> None:
        pass


cand , ref = ["BLEURT is a trained metric, that is, it is a regression model trained on ratings data.",\
    "the thing goes skrrrrya pam pam pam pam pam",\
    "and a kut tun tun boom yo"],\
    ["asd",\
    "The thing goes crazy.",\
    "And makes some noise."]


if __name__ == "__main__":
    testSuite = TestSuite(cand, ref)
    print(testSuite.results)
    testSuite.run_test()
    print(testSuite.results) 