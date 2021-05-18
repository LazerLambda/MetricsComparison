from markevaluate import MarkEvaluate as me
from bert_score import BERTScorer
from bleurt import score
from GRUEN import Main
from nubia.nubia_score import Nubia



class Metrics:

    def __init__(self):
        self.scorer_bertscore : BERTScorer = BERTScorer(lang="en")
        self.scorer_bleurt : score.BleurtScorer = score.BleurtScorer(checkpoint="bleurt/bleurt/test_checkpoint")
        self.n = Nubia()
        


    def comp_ME(self, cand : list, ref : list) -> dict:
        me_main : me = me.MarkEvaluate(cand, ref)
        results : dict = me_main.estimate()
        del me_main
        return results



    def comp_BERTScore(self, cand : list, ref : list) -> dict:
        scores_bertscore : tuple = self.scorer_bertscore.score(cand, ref) # actual type in tuple is tensor
        return {"P": scores_bertscore[0],\
            "R": scores_bertscore[1],\
            "F1": scores_bertscore[2]}



    def comp_BLEURT(self, cand : list, ref : list) -> dict:
        return {'BLEURT': self.scorer_bleurt.score(references=ref, candidates=cand)}



    def comp_GRUEN(self, cand : list, ref : list) -> dict:
        scores_gruen = Main.get_gruen(cand)
        return {"GRUEN": scores_gruen}



    def comp_NUBIA(self, cand : list, ref : list) -> dict:
        return list(
            map(
                lambda candref : self.n.score(candref[0], candref[1], verbose=False, get_features=True),
                zip(cand, ref)
                )
            )
