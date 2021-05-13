from markevaluate import MarkEvaluate as me
from bert_score import BERTScorer
from bleurt import score
from GRUEN import Main
from nubia.nubia_score import Nubia



class Metrics:
    
    @staticmethod
    def comp_ME(cand : list, ref : list) -> dict:
        me_main : me = me.MarkEvaluate(cand, ref)
        return me_main.estimate()


    @staticmethod
    def comp_BERTScore(cand : list, ref : list) -> dict:
        scorer_bertscore : BERTScorer = BERTScorer(lang="en")
        scores_bertscore : tuple = scorer_bertscore.score(cand, ref) # actual type in tuple is tensor
        return {"P": scores_bertscore[0],\
            "R": scores_bertscore[1],\
            "F1": scores_bertscore[2]}


    @staticmethod
    def comp_BLEURT(cand : list, ref : list) -> dict:
        scorer_bleurt : score.BleurtScorer = score.BleurtScorer(checkpoint="bleurt/bleurt/test_checkpoint") # checkpoint
        return {'BLEURT': scorer_bleurt.score(references=ref, candidates=cand)}


    @staticmethod
    def comp_GRUEN(cand : list, ref : list) -> dict:
        scores_gruen = Main.get_gruen(cand)
        return {"GRUEN": scores_gruen}


    @staticmethod
    def comp_NUBIA(cand : list, ref : list) -> dict:
        n = Nubia()
        return list(
            map(
                lambda candref : n.score(candref[0], candref[1], verbose=False, get_features=True),
                zip(cand, ref)
                )
            )
