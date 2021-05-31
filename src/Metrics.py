from .ME.markevaluate import MarkEvaluate as me
from bert_score import BERTScorer
from bleurt import score
# from .GRUEN import Main
# from .nubia.nubia_score import Nubia

import numpy as np


class Metrics:

    def __init__(self):
        self.scorer_bertscore: BERTScorer = BERTScorer(lang="en")
        self.scorer_bleurt: score.BleurtScorer = score.BleurtScorer(
            checkpoint="src/bleurt/bleurt/test_checkpoint")
        # self.n = Nubia()

    def comp_ME(self, cand: list, ref: list) -> dict:
        me_main: me = me.MarkEvaluate(cand, ref)
        results: dict = me_main.estimate()
        del me_main
        return results

    def comp_BERTScore(self, cand: list, ref: list) -> dict:
        scores_bertscore: tuple = self.scorer_bertscore.score(
            cand, ref)  # actual type in tuple is tensor
        return {"P": scores_bertscore[0],
                "R": scores_bertscore[1],
                "F1": scores_bertscore[2]}

    def comp_BLEURT(self, cand: list, ref: list) -> dict:
        return {'BLEURT': self.scorer_bleurt.score(references=ref, candidates=cand)}

    # def comp_GRUEN(self, cand: list, ref: list) -> dict:
    #     scores_gruen = Main.get_gruen(cand)
    #     return {"GRUEN": scores_gruen}

    # def comp_NUBIA(self, cand: list, ref: list) -> dict:
    #     return list(
    #         map(
    #             lambda candref: self.n.score(
    #                 candref[0], candref[1], verbose=False, get_features=True),
    #             zip(cand, ref)
    #         )
    #     )

    @staticmethod
    def con_BERTScore(acc, elem):
        for e in elem:
            if isinstance(e, dict):
                if 'P' in e.keys() and 'R' in e.keys() and 'F1' in e.keys():
                    return{
                        'P': acc['P'] + e['P'].tolist(),
                        'R': acc['R'] + e['R'].tolist(),
                        'F1': acc['F1'] + e['F1'].tolist()
                    }

    @staticmethod
    def con_BERTScore_f(acc, elem):
        get_indices: np.ndarray = lambda elem: np.asarray(elem[len(elem) - 1])
        
        for e in elem:
            if isinstance(e, dict):
                if 'P' in e.keys() and 'R' in e.keys() and 'F1' in e.keys():
                    return {
                        'P': np.concatenate(
                            (acc['P'], np.asarray(e['P'])[get_indices(elem) if get_indices(
                                elem).size != 0 else np.arange(len(e))])
                        ),
                        'R': np.concatenate(
                            (acc['R'], np.asarray(e['R'])[get_indices(elem) if get_indices(
                                elem).size != 0 else np.arange(len(e))])
                        ),
                        'F1': np.concatenate(
                            (acc['F1'], np.asarray(e['F1'])[get_indices(elem)
                            if get_indices(elem).size != 0 else np.arange(len(e))])
                        )
                    }

    @staticmethod
    def con_BLEURT(acc, elem):
        for e in elem:
            if isinstance(e, dict):
                if 'BLEURT' in e.keys():
                    return {
                        'BLEURT': acc['BLEURT'] + e['BLEURT']
                    }

    @staticmethod
    def con_BLEURT_f(acc, elem):
        get_indices: np.ndarray = lambda elem: np.asarray(elem[len(elem) - 1])
        for e in elem:
            if isinstance(e, dict):
                if 'BLEURT' in e.keys():
                    return {
                        'BLEURT': np.concatenate(
                            (acc['BLEURT'],
                            np.asarray(e['BLEURT'])[get_indices(elem) 
                                if get_indices(elem).size != 0 
                                else np.arange(len(e))]))
                    }


    # TODO correct Petersen to Petersen
    @staticmethod
    def con_ME(acc, elem):
        for e in elem:
            if isinstance(e, dict):
                if 'Schnabel' in e.keys() and 'Petersen' in e.keys() and 'CAPTURE' in e.keys():
                    return {
                        'Petersen': acc['Petersen'] + [e['Petersen']],
                        'Schnabel': acc['Schnabel'] + [e['Schnabel']],
                        'CAPTURE': acc['CAPTURE'] + [e['CAPTURE']]
                    }



    INIT_DICT : dict = {
        'BERTScore' : dict([(name, []) for name in ['P', 'R', 'F1']]),
        'BERTScore_f' : dict([(name, []) for name in ['P', 'R', 'F1']]),
        'BLEURT' : {'BLEURT' : []},
        'BLEURT_f' : {'BLEURT' : []},
        'ME' : dict([(name, []) for name in ['Schnabel', 'Petersen', 'CAPTURE']])
    }

    BLEURT: dict = {
        'REDUCE': ['BLEURT', 'BLEURT_f'],
        'REDUCE_FUN' : {
            'BLEURT' : con_BLEURT,
            'BLEURT_f' : con_BLEURT_f
            },
        'SUB' : ['P', 'R', 'F1']
    }

    BERTSCORE: dict = {
        'REDUCE': ['BERTScore', 'BERTScore_f'],
        'REDUCE_FUN' : {
            'BERTScore' : con_BERTScore,
            'BERTScore_f' : con_BERTScore_f
        },
        'SUB' : ['BLEURT']
    }

    ME: dict = {
        'REDUCE': ['ME'],
        'REDUCE_FUN' : {
            'ME' : con_ME
        },
        'SUB' : ['Schnabel', 'Petersen', 'CAPTURE']
    }

Metrics.REDUCE_DICT : dict = {
    'BERTScore' : Metrics.con_BERTScore,
    'BERTScore_f' : Metrics.con_BERTScore_f,
    'BLEURT' : Metrics.con_BLEURT,
    'BLEURT_f' : Metrics.con_BLEURT_f,
    'ME' : Metrics.con_ME
}

Metrics.SUBMETRICS : dict = {
    'BERTScore' : ['P', 'R', 'F1'],
    'BERTScore_f' : ['P', 'R', 'F1'],
    'BLEURT' : ['BLEURT'],
    'BLEURT_f' : ['BLEURT'],
    'ME' : ['Schnabel', 'Petersen', 'CAPTURE']
}