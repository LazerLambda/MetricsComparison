from .OneDim import OneDim


import copy
import math
import spacy

class POSDrop(OneDim):

    @staticmethod
    def drop_single_pos(sentence : str, doc : spacy.tokens.doc.Doc, pos : str) -> tuple:

        candidates : list = []

        for i in range(len(doc)):

            if doc[i].pos_ == pos:
                candidates.append(i)
            else:
                continue
        
        if len(candidates) == 0:
            return sentence, False
        
        diff : int = 0
        for i in candidates:
            bounds = doc[i].idx - diff, doc[i].idx + len(doc[i].text) - diff
            sentence = sentence[0:bounds[0]] + sentence[(bounds[1] + 1)::]
            diff += len(doc[i].text) + 1
        

        return sentence, True
    
    def perturbate(self, params: dict):

        tag : str = params['POS']

        for step in self.step_arr:
            ret_tuple : tuple = ([], []) 
            for _, (sentences, doc) in enumerate(self.texts):

                sentences : list = copy.deepcopy(sentences)
                indices : list = []

                sample : int = int(math.floor(step * len(sentences)))

                for i in range(sample):
                    new_sentence, success = self.drop_single_pos(sentence=sentences[i], doc=doc[i], pos=tag)

                    if success:
                        indices.append(i)
                        sentences[i] = new_sentence

                ret_tuple[0].append(sentences)
                ret_tuple[1].append(indices)

            self.dmgd_texts.append(ret_tuple)
    
