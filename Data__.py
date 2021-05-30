from __future__ import annotations

import checklist
import math
import nltk
import numpy as np
import pymongo
import random
import spacy

from checklist.perturb import Perturb
from datasets import load_dataset, dataset_dict
from progress.bar import ShadyBar
from typing import Tuple


class Data:

    dataset_properties : dict = {
        'cnn_dlml' : {
            'name' : 'cnn_dailymail',
            'version' : '3.0.0',
            'cutoff_fun' : lambda elem : {'article': elem['article'], '_id': elem['id']},
            'text' : 'article',
            'database_name' : 'cnn_data_combined'
        }
    }
    


    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.exp_data : dict = {
            'original' : [],
            'negated' : [],
            'word_drop' : [],
            'POS_drop' : [],
            'word_swap' :[]
        }
        self.data : list = list()
        self.length : int = 0



    def load_data(self) -> Data:
        self.data_set = load_dataset(\
            self.dataset_properties['cnn_dlml']['name'],\
            self.dataset_properties['cnn_dlml']['version'])
        self.data = self.concat_data_set(\
            self.data_set,\
            self.dataset_properties['cnn_dlml']['cutoff_fun'])
        self.length = \
            len(self.data_set['train']) + \
            len(self.data_set['validation']) + \
            len(self.data_set['test'])  ## TODO remove
        return self

    

    def create_exp_data(self) -> Data:

        if len(self.data) == 0:
            raise Exception("ERROR: Data is empty. Call load_data() first.")

        mdb_client = pymongo.MongoClient("mongodb://localhost:27017/")
        emdb = mdb_client["EvalMetrics"]

        if self.dataset_properties['cnn_dlml']['database_name'] in emdb.list_collection_names():
            return self
        else:
            # Create db collection
            col = emdb[self.dataset_properties['cnn_dlml']['database_name']]

            bar = ShadyBar('Writing original texts.', max=len(self.data))

            for entry in self.data:
                sentences : list = self.tokenize_sentences(entry[self.dataset_properties['cnn_dlml']['text']])
                self.exp_data['original'].append({
                    'text' : sentences,
                    'id' : entry['id']
                    })
                bar.next()
            bar.finish()

            col.insert_many(self.exp_data['original'])
            print("Inserted original texts to database.")

            bar = ShadyBar('Writing negated texts.', max=len(self.data))

            for entry in self.data:
                corrupted, not_corrupted = self.negate_data(sentences=sentences)
                self.exp_data['negated'].append({
                    'text': corrupted,
                    'not corrupted': not_corrupted,
                    'id' : entry['id']
                })
                bar.next()
            bar.finish()

            return self



    def negate_data(self, sentences : list) -> Tuple[list, list]:
        """ Perturbation function

        Complexity is O(n^2).

        Parameters
        ----------
        text : str
            text, passed as a string.
        
        Returns
        -------
        list
            list of sentences

        Raises
        ------
        Exception
            if percentage is not in the interval [0,1]
        """
        return_list : list = list()
        not_negated : list = list()

        for index, sentence in enumerate(list(self.nlp.pipe(sentences))):
            sent_tmp : str = sentence.text
            try:
                ret = Perturb.perturb([sentence], Perturb.add_negation, keep_original=False)
                if len(ret.data) > 0:
                    sent_tmp = ret.data[0][0]
                else:
                    not_negated.append(index)

            except Exception:
                not_negated.append(index)

            return_list.append(sent_tmp)


        return return_list, not_negated 


    



    def repeat_words(self, sentences : list, nTimes : int = 3, phraseLength : int = 4) -> Tuple[list, list]:
        """ Repeat wordssfunction

        Function to repeats some words in every sentence of the text. With the passed probability, a sentence in the text is perturbated.
        Complexity is O(n^2) (function call).

        Parameter
        ---------
        sentences : list
            list of already tokenized sentence tokens
        nTimes : int
            number of repetitions of each phrase
        phraseLength : int
            length of phrase to be repeated
        Returns
        -------
        list, list
            list of sentences, list of sentences where a phrase is repeated

        """

        indices : list = []
        for i in range(len(sentences)):
            
            tokens = self.nlp(sentences[i])

            if len(tokens) <= phraseLength:
                continue

            if self.createRepetitions(sentences=sentences, doc=tokens, sent_ind =i, phraseLength=phraseLength, nTimes=nTimes):
                indices.append(i)

        return sentences, indices


    ### STATIC METHODS
    @staticmethod
    def concat_data_set(data_set : dataset_dict.DatasetDict, cutoff_fun : callable) -> list:

        data : list = list()

        bar = ShadyBar('Creating dataset', max=len(data_set['train']) + len(data_set['validation']) + len(data_set['test']))

        for e in data_set['train']:
            data.append(
                cutoff_fun(e)
            )
            bar.next()

        for e in data_set['validation']:
            data.append(
                cutoff_fun(e)
            )
            bar.next()

        for e in data_set['test']:
            data.append(
                cutoff_fun(e)
            )
            bar.next()

        bar.finish()

        return data
    


    @staticmethod   
    def createRepetitions(\
            sentences : list,\
            doc : spacy.tokens.doc.Doc,\
            sent_ind : int,\
            phraseLength : int,\
            nTimes : int) -> bool:
        """ Creating Repetitions in one sentence

        Function to create repetitions in one sentence. To avoid 
        the repitition of punctations, only phrase without punctuations 
        will be choosen. Alteration is done inplace.
        Complexity is O(n).

        Parameter
        ---------
        sentences : list
            list of sentences in which one sentence will be perturbated
        doc : pacy.tokens.doc.Doc
            parsed tokens as a spaCy doc
        sent_ind : int
            index of sentence to be perturbated
        phraseLength : int
            length of a phrase to be repeated
        nTimes : int
            number of times the phrase will be repeated
        """
        # subtract 1 because of indexing
        for i in reversed(range(phraseLength - 1, len(doc))):
            token_slice = doc[(i - phraseLength):i]
            if not True in [token.pos_ == 'PUNCT' for token in token_slice]:

                index = doc[i].idx

                rep = " ".join([token.text for token in token_slice])
                further_tokens = " ".join([token.text for token in doc[i:len(doc)]])
                sentences[sent_ind ] = sentences[sent_ind ][0:index] + " " + rep + further_tokens

                #print(f"Repetition for a phrase with %i words %i times added. Sentence No.: %i" % (phraseLength, nTimes, sent_ind))
                return True
        return False



    @staticmethod
    def tokenize_sentences(text : str) -> list:
        return nltk.sent_tokenize(text)


if __name__ == "__main__":
    d = Data().load_data().create_exp_data()
    #d.create_exp_data()
