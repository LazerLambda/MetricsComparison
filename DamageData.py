import checklist
import math
import nltk
import numpy as np
import random
import spacy

from checklist.perturb import Perturb
from datasets import load_dataset
from progress.bar import ShadyBar
from typing import Tuple



class DamageData:


    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.data_set = load_dataset('cnn_dailymail', '3.0.0')



    def perturb_data(self, text : str, percentage : float) -> list:
        """ Perturbation function

        Parameters
        ----------
        text : str
            text, passed as a string.
        percentage : float
            float, in [0,1] which controls the amount of perturbated samples.
        
        Returns
        -------
        list
            list of sentences

        Raises
        ------
        Exception
            if percentage is not in the interval [0,1]
        """

        if percentage < 0 or percentage > 1:
            raise Exception("ERROR: percentage must be in [0,1].")

        sentences = nltk.sent_tokenize(text)
        doc = list(self.nlp.pipe(sentences))

        loopCond : bool = True
        maxLen : int = len(sentences)
        counter : int = 0

        while loopCond:

            if counter > maxLen:
                print("Couldn't find proper negations.")
                loopCond = False

            indices = random.sample(range(len(sentences)), math.floor(percentage * len(sentences)))

            for i in indices:

                ret = None

                try:
                    ret = Perturb.perturb([doc[i]], Perturb.add_negation, keep_original=False)
                    if len(ret.data) > 0:
                        sentences[i] = ret.data[0][0]
                        print(f"Sentence %i: Negated." % i)
                    else:
                        loopCond = False

                except TypeError:
                    print("Couldn't find proper negation. Another sentence will be tried.")
                    loopCond = True

                loopCond = False
            
            counter += 1

        return sentences



    @staticmethod
    def createRepetitions(sentences, tokens, sentence, phraseLength, nTimes):
        """
        """
        # subtract 1 because of indexing
        for i in reversed(range(phraseLength - 1, len(tokens))):
            token_slice = tokens[(i - phraseLength):i]
            if not True in [token.pos_ == 'PUNCT' for token in token_slice]:
                rep = " ".join([token.text for token in token_slice])

                further_tokens = " ".join([token.text for token in tokens[i:len(tokens)]])
                index = tokens[i].idx

                sentences[sentence] = sentences[sentence][0:index] + " " + rep + further_tokens
                print(f"Repetition for a phrase with %i words %i times added. Sentence No.: %i" % (phraseLength, nTimes, sentence))
                break

            else:
                continue



    def repeat_words(self, text : str, probability : float, nTimes : int = 3, phraseLength : int = 4) -> list:
        """
        """
        if probability > 1 or probability < 0:
            print("probability must be in [0,1]")
            return []

        sentences = nltk.sent_tokenize(text)
        for sentence in range(len(sentences)):
            
            # if function returns 1, repeat phrase
            if bool(np.random.binomial(size=1, n=1, p= probability)):

                tokens = self.nlp(sentences[sentence])

                if len(tokens) <= phraseLength:
                    continue

                self.createRepetitions(sentences=sentences, tokens=tokens, sentence=sentence, phraseLength=phraseLength, nTimes=nTimes)
        
        return sentences



    @staticmethod
    def swap_pair(sentence : str, doc : spacy.tokens.doc.Doc) -> str:
        """ Swap pair function

        Function to swap one random pair of words. Using the random sample function,
        two elements are choosen and swaped later on.

        Parameter
        ---------
        sentence : str
            sentence to be deteriorated
        doc : spacy.tokens.doc.Doc
            spacy document, to extract the indices of the tokens from

        Returns
        -------
        str
            deteriorated sentence
        """
        
        candidates : list = []
        candidates_text : list = []

        for i in range(len(doc)):

            lower_text = doc[i].text.lower()

            if doc[i].pos_ != "PUNCT" and not lower_text in candidates_text:
                candidates.append(i)
                candidates_text.append(lower_text)
            else:
                continue

        pair : list = random.sample(candidates, 2)
        first, second = (pair[0], pair[1]) if pair[0] < pair[1] else (pair[1], pair[0])
    
        first_bounds, second_bounds = \
            (doc[first].idx, doc[first].idx + len(doc[first].text)), \
            (doc[second].idx, doc[second].idx + len(doc[second].text))

        first_token, second_token = \
            sentence[first_bounds[0]:first_bounds[1]], \
            sentence[second_bounds[0]:second_bounds[1]]
        
        return sentence[0:(first_bounds[0])] + second_token + " " +\
            sentence[(first_bounds[1] + 1):(second_bounds[0])] + first_token + \
            sentence[(second_bounds[1])::]

        

    def word_swap(self, text : str, probability : float) -> list:
        """ Word swap function

        Function to swap a specific amound of words, determined by the probability parameter. A whole 
        text is passed, split into sentences and later on deteriorated to a specific degree.

        Parameter
        ---------
        text : str
            Whole text which wil be deteriorated
        probability : float
            float in [0,1]

        Raises
        ------
        Exception
            if probability is not in [0,1]

        Returns
        -------
        list
            list of sentences
        """

        if probability < 0 or probability > 1:
            raise Exception("Probability must be a number in [0,1].")

        sentences = nltk.sent_tokenize(text)

        ret_list : list = []

        for i, sentence in enumerate(sentences):

            if bool(np.random.binomial(size=1, n=1, p= probability)):

                times : int = random.randrange(1, 5)
                new_sentence = sentence

                for _ in range(times):
                    new_sentence = self.swap_pair(sentence=new_sentence, doc=self.nlp(new_sentence))

                ret_list.append(new_sentence)
                print("Sentence %i: Swapped %i times" % (i, times))

            else:
                ret_list.append(sentence)
        return ret_list


    @staticmethod
    def drop_single(sentence : str, doc : spacy.tokens.doc.Doc) -> list:
        """ Drop single word function

        Function to drop a single word from a sentence. 

        Parameter
        ---------
        sentence : str
            sentence to be deteriorated
        doc : spacy.tokens.doc.Doc
            spacy document, to extract the indices of the token from

        Returns
        -------
        str
            deteriorated sentence
        """

        candidates : list = []

        for i in range(len(doc)):

            if doc[i].pos_ != "PUNCT":
                candidates.append(i)
            else:
                continue
        
        to_drop : int = random.randrange(0, len(doc))

        bounds = doc[to_drop].idx, doc[to_drop].idx + len(doc[to_drop].text)

        return sentence[0:bounds[0]] + sentence[(bounds[1] + 1)::]
    


    def word_drop(self, text : str, probability : float) -> list:
        """ Word drop function

        Function to drop a specific amound of words, determined by the probability parameter. A whole 
        text is passed, split into sentences and later on deteriorated to a specific degree.

        Parameter
        ---------
        text : str
            Whole text which wil be deteriorated
        probability : float
            float in [0,1]

        Raises
        ------
        Exception
            if probability is not in [0,1]

        Returns
        -------
        list
            list of sentences
        """

        if probability < 0 or probability > 1:
            raise Exception("Probability must be a number in [0,1].")

        sentences = nltk.sent_tokenize(text)

        ret_list : list = []

        for i, sentence in enumerate(sentences):

            if bool(np.random.binomial(size=1, n=1, p= probability)):

                times : int = random.randrange(1, 5)
                new_sentence = sentence

                for _ in range(times):
                    new_sentence = self.drop_single(sentence=new_sentence, doc=self.nlp(new_sentence))

                ret_list.append(new_sentence)
                print("Sentence %i: Dropped %i words" % (i, times))

            else:
                ret_list.append(sentence)
        return ret_list