from __future__ import annotations

import checklist
import math
import nltk
import numpy as np
import random
import spacy

from checklist.perturb import Perturb
from progress.bar import ShadyBar
from typing import Tuple



class DamageData:


    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")



    def negate_data(self, text : str, percentage : float) -> Tuple[list, list]:
        """ Perturbation function

        Complexity is O(n^2).

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
                loopCond = False

            sample = random.sample(range(len(sentences)), math.floor(percentage * len(sentences)))
            indices : list = []

            for i in sample:

                ret = None

                try:
                    ret = Perturb.perturb([doc[i]], Perturb.add_negation, keep_original=False)
                    if len(ret.data) > 0:
                        sentences[i] = ret.data[0][0]
                        indices.append(i)
                    else:
                        loopCond = False

                except Exception:
                    loopCond = True

                loopCond = False
            
            counter += 1

        return sentences, indices



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
        will be choosen.  Alteration is done inplace.
        Search for an appropriate phrase to repeat with no punctuation in 
        it from the end of the sentence and then repeat the phrase nTimes.
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

                # print(f"Repetition for a phrase with %i words %i times added. Sentence No.: %i" % (phraseLength, nTimes, sent_ind))
                return True
        return False



    def repeat_words(self, text : str, probability : float, nTimes : int = 3, phraseLength : int = 4) -> Tuple[list, list]:
        """ Repeat words function

        Function to repeat some word phrases n times. Which sentence is perturbated is determined by a random variable.
        With the passed probability, a sentence in the text is perturbated.
        Complexity is O(n^2) (function call).

        Parameter
        ---------
        text : str
            The whole text to be deteriorated
        probability : float
            value in [0,1] determining the probabilty of deterioration
        nTimes : int
            number of repetitions of each phrase
        phraseLength : int
            length of phrase to be repeated

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
        indices : list = []
        for i in range(len(sentences)):
            
            # if function returns True, repeat phrase
            if bool(np.random.binomial(size=1, n=1, p=probability)):

                tokens = self.nlp(sentences[i])

                if len(tokens) <= phraseLength:
                    continue

                success : bool = self.createRepetitions(sentences=sentences, doc=tokens, sent_ind=i, phraseLength=phraseLength, nTimes=nTimes)

                if success:
                    indices.append(i)
        
        return sentences, indices



    @staticmethod
    def swap_pair(sentence : str, doc : spacy.tokens.doc.Doc) -> Tuple[str, bool]:
        """ Swap pair function

        Function to swap one random pair of words. Using the random sample function,
        two elements are choosen and swaped later on.
        Complexity is O(n).

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

        if len(candidates) < 3:
            return sentence, False


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
            sentence[(second_bounds[1])::], True

        

    def word_swap(self, text : str, probability : float) -> Tuple[list, list]:
        """ Word swap function

        Function to swap a specific amound of words, determined by the probability parameter. A whole 
        text is passed, split into sentences and later on deteriorated to a specific degree.
        Complexity is O(n^2) (function call).

        Parameter
        ---------
        text : str
            Whole text which will be deteriorated
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
        # TODO include times in return 

        if probability < 0 or probability > 1:
            raise Exception("Probability must be a number in [0,1].")

        sentences = nltk.sent_tokenize(text)

        if len(sentences) == 0:
            return [], []

        ret_list : list = []
        indices : list = []

        for i, sentence in enumerate(list(self.nlp.pipe(sentences))):

            if bool(np.random.binomial(size=1, n=1, p= probability)):

                times : int = random.randrange(1, len(sentence)) # altered afterwards

                new_sentence = sentences[i]
                for _ in range(times):
                    new_sentence, success = self.swap_pair(sentence=new_sentence, doc=sentence)
                ret_list.append(new_sentence)

                if success:
                    indices.append(i)

            else:
                ret_list.append(sentences[i])
        return ret_list, indices



    def word_swap_by_p(self, text : str, probability : float) -> Tuple[list, list]:
        """ Word swap function

        Function to swap a specific amound of words, determined by the probability parameter for every sentence. A whole 
        text is passed, split into sentences and later on deteriorated.
        Complexity is O(n^2) (function call).

        Parameter
        ---------
        text : str
            Whole text which will be deteriorated
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
        # TODO include times in return 

        if probability < 0 or probability > 1:
            raise Exception("Probability must be a number in [0,1].")

        sentences = nltk.sent_tokenize(text)

        if len(sentences) == 0:
            return [], []

        ret_list : list = []
        indices : list = []

        for i, sentence in enumerate(list(self.nlp.pipe(sentences))):


            times : int = random.randrange(0, math.ceil(probability * 0.5 * len(sentence))) # altered afterwards

            new_sentence = sentences[i]
            for _ in range(times):
                new_sentence, success = self.swap_pair(sentence=new_sentence, doc=sentence)
            ret_list.append(new_sentence)

            if success:
                indices.append(i)
        return ret_list, indices



    @staticmethod
    def drop_single(sentence : str, doc : spacy.tokens.doc.Doc) -> list:
        """ Drop single word function

        Function to drop a single word from a sentence.
        Complexity is O(n).

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

        if len(doc) == 0:
            return sentence

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
        Complexity is O(n^2) (function call).

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

        if len(sentences) == 0:
            return [], []

        ret_list : list = []
        indices : list = []

        for i, sentence in enumerate(list(self.nlp.pipe(sentences))):

            if bool(np.random.binomial(size=1, n=1, p= probability)):

                times : int = random.randrange(1, len(sentence)) # altered afterwards

                new_sentence = sentences[i]
                for _ in range(times):
                    new_sentence = self.drop_single(sentence=new_sentence, doc=sentence)

                ret_list.append(new_sentence)
                indices.append(i)

            else:
                ret_list.append(sentences[i])
        return ret_list, indices


    
    def word_drop_by_p(self, text : str, probability : float) -> list:
        """ Word drop function

        Function to drop a specific amound of words, determined by the probability parameter in a sentence. A whole 
        text is passed, split into sentences and later every sentence is deteriorated.
        Complexity is O(n^2) (function call).

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

        if len(sentences) == 0:
            return [], []

        ret_list : list = []
        indices : list = []

        for i, sentence in enumerate(list(self.nlp.pipe(sentences))):


            times : int = random.randrange(0, math.ceil(probability * 0.5 * len(sentence))) # altered afterwards

            new_sentence = sentences[i]
            for _ in range(times):
                new_sentence = self.drop_single(sentence=new_sentence, doc=sentence)

            ret_list.append(new_sentence)
            indices.append(i)

        return ret_list, indices



    @staticmethod
    def drop_single_pos(sentence : str, doc : spacy.tokens.doc.Doc, pos : str) -> list:
        """ Drop single word function

        Function to drop words based on their POS tag. 

        Parameter
        ---------
        sentence : str
            sentence to be deteriorated
        doc : spacy.tokens.doc.Doc
            spacy document, to extract the indices of the token from
        pos : str
            pos tag whose words will be dropped

        Returns
        -------
        str
            deteriorated sentence
        """

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
        


    def pos_drop(self, text : str,  probability : float, pos : str) -> Tuple[list, list]:
        """ POS drop function

        Function to drop random words with a specific POS-Tag from a sentence.

        Parameter
        ---------
        sentences : list
            list of already tokenized sentence tokens
        probability : float
            float in [0,1]
        pos : str
            pos tag which occurences will be dropped

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
        indices : list = []

        # TODO pipe sentences
        for i, sentence in enumerate(list(self.nlp.pipe(sentences))):
                if bool(np.random.binomial(size=1, n=1, p= probability)):
                    new_sentence, success = self.drop_single_pos(sentence=sentences[i], doc=sentence, pos=pos)

                    if success:
                        indices.append(i)
                    ret_list.append(new_sentence)
                else:
                    ret_list.append(sentences[i])
    
        return  ret_list, indices