#!/usr/bin/env python3

from multiprocessing import Lock
from multiprocessing.managers import BaseManager

import config
from utils import *

import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

cfg = config.get_localized_config()

class KerasModel():
    '''
    A class for using the Citation Needed model in multiprocess environment.

    Refer to https://github.com/Dref360/tuto_keras_web
    '''
    def __init__(self):
        self.mutex = Lock()
        self.vocab_w2v = pickle.load(open(cfg.vocb_path, 'rb'))
        self.section_dict = pickle.load(open(cfg.section_path, 'rb'), encoding='latin1')
        self.model = load_model(cfg.model_path)

        # Keras builds the necessary functions using _make_predict_function
        # the first time we call predict(). This isn't safe to call predict
        # from several threads/processes, so we need to call this function
        # before spawning processes.
        # See https://stackoverflow.com/questions/43136293/
        self.model._make_predict_function()

    def predict(self, input):
        with self.mutex:
            return self.model.predict(input)

    def run_citation_needed(self, sentences):
        # Prepare the input data for the model
        X = []
        sections = []
        max_len = cfg.word_vector_length
        for text, _, section in sentences:
            # TODO: Replacing text_to_word_list with tokenization
            # /stemming/lemming functions from established Python
            # packages where possible, so we don't have to maintain
            # our own versions per language..
            # See https://github.com/AikoChou/citationdetective/issues/2
            wordlist = text_to_word_list(text)
            X_inst = []
            for word in wordlist:
                if max_len != -1 and len(X_inst) >= max_len:
                    break
                # Construct word vectors
                X_inst.append(self.vocab_w2v.get(word, self.vocab_w2v['UNK']))
            X.append(X_inst)
            sections.append(self.section_dict.get(section, 0))
        # Get the final input data by padding all word vectors to max_len
        X = pad_sequences(X, maxlen=max_len, value=self.vocab_w2v['UNK'], padding='pre')
        sections = np.array(sections)
        # Perform inference using the model
        sentences_score = self.predict([X, sections])
        return sentences_score

class KerasManager(BaseManager):
    pass

KerasManager.register('KerasModel', KerasModel)
