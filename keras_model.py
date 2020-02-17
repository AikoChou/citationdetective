from keras.models import load_model
from multiprocessing import Lock
from multiprocessing.managers import BaseManager

import config
cfg = config.get_localized_config()

class KerasModel():
    def __init__(self):
        self.mutex = Lock()
        self.model = None

    def initialize(self):
        self.model = load_model(cfg.model_path)
        self.model._make_predict_function()

    def predict(self, input):
        with self.mutex:
            return self.model.predict(input)

class KerasManager(BaseManager):
    pass

KerasManager.register('KerasModel', KerasModel)

if __name__ == '__main__':
    import os
    import numpy as np

    X = np.load('X.npy')
    sections = np.load('sections.npy')
    with KerasManager() as manager:
        print('Main', os.getpid())
        kerasmodel = manager.KerasModel()
        kerasmodel.initialize()
        print(kerasmodel.predict([X, sections]))
