import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils

class Models:
    def __init__(self):
        self.reg = {
            'svr': SVR(),
            'gradient': GradientBoostingRegressor()
        }

        self.params = {
            'svr': {
                'kernel': ['linear', 'poly', 'rbf'],
                'gamma': ['scale', 'auto'],
                'C': [1, 5, 10]
            },
            'gradient': {
                'loss': ['absolute_error', 'huber', 'quantile', 'squared_error'],
                'learning_rate': [0.1, 0.5, 1],
            }
        }

    def grid_training(self, x, y):

        best_score = 999
        best_model = None

        for name, reg in self.reg.items():
            grid_reg = GridSearchCV(reg, self.params[name], cv=3).fit(x, y.values.ravel())
            score = np.abs(grid_reg.best_score_)

            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_


        utils = Utils()
        utils.model_export(best_model, best_score)
