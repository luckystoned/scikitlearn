import joblib
import pandas as pd

class Utils:

    def load_from_csv(self):
        return pd.read_csv('./in/felicidad.csv')

    def load_from_mysql(self):
        pass

    def feature_target(self, dataset, drop_cols, y):
        x = dataset.drop(drop_cols, axis=1)
        y = dataset[[y]].squeeze()
        return x, y
    
    def model_export(self, clf, score):
        print('Best Model Score: ', score)
        joblib.dump(clf, './models/best_model.pkl')