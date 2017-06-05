from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from smart_ml.smart_ml_general import *


class SmartTrainer:
    def __init__(self, train_x=None, train_y=None, is_auto=True):
        self.train_x = train_x
        self.train_y = train_y
        self.is_auto = is_auto

    def model_selection(self):
        model_list = [
            'LogisticRegression',
            'RandomForestClassifier',
            'RidgeClassifier',
            'GradientBoostingClassifier',
            'ExtraTreesClassifier',
            'AdaBoostClassifier',
            'XGBClassifier'
        ]
        model_map = {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'RidgeClassifier': RidgeClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'XGBClassifier': XGBClassifier()
        }
        while True:
            if self.is_auto:
                return model_list, model_map
            else:
                for i in range(0, len(model_list)):
                    print(str(i) + ":" + model_list[i])
                x = Stdin.get_int_input("Choose a model:")
            if x in range(0, len(model_list)):
                return model_list[x], model_map[model_list[x]]
            else:
                print("Error Input")

    def select_best_model(self):
        data_x, test_x, data_y, test_y = train_test_split(self.train_x, self.train_y, test_size=0.2, random_state=42)

        if self.is_auto:
            model_name_list, model_map = self.model_selection()
            best_model_name = ''
            best_model = None
            best_score = 0
            for i in range(0, len(model_name_list)):
                model = model_map[model_name_list[i]]
                model.fit(data_x, data_y)
                scores = cross_val_score(model, data_x, data_y, cv=10)
                score = sum(scores) / len(scores)
                if score > best_score:
                    best_model_name = model_name_list[i]
                    best_model = model
                    best_score = score
            return best_model_name, best_model, best_score
        else:
            model_name, model = self.model_selection()
            model.fit(data_x, data_y)
            scores = cross_val_score(model, data_x, data_y, cv=5)
            score = sum(scores) / len(scores)

            pre_y = model.predict(test_x)
            print(classification_report(test_y, pre_y))

            return model_name, model, score
