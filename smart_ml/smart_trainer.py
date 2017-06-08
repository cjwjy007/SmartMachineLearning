from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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
        self.model = None
        self.model_name = None
        self.model_params = None

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

    def get_model_params(self):
        if self.model_name is None:
            return
        params_map = {
            'LogisticRegression': {
                'C': [.0001, .001, .01, .1, 1, 10, 100, 1000],
                'class_weight': [None, 'balanced'],
                'solver': ['newton-cg', 'lbfgs', 'sag'],
                'penalty': ['l1', 'l2']
            },
            'RandomForestClassifier': {
                'criterion': ['entropy', 'gini'],
                'class_weight': [None, 'balanced'],
                'max_features': ['sqrt', 'log2', None],
                'min_samples_split': [1.0, 2, 5, 20, 50, 100],
                'min_samples_leaf': [1, 2, 5, 20, 50, 100],
                'bootstrap': [True, False],
                'max_depth': [5, 8, 15, 25, 30, None],
                'n_estimators': [10, 100, 200]
            },
            'RidgeClassifier': {
                'alpha': [.0001, .001, .01, .1, 1, 10, 100, 1000],
                'class_weight': [None, 'balanced'],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
                'fit_intercept': [True, False],
                'normalize': [True, False]
            },
            'GradientBoostingClassifier': {
                'loss': ['deviance', 'exponential'],
                'max_depth': [1, 2, 3, 5],
                'max_features': ['sqrt', 'log2', None],
                'subsample': [0.5, 1.0]
            },
            'ExtraTreesClassifier': {
                'max_features': ['auto', 'sqrt', 'log2', None],
                'min_samples_split': [1, 2, 5, 20, 50, 100],
                'min_samples_leaf': [1, 2, 5, 20, 50, 100],
                'bootstrap': [True, False]
            },
            'AdaBoostClassifier': {
                'loss': ['linear', 'square', 'exponential']
            },
            'XGBClassifier': {
                'max_depth': [1, 3, 5, 7, 10, 15, 25],
                'learning_rate': [0.1],
                'min_child_weight': [1, 5, 10, 50],
                'subsample': [0.5, 0.6, 0.7, 0.8, 1.0],
                'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 1.0]
            },
        }
        return params_map[self.model_name]

    def random_grid_search(self):
        n_iter_search = 20
        random_search = RandomizedSearchCV(estimator=self.model, param_distributions=self.model_params,
                                           n_iter=n_iter_search)

        random_search.fit(self.train_x, self.train_y)
        print("best_param:", end='')
        print(random_search.best_params_)
        print("best_score:", end='')
        print(random_search.best_score_)
        self.model = random_search.best_estimator_

    def select_best_model(self):
        data_x, test_x, data_y, test_y = train_test_split(self.train_x, self.train_y, test_size=0.2,
                                                          random_state=42)

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
            self.model_name, self.model = self.model_selection()
            self.model_params = self.get_model_params()

            # scores = cross_val_score(self.model, data_x, data_y, cv=5)
            # score = sum(scores) / len(scores)

            self.model.fit(data_x, data_y)
            pre_y = self.model.predict(test_x)
            print(classification_report(test_y, pre_y))

            print("best_model_name: " + self.model_name)
            self.random_grid_search()
            return self.model_name, self.model
