from sklearn.model_selection import train_test_split
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import auc
from sklearn import cluster
from xgboost import XGBClassifier
from smart_ml.smart_ml_general import *
import pandas as pd


class SmartFeatureSelection:
    def __init__(self, train_x=None, train_y=None, is_auto=True):
        self.train_x = train_x
        self.train_y = train_y
        self.is_auto = is_auto
        self.selector = None
        self.mask = []

    def get_selector(self):
        selector_list = [
            'select_k_best',
            'select_from_model',
            'auc_selector'
        ]
        selector_map = {
            selector_list[0]: SelectKBest(k=7),
            selector_list[1]: SelectFromModel(RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=15),),
                                              # threshold='10*mean'),
            selector_list[2]: SelectKBest(k=7)
            # todo
        }
        while True:
            if self.is_auto:
                return selector_map[selector_list[1]]
            else:
                for i in range(0, len(selector_list)):
                    print(str(i) + ":" + selector_list[i])
                x = Stdin.get_int_input("Choose a selector:")
                if x in range(0, len(selector_list)):
                    return selector_map[selector_list[x]]
                else:
                    print("Error Input")

    def feature_selection(self):
        self.selector = self.get_selector()
        self.selector.fit(self.train_x, self.train_y)
        return self.train_x.loc[:, self.get_mask()]

    def get_mask(self):
        return self.selector.get_support()

    def auc_select(self):
        x_train, x_valid, y_train, y_valid = train_test_split(self.train_x, self.train_y,
                                                              test_size=0.5, random_state=1)
        very_simple_learner = GradientBoostingClassifier(n_estimators=20, max_features=1, max_depth=3,
                                                         min_samples_leaf=100, learning_rate=0.1,
                                                         subsample=0.65, loss='deviance', random_state=1)

        single_feature_table = pd.DataFrame(index=range(len(x_train.columns)), columns=['feature', 'AUC'])
        for k, feature in enumerate(x_train.columns):
            train_input_feature = x_train[feature].values.reshape(-1, 1)
            valid_input_feature = x_valid[feature].values.reshape(-1, 1)
            very_simple_learner.fit(train_input_feature, y_train)

            valid_auc = auc(y_valid, very_simple_learner.predict_proba(valid_input_feature)[:, 1])
            single_feature_table.ix[k, 'feature'] = feature
            single_feature_table.ix[k, 'AUC'] = valid_auc
            single_feature_table = single_feature_table.sort_values(by='AUC', axis=0, ascending=False).reset_index(
                drop=True)

        num_features_in_combination = 5
        num_combinations = 400
        num_best_single_features_to_select_from = 20

        x_train, x_valid, y_train, y_valid = train_test_split(self.train_x, self.train_y,
                                                              test_size=0.5, random_state=1)
        weak_learner = GradientBoostingClassifier(n_estimators=30, max_features=2, max_depth=3,
                                                  min_samples_leaf=100, learning_rate=0.1,
                                                  subsample=0.65, loss='deviance', random_state=1)

        features_to_use = single_feature_table.ix[0:num_best_single_features_to_select_from - 1, 'feature']
        feature_column_names = ['feature' + str(x + 1) for x in range(num_features_in_combination)]
        feature_combinations_table = pd.DataFrame(index=range(num_combinations),
                                                  columns=feature_column_names + ['combinedAUC'])

        # for num_combinations iterations
        for combination in range(num_combinations):
            # generate random feature combination
            random_selection_of_features = sorted(
                np.random.choice(len(features_to_use), num_features_in_combination, replace=False))

            # store the feature names
            combination_feature_names = [features_to_use[x] for x in random_selection_of_features]
            for i in range(len(random_selection_of_features)):
                feature_combinations_table.ix[combination, feature_column_names[i]] = combination_feature_names[i]

            # build features matrix to get the combination AUC
            train_input_features = x_train.ix[:, combination_feature_names]
            valid_input_features = x_valid.ix[:, combination_feature_names]
            # train learner
            weak_learner.fit(train_input_features, y_train)
            # store AUC results
            valid_auc = auc(y_valid, weak_learner.predict_proba(valid_input_features)[:, 1])
            feature_combinations_table.ix[combination, 'combinedAUC'] = valid_auc
        feature_combinations_table = feature_combinations_table.sort_values(by='combinedAUC', axis=0,
                                                                            ascending=False).reset_index(drop=True)
        combination_overlap_matrix = np.zeros((num_combinations, num_combinations))
        num_features_to_select = 15

        cluserer = cluster.KMeans(n_clusters=num_features_to_select)
        cluster_inds = cluserer.fit_predict(combination_overlap_matrix)

        # %% reorder features according to their new clusters

        # group the rows into clusters
        clustered_rows = {}
        cluster_max_auc = {}
        cluster_max_ind = {}
        for clusterInd in np.unique(cluster_inds):
            clustered_rows[clusterInd] = combination_overlap_matrix[cluster_inds == clusterInd, :]
            cluster_max_auc[clusterInd] = feature_combinations_table.ix[cluster_inds == clusterInd, 'combinedAUC'].max(
                axis=0)
            cluster_max_ind[clusterInd] = feature_combinations_table.ix[
                cluster_inds == clusterInd, 'combinedAUC'].idxmax(
                axis=0)

        import operator
        sorted_clusters_by_max_auc_tuple = sorted(cluster_max_auc.items(), key=operator.itemgetter(1), reverse=True)

        # calculate the reordering vector
        final_features_to_keep = []
        reordered_vector = None
        for k, item in enumerate(sorted_clusters_by_max_auc_tuple):
            if k == 0:
                reordered_vector = np.array((cluster_inds == item[0]).nonzero())
            else:
                reordered_vector = np.hstack((reordered_vector, np.array((cluster_inds == item[0]).nonzero())))
            final_features_to_keep.append(cluster_max_ind[item[0]])
        reordered_vector = reordered_vector.flatten()

        # reorder the matrix by rows and columns
        reordered_matrix = combination_overlap_matrix[reordered_vector, :]
        reordered_matrix = reordered_matrix[:, reordered_vector]

        print(feature_combinations_table.ix[final_features_to_keep, :])
