import random

import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import sys
from io import StringIO

# from auto_ml import *
pd.options.mode.chained_assignment = None


class Stdin:
    @staticmethod
    def input_stdin(text):
        old_stdin = sys.stdin
        sys.stdin = StringIO(text)
        return old_stdin

    @staticmethod
    def recover_stdin(old_stdin):
        sys.stdin = old_stdin

    @staticmethod
    def get_int_input(text):
        try:
            x = int(input(text))
        except ValueError as e:
            return -1
            pass
        return x


class SmartML:
    handle_missing_value_str = """
            选择缺失值处理方式
            0:自定义填充缺失值
            1:填入均值
            2:填入出现次数最多的值
            3:使用模型预测
            4:删除这一列
            5:不处理"""

    handle_cate_encode_str = """
            0.为它顺序编码
            1.为它二进制编码
            2.不处理"""

    handle_num_encode_str = """
            0.为它进行标准化
            1.为它进行规范化
            2.不处理"""

    def __init__(self, train_df=None):
        self.train_df = train_df
        pass

    def is_categorical_data(self, dtype):
        if dtype is np.dtype(np.object):
            return True
        else:
            return False

    def auto_learn(self, preprocessed_df=None):

        if preprocessed_df is None:
            # columns info
            columns_list = self.train_df.columns.values.tolist()
            self.print_columns(columns_list)

            # pick predict column
            y_index = self.pick_y(columns_list)

            # preprocess Y
            self.data_y_preprocessing(train_columns=self.train_df, y_index=y_index, is_auto=True)

            # extract Y column
            preprocessed_y = self.train_df.iloc[:, y_index]
            self.train_df.drop(self.train_df.columns[y_index], axis=1, inplace=True)

            # preprocess X
            preprocessed_x = self.data_x_preprocessing(train_columns=self.train_df, pre_column=preprocessed_y,
                                                       is_auto=True)

            # save the Dataframe as csv
            pd.concat([preprocessed_y, preprocessed_x], axis=1).to_csv("preprocessed_df.csv")
        else:
            # get x and y
            preprocessed_y = preprocessed_df.iloc[:, 1]
            preprocessed_x = preprocessed_df.drop(preprocessed_df.columns[0], axis=1)
            preprocessed_x = preprocessed_x.drop(preprocessed_x.columns[0], axis=1)

        # train data
        best_model_name, best_model, best_score = self.select_best_model(data_x=preprocessed_x, data_y=preprocessed_y,
                                                                         is_auto=False)
        print("best_model_name: " + best_model_name)
        print("best_score： " + str(best_score))

    def data_y_preprocessing(self, train_columns, y_index, is_auto):
        self.transform_y(train_df=train_columns, index_y=y_index)
        self.data_encoding_y(train_df=train_columns, index_y=y_index)

    def transform_y(self, train_df, index_y):
        column_y = train_df.iloc[:, index_y]
        values_y = column_y.value_counts()
        column_y_map = {}
        for idx, val in enumerate(values_y):
            x = Stdin.get_int_input("map " + values_y.index[idx] + " to(if input -1,delete column):")
            if x == -1:
                column_y_map[values_y.index[idx]] = np.nan
            else:
                column_y_map[values_y.index[idx]] = x
        train_df.iloc[:, index_y] = column_y.map(column_y_map)
        train_df.dropna(subset=[column_y.name], inplace=True)
        train_df.index = range(len(train_df))

    def data_encoding_y(self, train_df, index_y):
        column_y = self.train_df.iloc[:, index_y]
        if self.is_categorical_data(column_y.dtype):
            self.train_df.iloc[:, index_y] = pd.Series(preprocessing.LabelEncoder().fit_transform(column_y),
                                                       name=column_y.name)

    def data_x_preprocessing(self, train_columns, pre_column, is_auto):
        train_columns = self.feature_pre_selection_x(train_columns, is_auto)
        train_columns = self.fill_missing_x(train_columns, is_auto)
        train_columns = self.data_encoding_x(train_columns, is_auto)
        train_columns = self.feature_selection_x(train_columns, pre_column)
        return train_columns

    def feature_pre_selection_x(self, train_columns, is_auto):
        delete_columns_name = []
        for i in range(0, len(train_columns.columns.values.tolist())):
            column = train_columns.iloc[:, i]
            column_notnull = column[column.notnull()]
            if column_notnull.size <= 10:
                delete_columns_name.append(column_notnull.name)
                continue
            # categorical
            if self.is_categorical_data(column.dtype):
                # values are totally different
                if column_notnull.value_counts().values.size / column_notnull.size > 0.8:
                    print("数据 {name}中的值几乎不相同，自动删除".format(name=column.name))
                    delete_columns_name.append(column_notnull.name)
                    continue
            # numerical
            else:
                # values are totally different
                if column_notnull.size == column_notnull.value_counts().values.size:
                    if is_auto:
                        if (column_notnull[0] + column_notnull[
                                column_notnull.size - 1]) * column_notnull.size / 2 == sum(column_notnull):
                            print("数据 {name}中的值为等差数列，可能为ID，自动删除".format(name=column.name))
                            delete_columns_name.append(column_notnull.name)
                    else:
                        print("数据 {name}中的值完全不相同，是否删除？".format(name=column.name))
                        if True if input("0: delete\n1:reserve") == str(0) else False:
                            delete_columns_name.append(column_notnull.name)
                    continue

        for delete_column_name in delete_columns_name:
            train_columns.drop(delete_column_name, axis=1, inplace=True)

        return train_columns

    def fill_missing_x(self, train_columns, is_auto):
        delete_columns_name = []
        for i in range(0, len(train_columns.columns.values.tolist())):
            train_column = train_columns.iloc[:, i]
            # 不存在缺失值
            if not train_column.isnull().any():
                print("数据 {name} 无缺失值".format(name=train_column.name))
                continue

            data_notnull = train_column[train_column.notnull()]
            # 判断缺失值比例
            if data_notnull.size / train_column.size < 0.1:
                print("数据 {name}中的缺失值比例大于90%，建议删除".format(name=train_column.name))

            # 选择缺失值处理方式
            print("数据 {name} 存在缺失值".format(name=train_column.name))
            while True:
                if is_auto:
                    # missing rate > 70%
                    if data_notnull.size / train_column.size < 0.3:
                        x = 4
                    else:
                        # categorical value
                        if self.is_categorical_data(data_notnull.dtype):
                            train_column.loc[train_column.isnull()] = 'nan_mark'
                            break
                        # numerical value
                        else:
                            # filling in the mean value or mode value
                            if True if random.randint(0, 1) == 0 else False:
                                train_column.loc[train_column.isnull()] = data_notnull.mean()
                            else:
                                train_column.loc[train_column.isnull()] = data_notnull.mode()[0]
                            break

                else:
                    print("数据参考：")
                    print(train_column.value_counts().head())
                    print(self.handle_missing_value_str)
                    x = Stdin.get_int_input('')

                if x == 0:
                    value = input("请输入要填入的数值")
                    if train_column.dtype.name == 'float64':
                        value = float(value)
                    elif train_column.dtype.name == 'int64':
                        value = int(value)
                    train_column.loc[train_column.isnull()] = value
                    break
                elif x == 1:
                    if self.is_categorical_data(data_notnull.dtype):
                        train_column.loc[train_column.isnull()] = data_notnull.mode()[0]
                    else:
                        train_column.loc[train_column.isnull()] = data_notnull.mean()
                    break
                elif x == 2:
                    train_column.loc[train_column.isnull()] = data_notnull.mode()[0]
                    break
                elif x == 3:
                    # todo
                    break
                elif x == 4:
                    delete_columns_name.append(train_column.name)
                    break
                elif x == 5:
                    break
                else:
                    print("错误输入")
        for delete_column_name in delete_columns_name:
            train_columns.drop(delete_column_name, axis=1, inplace=True)
        return train_columns

    def data_encoding_x(self, train_columns, is_auto):
        encoded_columns = pd.DataFrame()
        for column_name in train_columns.columns.values.tolist():
            train_column = train_columns[column_name]
            encoded_column = None
            if self.is_categorical_data(train_column.dtype):
                while True:
                    if is_auto:
                        x = 1
                    else:
                        print("{name}为categorical类型".format(name=train_column.name))
                        print(self.handle_cate_encode_str)
                        x = Stdin.get_int_input('')

                    if x == 0:
                        encoded_column = pd.Series(preprocessing.LabelEncoder().fit_transform(train_column),
                                                   name=train_column.name)
                        break
                    elif x == 1:
                        # test if there are too many unique values,cast minority to MINORITY_MARK
                        MAX_VALUE_COUNTS = 100
                        MINORITY_MARK = -1
                        value_counts_dic = train_column.value_counts()
                        if value_counts_dic.values.size > MAX_VALUE_COUNTS:
                            min_value_counts = value_counts_dic[MAX_VALUE_COUNTS]
                            # LabelEncoder to accelerate the loop
                            train_column = pd.Series(preprocessing.LabelEncoder().fit_transform(train_column),
                                                     name=train_column.name)
                            value_counts_dic = train_column.value_counts()
                            print("column {name} is doing {times} times loop which may take a long time"
                                  .format(name=train_column.name, times=len(train_column)))
                            for idx, val in enumerate(train_column):
                                if value_counts_dic[val] < min_value_counts:
                                    train_column[idx] = MINORITY_MARK

                        # ont-hot encoder
                        pass
                        encoded_column = pd.get_dummies(train_column, prefix=train_column.name)
                        break
                    elif x == 2:
                        break
                    else:
                        print("错误输入")
            else:
                while True:
                    if is_auto:
                        x = 2
                    else:
                        print("{name}为numerical类型".format(name=train_column.name))
                        print(self.handle_num_encode_str)
                        x = Stdin.get_int_input('')
                    if x == 0:
                        # Scaled data has zero mean and unit variance
                        encoded_column = pd.Series(preprocessing.scale(train_column), name=train_column.name)
                        break
                    elif x == 1:
                        # scaling individual samples to have unit norm
                        encoded_column = pd.Series(preprocessing.normalize(train_column, norm='l2')[0],
                                                   name=train_column.name)
                        break
                    elif x == 2:
                        break
                    else:
                        print("错误输入")
            if encoded_column is not None:
                encoded_columns = pd.concat([encoded_columns, encoded_column], axis=1)
            else:
                encoded_columns = pd.concat([encoded_columns, train_column], axis=1)
        return encoded_columns

    def feature_selection_x(self, train_columns, pre_column):
        selector = SelectFromModel(RandomForestClassifier(n_jobs=-1, max_depth=10, n_estimators=15),
                                   threshold='20*mean')
        selector.fit(train_columns, pre_column)
        mask = selector.get_support()
        train_columns = train_columns.loc[:, mask]
        return train_columns

    def print_columns(self, cols_list):
        for idx, val in enumerate(cols_list):
            print(str(idx) + ':' + val)

    def pick_y(self, cols_list):
        while True:
            y = int(Stdin.get_int_input("选择要预测的列"))
            if y in range(0, len(cols_list)):
                break
            else:
                print("无此列")
        return y

    def data_split(self, df):
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=20)
        return df_train, df_test

    def model_selection(self, is_auto):
        model_list = [
            'LogisticRegression',
            'RandomForestClassifier',
            'RidgeClassifier',
            'GradientBoostingClassifier',
            'ExtraTreesClassifier',
            'AdaBoostClassifier'
        ]
        model_map = {
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'RidgeClassifier': RidgeClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
        }
        while True:
            if is_auto:
                return model_list, model_map
            else:
                for i in range(0, len(model_list)):
                    print(str(i) + ":" + model_list[i])
                x = Stdin.get_int_input("选择你要使用的模型")
            if x in range(0, len(model_list)):
                return model_list[x], model_map[model_list[x]]
            else:
                print("Error Input")

    def select_best_model(self, data_x, data_y, is_auto):
        if is_auto:
            model_name_list, model_map = self.model_selection(is_auto)
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
            model_name, model = self.model_selection(is_auto)
            model.fit(data_x, data_y)
            scores = cross_val_score(model, data_x, data_y, cv=10)
            score = sum(scores) / len(scores)
            return model_name, model, score


if __name__ == '__main__':
    # input data
    # old_stdin = Stdin.input_stdin("0\n1\n4\n2\n2\n0\n0\n2\n2\n1\n0\n1\n3\n")
    # Stdin.recover_stdin(old_stdin)
    # old_stdin = Stdin.input_stdin("16\n-1\n1\n0\n-1\n-1\n-1\n-1\n1\n-1\n0\n0\n")

    # read data
    train_df = pd.read_csv("/Users/wei/PycharmProjects/MachineLearning/Loan/loan.csv")
    preprocessed_df = pd.read_csv("preprocessed_df.csv")

    # initial SmartML
    smart_ml = SmartML(train_df=train_df)
    smart_ml.auto_learn(preprocessed_df=preprocessed_df)
