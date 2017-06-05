from sklearn import preprocessing

from smart_ml.smart_ml_general import *
import pandas as pd


class SmartDataEncoder:
    def __init__(self, train_df=None, is_auto=True):
        self.train_df = train_df
        self.is_auto = is_auto
        self.handle_cate_encode_str = """
                0.LabelEncoder
                1.One-hot Encoder
                2.Ignore"""

        self.handle_num_encode_str = """
                0.Scale
                1.Normalize
                2.Ignore"""

    def data_encode(self):
        encoded_columns = pd.DataFrame()
        for col in self.train_df.columns:
            print("encoding {name}".format(name=col))
            train_column = self.train_df[col]
            encoded_column = None
            if is_categorical_data(train_column.dtype):
                while True:
                    if self.is_auto:
                        x = 1
                    else:
                        print("{name} is categorical type".format(name=train_column.name))
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
                            print("column {name} is performing {times} times loop which may take a long time"
                                  .format(name=train_column.name, times=len(train_column)))
                            for idx, val in enumerate(train_column):
                                if value_counts_dic[val] < min_value_counts:
                                    train_column[idx] = MINORITY_MARK

                        # ont-hot encoder
                        encoded_column = pd.get_dummies(train_column, prefix=train_column.name)
                        break
                    elif x == 2:
                        break
                    else:
                        print("Error Input")
            else:
                while True:
                    if self.is_auto:
                        x = 2
                    else:
                        print("{name} is numerical type".format(name=train_column.name))
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
                        print("Error Input")
            if encoded_column is not None:
                encoded_columns = pd.concat([encoded_columns, encoded_column], axis=1)
            else:
                encoded_columns = pd.concat([encoded_columns, train_column], axis=1)
        return encoded_columns
