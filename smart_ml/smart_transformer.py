from smart_ml.smart_ml_general import is_categorical_data, Stdin
import numpy as np


class SmartTransformer:
    def __init__(self, train_df=None, train_y=None, is_auto=True):
        self.train_df = train_df
        self.train_y = train_y
        self.is_auto = is_auto

    def transform_y(self):
        # map non-numerical y value
        if is_categorical_data(self.train_y.dtype):
            values_y = self.train_y.value_counts()
            column_y_map = {}
            for idx, val in enumerate(values_y):
                x = Stdin.get_int_input("map " + str(values_y.index[idx]) + " to(if input -1,delete column):")
                if x == -1:
                    column_y_map[values_y.index[idx]] = np.nan
                else:
                    column_y_map[values_y.index[idx]] = x
            self.train_y = self.train_y.map(column_y_map)
        # remove y missing value
        self.train_df.dropna(subset=[self.train_y.name], inplace=True)
        self.train_df.reset_index(drop=True, inplace=True)
        # imbalance_data
        self.transform_imbalance_data()

    def transform_imbalance_data(self):
        values_y = self.train_y.value_counts()
        values_size = values_y.size
        min_balance_percentage_name = None
        min_value_count = values_y[0]
        for idx, val in enumerate(values_y):
            balance_percentage = val / (self.train_y.size / values_size)
            if balance_percentage < 0.2 and val <= min_value_count:
                min_balance_percentage_name = idx
                min_value_count = val
        # data is balanced
        if min_balance_percentage_name is None:
            return
        # balance data
        # if min_value_count > 10000 oversample column to min_value_count
        # if min_value_count < 10000 oversample column to 1.5 * min_value_count
        if min_value_count < 10000:
            min_value_count = int(min_value_count * 1.5)

        for idx, val in enumerate(values_y):
            if idx is not min_balance_percentage_name:
                if val > min_value_count:
                    index_to_drop = self.train_y.ix[self.train_y == idx].sample(n=val - min_value_count,
                                                                                random_state=7).index
                    self.train_df.drop(index_to_drop, inplace=True)
                    self.train_df.reset_index(drop=True, inplace=True)
