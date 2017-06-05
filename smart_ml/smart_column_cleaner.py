import numpy as np
from smart_ml.smart_ml_general import *


class SmartColumnCleaner:
    def __init__(self, train_df=None):
        self.train_df = train_df

    def clean_columns(self):
        self._clean_constant_columns()
        self._clean_duplicate_columns()
        self._clean_high_missing_rate_columns()

    def _clean_duplicate_columns(self):
        cols_to_remove = []
        for col in self.train_df.columns:
            if not is_categorical_data(self.train_df[col].dtype):
                if self.train_df[col].std() == 0:
                    cols_to_remove.append(col)

        self.train_df.drop(cols_to_remove, axis=1, inplace=True)

    def _clean_constant_columns(self):
        cols_to_remove = []
        columns = self.train_df.columns
        for i in range(len(columns) - 1):
            v = self.train_df[columns[i]].values
            for j in range(i + 1, len(columns)):
                if np.array_equal(v, self.train_df[columns[j]].values):
                    cols_to_remove.append(columns[j])

        self.train_df.drop(cols_to_remove, axis=1, inplace=True)

    def _clean_high_missing_rate_columns(self):
        cols_to_remove = []
        for col in self.train_df.columns:
            col_series = self.train_df[col]
            col_notnull = col_series[col_series.notnull()]
            if col_notnull.size / col_series.size <= 0.1:
                cols_to_remove.append(col_notnull.name)

        self.train_df.drop(cols_to_remove, axis=1, inplace=True)

    def clean_values_almost_different_columns(self):
        cols_to_remove = []
        for col in self.train_df.columns:
            col_series = self.train_df[col]
            col_notnull = col_series[col_series.notnull()]
            if col_notnull.value_counts().values.size / col_notnull.size > 0.7:
                print("values in {name} are almost different,automatically delete".format(name=col_notnull.name))
                cols_to_remove.append(col_notnull.name)

        self.train_df.drop(cols_to_remove, axis=1, inplace=True)
