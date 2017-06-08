import random

from smart_ml.smart_ml_general import *


class SmartValueFiller:
    def __init__(self, train_df=None, is_auto=True, pipeline=None):
        self.pipeline = pipeline
        self.train_df = train_df
        self.is_auto = is_auto
        self.handle_missing_value_str = """
                How to handle missing value?
                0:Fill in custom value
                1:Fill in mean value
                2:Fill in mode value
                3:Use model to predict
                4:Delete this column
                5:Ignore"""

    def filling_value(self):
        delete_columns_name = []
        for col_name in self.train_df.columns:
            self._filling_column(col=col_name, delete_columns_name=delete_columns_name)

        self.train_df.drop(delete_columns_name, axis=1, inplace=True)

    def _filling_column(self, col=None, delete_columns_name=None):
        train_column = self.train_df[col]
        # not contain missing values
        if not train_column.isnull().any():
            print("data {name} doesn't contain missing values".format(name=train_column.name))
            return

        data_notnull = train_column[train_column.notnull()]
        # handle missing value
        print("data {name} contains missing values".format(name=train_column.name))
        while True:
            if self.is_auto:
                # categorical value
                if is_categorical_data(data_notnull.dtype):
                    train_column.loc[train_column.isnull()] = 'nan_mark'
                    self.pipeline.append([col, 1])
                    break
                # numerical value
                else:
                    # filling in the mean value or mode value
                    if True if random.randint(0, 1) == 0 else False:
                        train_column.loc[train_column.isnull()] = data_notnull.mean()
                        self.pipeline.append([col, 2])

                    else:
                        train_column.loc[train_column.isnull()] = data_notnull.mode()[0]
                        self.pipeline.append([col, 3])
                    break

            else:
                print("data reference：")
                print(train_column.value_counts().head())
                print(self.handle_missing_value_str)
                x = Stdin.get_int_input('')

            if x == 0:
                value = input("Enter the value to be filled in")
                if train_column.dtype.name == 'float64':
                    value = float(value)
                elif train_column.dtype.name == 'int64':
                    value = int(value)
                train_column.loc[train_column.isnull()] = value
                break
            elif x == 1:
                if is_categorical_data(data_notnull.dtype):
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
                print("Error Input")
