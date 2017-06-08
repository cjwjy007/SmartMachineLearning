from smart_ml.smart_column_cleaner import *
from smart_ml.smart_value_filler import *
from smart_ml.smart_data_encoder import *
from smart_ml.smart_feature_selection import *
from smart_ml.smart_trainer import *
from smart_ml.smart_transformer import *

pd.options.mode.chained_assignment = None


class SmartML:
    def __init__(self, train_df=None, preprocessed_df=None, test_df=None):
        self.train_df = train_df
        self.preprocessed_df = preprocessed_df
        self.test_df = test_df
        self.smart_pipeline = []

    def auto_learn(self):
        if self.preprocessed_df is None:
            # columns info
            columns_list = self.train_df.columns.values.tolist()
            self.print_columns(columns_list)

            # pick predict column
            y_index = self.pick_y(columns_list)

            # preprocess Y
            self.data_y_preprocessing(train_columns=self.train_df, pre_column=self.train_df.iloc[:, y_index],
                                      is_auto=True)

            # extract Y column
            preprocessed_y = self.train_df.iloc[:, y_index]
            self.train_df.drop(self.train_df.columns[y_index], axis=1, inplace=True)

            # preprocess X
            preprocessed_x = self.data_x_preprocessing(train_columns=self.train_df, pre_column=preprocessed_y,
                                                       is_auto=True)

            # save the Dataframe as csv
            pd.concat([preprocessed_y, preprocessed_x], axis=1).to_csv("preprocessed_df.csv", index=False)
        else:
            # get x and y
            preprocessed_y = self.preprocessed_df.iloc[:, 0]
            preprocessed_x = self.preprocessed_df.drop(self.preprocessed_df.columns[0], axis=1)

        # train data
        smart_trainer = SmartTrainer(train_x=preprocessed_x, train_y=preprocessed_y, is_auto=False)
        best_model_name, best_model = smart_trainer.select_best_model()

        if self.test_df is not None:
            # transform test set
            preprocessed_test_x = self.transform_test(preprocessed_x.columns)
            # predict test set
            y_pre = best_model.predict(preprocessed_test_x)
            submission = pd.DataFrame({
                self.test_df.columns[0]: self.test_df[self.test_df.columns[0]],
                preprocessed_y.name: y_pre
            })
            submission.to_csv('submission.csv', index=False)

    def data_y_preprocessing(self, train_columns, pre_column, is_auto):
        smart_transformer = SmartTransformer(train_df=train_columns, train_y=pre_column, is_auto=is_auto)
        smart_transformer.transform_y()

    def data_x_preprocessing(self, train_columns, pre_column, is_auto):
        self._feature_pre_selection_x(train_columns)
        self._fill_missing_x(train_columns, is_auto)
        train_columns = self._data_encoding_x(train_columns, is_auto)
        train_columns = self._feature_selection_x(train_columns, pre_column, is_auto)
        return train_columns

    def _feature_pre_selection_x(self, train_columns):
        smart_cleaner = SmartColumnCleaner(train_df=train_columns, pipeline=self.smart_pipeline)
        smart_cleaner.clean_columns()

    def _fill_missing_x(self, train_columns, is_auto):
        smart_value_filler = SmartValueFiller(train_df=train_columns, is_auto=is_auto, pipeline=self.smart_pipeline)
        smart_value_filler.filling_value()

    def _data_encoding_x(self, train_columns, is_auto):
        smart_encoder = SmartDataEncoder(train_df=train_columns, is_auto=is_auto, pipeline=self.smart_pipeline)
        return smart_encoder.data_encode()

    def _feature_selection_x(self, train_columns, pre_column, is_auto):
        smart_feature_selection = SmartFeatureSelection(train_x=train_columns, train_y=pre_column, is_auto=is_auto)
        return smart_feature_selection.feature_selection()

    def transform_test(self, train_column=None):
        self._feature_pre_selection_x(self.test_df)
        self._fill_missing_x(self.test_df, True)
        test_df = self._data_encoding_x(self.test_df, True)
        for idx,val in enumerate(train_column):
            if val not in test_df.columns:
                new_se = pd.Series(name=val)
                test_df = pd.concat([test_df,new_se],axis=1)
                test_df[val] = 0
        test_df = test_df[train_column]
        return test_df

    def print_columns(self, cols_list):
        for idx, val in enumerate(cols_list):
            print(str(idx) + ':' + val)

    def pick_y(self, cols_list):
        while True:
            y = int(Stdin.get_int_input("choose predicting column:"))
            if y in range(0, len(cols_list)):
                break
            else:
                print("invalid column")
        return y


if __name__ == '__main__':
    # input data
    # old_stdin = Stdin.input_stdin("0\n1\n4\n2\n2\n0\n0\n2\n2\n1\n0\n1\n3\n")
    # Stdin.recover_stdin(old_stdin)
    # old_stdin = Stdin.input_stdin("16\n-1\n1\n0\n-1\n-1\n-1\n-1\n1\n-1\n0\n1\n")

    # read data
    # satisfaction df
    # t_df = pd.read_csv("/Users/wei/Downloads/train.csv")
    # te_df = pd.read_csv("/Users/wei/Downloads/test.csv")
    # titanic df
    t_df = pd.read_csv("/Users/wei/PycharmProjects/MachineLearning/Titanic/train.csv")
    te_df = pd.read_csv("/Users/wei/Downloads/titanic_test.csv")
    # p_df = pd.read_csv("preprocessed_df.csv")

    # initial SmartML
    smart_ml = SmartML(train_df=t_df, preprocessed_df=None, test_df=te_df)
    smart_ml.auto_learn()
