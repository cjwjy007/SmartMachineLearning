from smart_ml.smart_column_cleaner import *
from smart_ml.smart_value_filler import *
from smart_ml.smart_data_encoder import *
from smart_ml.smart_feature_selection import *
from smart_ml.smart_trainer import *

pd.options.mode.chained_assignment = None


class SmartML:
    def __init__(self, train_df=None, preprocessed_df=None):
        self.train_df = train_df
        self.preprocessed_df = preprocessed_df

    def auto_learn(self, ):
        if self.preprocessed_df is None:
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

            # self.auc(trainFeatures=train_df,trainLabels=preprocessed_y)
            # preprocess X
            preprocessed_x = self.data_x_preprocessing(train_columns=self.train_df, pre_column=preprocessed_y,
                                                       is_auto=True)

            # save the Dataframe as csv
            pd.concat([preprocessed_y, preprocessed_x], axis=1).to_csv("preprocessed_df.csv")
        else:
            # get x and y
            preprocessed_y = self.preprocessed_df.iloc[:, 0]
            preprocessed_x = self.preprocessed_df.drop(self.preprocessed_df.columns[0], axis=1)

        # train data
        smart_trainer = SmartTrainer(train_x=preprocessed_x, train_y=preprocessed_y, is_auto=False)
        best_model_name, best_model, best_score = smart_trainer.select_best_model()
        print("best_model_name: " + best_model_name)
        print("best_score： " + str(best_score))

    def data_y_preprocessing(self, train_columns, y_index, is_auto):
        self.transform_y(train_df=train_columns, index_y=y_index)
        self.data_encoding_y(train_df=train_columns, index_y=y_index)

    def transform_y(self, train_df, index_y):
        column_y = train_df.iloc[:, index_y]
        if is_categorical_data(column_y.dtype):
            values_y = column_y.value_counts()
            column_y_map = {}
            for idx, val in enumerate(values_y):
                x = Stdin.get_int_input("map " + str(values_y.index[idx]) + " to(if input -1,delete column):")
                if x == -1:
                    column_y_map[values_y.index[idx]] = np.nan
                else:
                    column_y_map[values_y.index[idx]] = x
            train_df.iloc[:, index_y] = column_y.map(column_y_map)
        train_df.dropna(subset=[column_y.name], inplace=True)
        train_df.index = range(len(train_df))

    def data_encoding_y(self, train_df, index_y):
        column_y = self.train_df.iloc[:, index_y]
        if is_categorical_data(column_y.dtype):
            self.train_df.iloc[:, index_y] = pd.Series(preprocessing.LabelEncoder().fit_transform(column_y),
                                                       name=column_y.name)

    def data_x_preprocessing(self, train_columns, pre_column, is_auto):
        self._feature_pre_selection_x(train_columns)
        self._fill_missing_x(train_columns, is_auto)
        train_columns = self._data_encoding_x(train_columns, is_auto)
        train_columns = self._feature_selection_x(train_columns, pre_column, is_auto)
        return train_columns

    def _feature_pre_selection_x(self, train_columns):
        smart_cleaner = SmartColumnCleaner(train_df=train_columns)
        smart_cleaner.clean_columns()

    def _fill_missing_x(self, train_columns, is_auto):
        smart_value_filler = SmartValueFiller(train_df=train_columns, is_auto=is_auto)
        smart_value_filler.value_filling()

    def _data_encoding_x(self, train_columns, is_auto):
        smart_encoder = SmartDataEncoder(train_df=train_columns, is_auto=is_auto)
        return smart_encoder.data_encode()

    def _feature_selection_x(self, train_columns, pre_column, is_auto):
        smart_feature_selection = SmartFeatureSelection(train_x=train_columns, train_y=pre_column, is_auto=is_auto)
        return smart_feature_selection.feature_selection()

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
    # train_df = pd.read_csv("/Users/wei/Downloads/train.csv")
    # titanic df
    t_df = pd.read_csv("/Users/wei/PycharmProjects/MachineLearning/Titanic/train.csv")
    p_df = pd.read_csv("loan_preprocessed_df.csv")

    # initial SmartML
    smart_ml = SmartML(train_df=t_df, preprocessed_df=None)
    smart_ml.auto_learn()
