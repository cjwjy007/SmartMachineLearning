import unittest

from AutoMachineLearning.smart_ml.AutoML import *


class MyTestCase(unittest.TestCase):
    def _test_feature_pre_selection(self):
        test_df = pd.DataFrame(
            {
                'col_a': [1, 2, 3],
                'col_b': ['a', 'b', 'c'],
                'col_c': [1.0, 2.0, 3.0],
            }
        )
        result_df = feature_pre_selection(test_df)
        expected_df = pd.DataFrame(
            {
                'col_a': [1, 2, 3],
                'col_c': [1.0, 2.0, 3.0],
            }
        )

        self.assertTrue(pd.DataFrame.equals(result_df, expected_df), "test_feature_pre_selection failed")

    def _test_fill_missing(self):
        test_df = pd.DataFrame(
            {
                'col_a': [1, 2, 3],
                'col_b': ['a', 'b', 'c'],
                'col_c': [1.0, 2.0, np.nan],
                'col_d': ['b', 'b', None],
            }
        )

        old_stdin = input_stdin("a\n0\n3.0\n1\n")

        result_df = fill_missing(test_df)

        recover_stdin(old_stdin)

        expected_df = pd.DataFrame(
            {
                'col_a': [1, 2, 3],
                'col_b': ['a', 'b', 'c'],
                'col_c': [1.0, 2.0, 3.0],
                'col_d': ['b', 'b', 'b'],
            }
        )

        self.assertTrue(pd.DataFrame.equals(result_df, expected_df), "test_fill_missing failed")

    def _test_data_encoding(self):
        test_df = pd.DataFrame(
            {
                'col_a': [1, 2, 3],
                'col_b': ['a', 'b', 'c'],
                'col_c': ['aa', 'bb', 'cc'],
                'col_d': [1.0, 2.0, 3.0]
            }
        )

        old_stdin = input_stdin("0\n0\n1\n2\n")

        result_df = data_encoding(test_df)

        recover_stdin(old_stdin)

        expected_df = pd.DataFrame(
            {
                'col_a': [1, 2, 3],
                'col_b': [0, 1, 2],
                'col_c_aa': np.uint8([1, 0, 0]),
                'col_c_bb': np.uint8([0, 1, 0]),
                'col_c_cc': np.uint8([0, 0, 1]),
                'col_d': [1.0, 2.0, 3.0]
            }
        )

        self.assertTrue(pd.DataFrame.equals(result_df.iloc[:, 1:], expected_df.iloc[:, 1:]),
                        "test_data_encoding failed")

    def _test_feature_selection(self):
        test_df = pd.read_csv("preprocessed_titanic.csv")

        train_df = pd.read_csv("/Users/wei/PycharmProjects/AutoLearn/data/train.csv")

        y_column = train_df.iloc[:, 1]

        result_df = feature_selection(test_df, y_column)

        result_df.to_csv('feature_selection_processed.csv', index=False)

    def _test_model_selection(self):
        model_name, model = model_selection()


if __name__ == '__main__':
    unittest.main()
