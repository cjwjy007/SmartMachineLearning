import numpy as np
import sys
from io import StringIO


def is_categorical_data(dtype):
    if dtype is np.dtype(np.object):
        return True
    else:
        return False


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
