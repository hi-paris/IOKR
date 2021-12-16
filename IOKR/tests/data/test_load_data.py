"""Test module for the module: load_data.py"""

import pytest
import numpy as np
from IOKR.data.load_data import load_bibtex



class TestLoadBibtex():
    """Test class for the function: load_bibtex"""

    def test_returned_variables_not_empty(self):
        """test checking if returned variables from load_bibtex are not empty"""
        load = load_bibtex("IOKR/data/bibtex")
        print(load)
        assert load[0] is not None, "Expected variable: 'X'"
        assert load[1] is not None, "Expected variable: 'Y'"
        assert load[2] != "", "Expected variable: 'X_txt'"
        assert load[3] != "", "Expected variable: 'Y_txt'"

    def test_returned_variables_good_type(self):
        """Test checking if returned variables from load_bibtex are the expected type"""
        load = load_bibtex("IOKR/data/bibtex")
        actual_x = type(load[0])
        actual_y = type(load[1])
        actual_x_txt = type(load[2])
        actual_y_txt = type(load[3])
        expected1 = "np.array"
        expected2 = 'list'
        print(actual_x, actual_y, actual_x_txt, actual_y_txt)
        assert isinstance(load[0], np.ndarray), f"'X' should be {expected1}, but is {actual_x} "
        assert isinstance(load[1], np.ndarray), f"'Y' should be {expected1}, but is {actual_y} "
        assert isinstance(load[2], list), f"'X_txt' should be {expected2}, but is {actual_x_txt} "
        assert isinstance(load[3], list), f"'Y_txt' should be {expected2}, but is {actual_y_txt} "
