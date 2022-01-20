import pytest
from IOKR.data.load_data import load_bibtex
from IOKR.model.utils import project_root
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
from IOKR.model.model import IOKR as iokr
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal

#Global Variables of interest
path = join(project_root(), "data/bibtex")
X, y, _, _ = load_bibtex(path)


def fitted_IOKR(X, y, L=1e-5, sx=1000, sy=10):
    """"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = iokr()
    clf.verbose = 1
    clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
    Y_pred_train = clf.predict(X_test=X_train)
    Y_pred_test = clf.predict(X_test=X_test)
    f1_train = f1_score(Y_pred_train, Y_train, average='samples')
    f1_test = f1_score(Y_pred_test, Y_test, average='samples')

    return {'Train-score': f1_train, 'Test-score': f1_test}

'''WAITING FOR Y_candidates modification
class TestFit():

    def test_fit_prints(self, capfd):
        """Test if fit function actually prints something"""
        scores = fitted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        out, err = capfd.readouterr()
        assert err == "", f'{err}: need to be fixed'
        assert out != "", f'Fitting Time should have been printed '


class TestPredict():

    def test_model_return_vals(self):
        """
        Tests for the returned values of the modeling function
        """
        scores = fitted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        # Check returned scores' type
        assert isinstance(scores['Train-score'], float)
        assert isinstance(scores['Test-score'], float)
        # Check returned scores' range
        assert scores['Train-score'] >= 0.0
        assert scores['Train-score'] <= 1.0
        assert scores['Test-score'] >= 0.0
        assert scores['Test-score'] <= 1.0

    def test_model_return_object(self):
        """
        Tests the returned object of the modeling function
        """
        scores = fitted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        # Check the return object type
        assert isinstance(scores, dict), f'"scores" should be a dict, but instead is a {type(scores)}'
        # Check the length of the returned object
        assert len(scores) == 2, f'"scores" should have a length of 2, but instead it is {len(scores)}'
        # Check the correctness of the names of the returned dict keys
        assert 'Train-score' in scores and 'Test-score' in scores, f'Names in "score" should be "Train-score" and ' \
                                                                   f'"Test-score" '

    def test_raised_exception(self):
        """
        Tests for raised exception with pytest.raises context manager
        """
        # ValueError
        with pytest.raises(ValueError):
            # Insert a np.nan into the X array
            Xt, yt = X, y
            Xt[1] = np.nan
            scores = fitted_IOKR(Xt, yt, L=1e-5, sx=1000, sy=10)
            # Insert a np.nan into the y array
            Xt, yt = X, y
            y[1] = np.nan
            scores = fitted_IOKR(Xt, yt, L=1e-5, sx=1000, sy=10)

        with pytest.raises(ValueError) as exception:
            # Insert a string into the X array
            Xt, yt = X, y
            X[1] = "A string"
            scores = fitted_IOKR(Xt, yt, L=1e-5, sx=1000, sy=10)
            assert "could not convert string to float" in str(exception.value)
'''

'''IS NOT WORKING YET
def test_wrong_input_raises_assertion():
    """
    Tests for various assertion cheks written in the modeling function
    """
    # Test that it handles the case of: X is a string
    msg = fitted_IOKR('X', y, L=1e-5, sx=1000, sy=10)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "X must be a Numpy array"
    # Test that it handles the case of: y is a string
    msg = fitted_IOKR(X, 'y', L=1e-5, sx=1000, sy=10)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "y must be a Numpy array"

    # Test that it handles the case of: test_frac is a string
    msg = train_linear_model(X, y, test_frac='0.2')
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be a floating point number"
    # Test that it handles the case of: test_frac is within 0.0 and 1.0
    msg = train_linear_model(X, y, test_frac=-0.2)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be between 0.0 and 1.0"
    msg = train_linear_model(X, y, test_frac=1.2)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Test set fraction must be between 0.0 and 1.0"
    # Test that it handles the case of: filename for model save a string
    msg = train_linear_model(X, y, filename=2.0)
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Filename must be a string"
    # Test that function is checking input vector shape compatibility
    X = X.reshape(10, 10)
    msg = train_linear_model(X, y, filename='testing')
    assert isinstance(msg, AssertionError)
    assert msg.args[0] == "Row numbers of X and y data must be identical"
'''