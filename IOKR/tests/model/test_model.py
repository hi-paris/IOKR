import pytest
from IOKR.data.load_data import load_bibtex, load_corel5k
from IOKR.model.utils import project_root
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import mean_squared_error as MSE
import numpy as np
from IOKR.model.model import IOKR as iokr
import time
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal

# Global Variables of interest
# Paths loading datasets until DATASETS works
path = join(project_root(), "data/bibtex")
X, y, _, _ = load_bibtex(path)
# path = join(project_root(), "data/corel5k")
# X, y, _, _ = load_corel5k(path)

# Not used currently
bibtex = load_bibtex(join(project_root(), "data/bibtex"))
corel5k = load_corel5k(join(project_root(), "data/corel5k"))

DATASETS = {
    "bibtex": {'X': bibtex[0], 'Y': bibtex[1]},
    "corel5K": {'X': corel5k[0], 'Y': corel5k[1]},
}


def fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10):
    """Function running IOKR and returning Y_train, Y_test, and Y_preds, for readability purposes"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = iokr()
    clf.verbose = 1
    clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
    Y_pred_train = clf.predict(X_test=X_train, Y_candidates=Y_train)
    Y_pred_test = clf.predict(X_test=X_test, Y_candidates=Y_train)

    return {'Y_train':Y_train,
            'Y_test':Y_test,
            'Y_pred_train': Y_pred_train,
            'Y_pred_test': Y_pred_test}


class TestFit:
    """class for the tests concerning .fit()"""
    #IDEES DE TEST:
    #   quand input(shape X et y) not equal
    #   GridSearchCV?

    def test_fit_prints(self, capfd):
        """Test if fit function actually prints something"""
        scores = fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        out, err = capfd.readouterr()
        assert err == "", f'{err}: need to be fixed'
        assert out != "", f'Fitting Time should have been printed '

    def test_fit_time(self, X, y, L=1e-5, sx=1000, sy=10):
        """Test the time for fitting"""
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        clf = iokr()
        clf.verbose = 1
        t0 = time.time()
        clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
        fit_time = time.time() - t0
        assert fit_time < 100, f'"fit_time" is over 100 seconds'

    def test_numerical_stability(self, L=1e-5, sx=1000, sy=10):
        """Check numerical stability."""
        X = np.array([
            [152.08097839, 140.40744019, 129.75102234, 159.90493774],
            [142.50700378, 135.81935120, 117.82884979, 162.75781250],
            [127.28772736, 140.40744019, 129.75102234, 159.90493774],
            [132.37025452, 143.71923828, 138.35694885, 157.84558105],
            [103.10237122, 143.71928406, 138.35696411, 157.84559631],
            [127.71276855, 143.71923828, 138.35694885, 157.84558105],
            [120.91514587, 140.40744019, 129.75102234, 159.90493774]])

        y = np.array(
            [1., 0.70209277, 0.53896582, 0., 0.90914464, 0.48026916, 0.49622521])

        with np.errstate(all="raise"):
            reg = iokr()
            reg.fit(X, y, L=L, sx=sx, sy=sy)
            reg.fit(X, -y, L=L, sx=sx, sy=sy)
            reg.fit(-X, y, L=L, sx=sx, sy=sy)
            reg.fit(-X, -y, L=L, sx=sx, sy=sy)

    def test_raise_error_on_1d_input(self, L=1e-5, sx=1000, sy=10):
        """Test that an error is raised when X or Y are 1D arrays"""
        Xt = X[:, 0].ravel()
        Xt_2d = X[:, 0].reshape((-1, 1))
        yt = y

        with pytest.raises(ValueError):
            iokr().fit(Xt, yt, L=L, sx=sx, sy=sy)

        iokr().fit(Xt_2d, yt, L=L, sx=sx, sy=sy)
        with pytest.raises(ValueError):
            iokr().predict([Xt])

    def test_warning_on_big_input(self, L=1e-5, sx=1000, sy=10):
        """Test if the warning for too large inputs is appropriate"""
        Xt = np.repeat(10 ** 40., 4).astype(np.float64).reshape(-1, 1)
        clf = iokr()
        try:
            clf.fit(Xt, [0, 1, 0, 1], L=L, sx=sx, sy=sy)
        except ValueError as e:
            assert "float32" in str(e)



class TestPredict:

    def test_model_return_values(self):
        """
        Tests for the returned values of the modeling function
        """
        arrays = fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        # Check returned arrays' type
        assert arrays['Y_train'] != "", "'Y_train' is empty"
        assert arrays['Y_test'] != "", "'Y_test' is empty"
        assert arrays['Y_pred_train'] != "", "'Y_pred_train' is empty"
        assert arrays['Y_pred_test'] != "", "'Y_pred_test' is empty"
        assert isinstance(arrays['Y_train'], np.ndarray), \
            f"'Y_train' should be a 'np.ndarray, but is {type(arrays['Y_train'])}"
        assert isinstance(arrays['Y_test'], np.ndarray), \
            f"'Y_test' should be a 'np.ndarray, but is {type(arrays['Y_test'])}"
        assert isinstance(arrays['Y_pred_train'], np.ndarray), \
            f"'Y_pred_train' should be a 'np.ndarray, but is {type(arrays['Y_pred_train'])}"
        assert isinstance(arrays['Y_pred_test'], np.ndarray), \
            f"'Y_pred_test' should be a 'np.ndarray, but is {type(arrays['Y_pred_test'])}"

    def test_model_return_object(self):
        """
        Tests the returned object of the modeling function
        """
        returned_IOKR = fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        # Check the return object type
        assert isinstance(returned_IOKR, dict), f'"returned_IOKR" should be a dict, but instead is a {type(returned_IOKR)}'
        # Check the length of the returned object
        assert len(returned_IOKR) == 4, f'"returned_IOKR" should have a length of 4, but instead it is {len(returned_IOKR)}'

    def test_raised_exception(self):
        """
        Tests for raised exceptions (To complete)
        """
        # ValueError
        with pytest.raises(ValueError):
            # Insert a np.nan into the X array
            Xt, yt = X, y
            Xt[1] = np.nan
            scores = fitted_predicted_IOKR(Xt, yt, L=1e-5, sx=1000, sy=10)
            # Insert a np.nan into the y array
            Xt, yt = X, y
            yt[1] = np.nan
            scores = fitted_predicted_IOKR(Xt, yt, L=1e-5, sx=1000, sy=10)

        with pytest.raises(ValueError) as exception:
            # Insert a string into the X array
            Xt, yt = X, y
            Xt[1] = "A string"
            scores = fitted_predicted_IOKR(Xt, yt, L=1e-5, sx=1000, sy=10)
            assert "could not convert string to float" in str(exception.value)

    '''IS NOT WORKING YET
    def test_wrong_input_raises_assertion():
        """
        Tests for various assertion cheks written in the modeling function
        """
        # Test that it handles the case of: X is a string
        msg = fitted_predicted_IOKR('X', y, L=1e-5, sx=1000, sy=10)
        assert isinstance(msg, AssertionError)
        assert msg.args[0] == "X must be a Numpy array"
        # Test that it handles the case of: y is a string
        msg = fitted_predicted_IOKR(X, 'y', L=1e-5, sx=1000, sy=10)
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

    def test_predict_time(self, X, y, L=1e-5, sx=1000, sy=10):
        """Test the time for predicting"""
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        clf = iokr()
        clf.verbose = 1
        clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
        train_t0 = time.time()
        Y_pred_train = clf.predict(X_test=X_train, Y_candidates=Y_train)
        train_pred_time = time.time() - train_t0
        test_t0 = time.time()
        Y_pred_test = clf.predict(X_test=X_test, Y_candidates=Y_train)
        test_pred_time = time.time() - test_t0
        assert train_pred_time < 100, f'"train_pred_time" is over 100 seconds'
        assert test_pred_time < 100, f'"test_pred_time" is over 100 seconds'

    def test_recall(self):
        """Test the recall score"""
        fp_IOKR = fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        recall_test = recall_score(fp_IOKR['Y_test'], fp_IOKR['Y_pred_test'])
        recall_train = recall_score(fp_IOKR['Y_train'], fp_IOKR['Y_pred_train'])
        threshold = 0
        assert recall_test > threshold, f'recall_test = {recall_test}, but threshold set to {threshold}'
        assert recall_train > threshold, f'recall_train = {recall_train}, but threshold set to {threshold}'

    def test_precision(self):
        """Tests the precision score"""
        fp_IOKR = fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        precision_test = precision_score(fp_IOKR['Y_test'], fp_IOKR['Y_pred_test'])
        precision_train = precision_score(fp_IOKR['Y_train'], fp_IOKR['Y_pred_train'])
        threshold = 0
        assert precision_test > threshold, f'precision_test = {precision_test}, but threshold set to {threshold}'
        assert precision_train > threshold, f'precision_train = {precision_train}, but threshold set to {threshold}'

    def test_f1_score(self):
        """Tests the F1_score"""
        fp_IOKR = fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        f1_train = f1_score(fp_IOKR['Y_pred_train'], fp_IOKR['Y_train'], average='samples')
        f1_test = f1_score(fp_IOKR['Y_pred_test'], fp_IOKR['Y_test'], average='samples')
        threshold = 0
        # Check f1 score range
        assert 1.0 >= f1_train >= 0.0, f"f1 score is of {f1_train}, should be between 0 and 1"
        assert 1.0 >= f1_test >= 0.0, f"f1 score is of {f1_test}, should be between 0 and 1"
        #assert f1 score is enough
        assert f1_test > threshold, f'f1_test = {f1_test}, but threshold set to {threshold}'
        assert f1_train > threshold, f'f1_train = {f1_train}, but threshold set to {threshold}'

    def test_accuracy_score(self):
        """Check accuracy of the model"""
        fp_IOKR = fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        accuracy = accuracy_score(fp_IOKR['Y_test'], fp_IOKR['Y_pred_test'])
        threshold = 0
        assert accuracy > threshold, f'accuracy = {accuracy}, but threshold set to {threshold} '

    def test_mse(self):
        """Checks the MSE score of the model"""
        fp_IOKR = fitted_predicted_IOKR(X, y, L=1e-5, sx=1000, sy=10)
        train_mse = MSE(fp_IOKR['Y_train'], fp_IOKR['Y_pred_train'])
        test_mse = MSE(fp_IOKR['Y_test'], fp_IOKR['Y_pred_test'])
        threshold = 0
        assert train_mse > threshold, f'accuracy = {train_mse}, but threshold set to {threshold} '
        assert test_mse > threshold, f'accuracy = {test_mse}, but threshold set to {threshold} '


class TestAlphaTrain:

    def test_alpha_train_returns(self):
        test_size= 0.33
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        clf = iokr()
        #clf.verbose = 1
        A = clf.Alpha_train(X_test)
        good_shape = (int(y.shape[0] * test_size), int(y.shape[0] * test_size))
        assert A != None, f"A is None"
        assert A != "", f"A is empty"
        assert isinstance(A, np.ndarray), f"A should be 'np.ndarray', but is {type(A)}"
        assert A.shape == good_shape, f"Shape of A should be of {good_shape}, but is {A.shape}"



