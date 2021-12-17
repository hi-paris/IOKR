import pytest
from IOKR.data.load_data import load_bibtex
from IOKR.model.model import IOKR
from IOKR.model.utils import project_root
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
#import numpy as np

#Global Variables of interest
path = join(project_root(), "data/bibtex")
X, Y, _, _ = load_bibtex(path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
clf = IOKR()
clf.verbose = 1
L = 1e-5
sx = 1000
sy = 10


class TestFit():

    def test_fit_prints(self, capfd):
        """Test if fit function actually prints something"""
        clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
        out, err = capfd.readouterr()
        assert err == "", f'{err}: need to be fixed'
        assert out != "", f'Fitting Time should have been printed '


class TestPredict():

    def test_predict_used_to_get_f1_scores(self):
        """Test if f1 scores can be obtained with the scores from predict function"""
        clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
        Y_pred_train = clf.predict(X_test=X_train)
        Y_pred_test = clf.predict(X_test=X_test)
        f1_train = f1_score(Y_pred_train, Y_train, average='samples')
        f1_test = f1_score(Y_pred_test, Y_test, average='samples')
        assert f1_train is not None, f'f1_train should not be None'
        assert f1_test is not None, f'f1_test should not be None'
