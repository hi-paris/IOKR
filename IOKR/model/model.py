# Implementation
import time
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import f1_score
import numpy as np
import pandas as pd
#from IOKR.data.load_data import load_bibtex
from sklearn.model_selection import train_test_split
import arff
import os
from line_profiler import LineProfiler


# insert at position 1 in the path, as 0 is the path of this file.

#dir_path = os.path.dirname("/Users/gaetanbrison/Documents/GitHub/IOKR/IOKR/data/bibtex")
#dir_path = os.path.dirname(os.path.realpath(__file__))
#dataset = arff.load(open('/Users/gaetanbrison/Documents/GitHub/IOKR/IOKR/data/bibtex/bibtex.arff'), "r")

"""
Created on December 12, 2021
"""


class IOKR:
    """
    Class used to apply Input and Output Kernel Regression
    """
#    @profile
    def __init__(self):
        """
        Initialization of the below parameters.

        Parameters
        ----------
        X_train :  sparse matrix - containing explanatory variables of the train set
        Y_train: sparse matrix - containing target variable of the train set
        Ky: output scalar kernel
        M: gram matrix on training set
        verbose: binary - display more parameters
        linear:

        """
        self.X_train = None
        self.Y_train = None
        self.Ky = None
        self.sy = None
        self.M = None
        self.verbose = 0
        self.output_kernel = None

#    @profile
    def fit(self, X, Y, L, sx, sy):
        """
        Model Fitting

        """

        t0 = time.time()
        self.X_train = X
        self.sx = sx
        Kx = rbf_kernel(X, X, gamma=1 / (2 * self.sx))
        n = Kx.shape[0]
        self.M = np.linalg.inv(Kx + n * L * np.eye(n))
        self.Y_train = Y
        
        if self.verbose > 0:
            print(f'Fitting time: {time.time() - t0} in s')
            
    def Alpha_train(self, X_test):
        
        Kx = rbf_kernel(self.X_train, X_test, gamma=1 / (2 * self.sx))
        A = self.M.dot(Kx)
        
        return A

#    @profile
    def predict(self, X_test, Y_candidates):

        """
        Model Prediction

        """
        

        t0 = time.time()
        
        A = self.Alpha_train(X_test)
        Ky = self.output_kernel(self.Y, Y_candidates)
        scores = self.Ky.dot(A)
        
        idx_pred = np.argmax(scores, axis=0)
        
        if self.verbose > 0:
            print(f'Decoding time: {time.time() - t0} in s')
            
        Y_pred = Y_candidates[idx_pred]
        
        return Y_pred




# path = "/Users/gaetanbrison/Documents/GitHub/IOKR/IOKR/data/bibtex"
# X, Y, _, _ = load_bibtex(path)
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
# clf = IOKR()
# clf.verbose = 1
# L = 1e-5
# sx = 1000
# sy = 10
#
# clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
# Y_pred_train = clf.predict(X_train=X_train)
# Y_pred_test = clf.predict(X_test=X_test)
# f1_train = f1_score(Y_pred_train, Y_train, average='samples')
# f1_test = f1_score(Y_pred_test, Y_test, average='samples')
#
# print(f'Train f1 score: {f1_train} / Test f1 score {f1_test}')
