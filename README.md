# IOKR

![Build Status](https://github.com/hi-paris/IOKR/workflows/pytesting/badge.svg)

![Code Coverage](https://github.com/hi-paris/IOKR/workflows/.coverage.svg)

![[Build Status](https://app.travis-ci.com/hi-paris/IOKR.svg?branch=main)](https://app.travis-ci.com/hi-paris/IOKR)

Refs:
- Brouard, C., d'Alché-Buc, F., Szafranski, M. Semi-supervised Penalized Output Kernel Regression for Link Prediction, ICML 2011: 593-600, (2011).
- Brouard, C., Szafranski, M., d'Alché-Buc, F.Input Output Kernel Regression: Supervised and Semi-Supervised Structured Output Prediction with Operator-Valued Kernels. J. Mach. Learn. Res. 17: 176:1-176:48 (2016).
- Brouard, C., Shen, H., Dührkop, K., d'Alché-Buc, F., Böcker, S., & Rousu, J. (2016). Fast metabolite identification with input output kernel regression. Bioinformatics, 32(12), i28-i36 (2016).


![alt text](images/readme_iokr_equations.png)

# How to Run IOKR Locally

## 01 Setup computer

- Github Desktop
- Pycharm

Steps: 

1. On Github Desktop add IOKR repository hosted privately on Hi! PARIS Github [https://github.com/hi-paris](https://github.com/hi-paris)
2. Open the repository in your external editor in this case Pycharm

## 02 Pycharm Terminal Commands

Put yourself in the main directory "IOKR"

```bash
--------------------------------
### Install Requirements (dependencies packages)
pip install -r requirements.txt

--------------------------------
### Upgrade your pip command to avoid problems
python -m pip install --upgrade pip

--------------------------------
### Upgrade setuptools to run setup file
pip install --upgrade setuptools
setup.py install

--------------------------------
### Create the local package
pip install -e .

--------------------------------
### Install IOKR
pip install IOKR
```

Once this is done you should see the following message

![alt text](images/package-iokr.png)

## 03 Test packages

You can run the following commands in your python console in Pycharm

```python
from IOKR.model.model import IOKR
from sklearn.model_selection import train_test_split
from IOKR.data.load_data import load_bibtex
from sklearn.metrics import f1_score

path = "IOKR/data/bibtex"
X, Y, _, _ = load_bibtex(path)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

clf = IOKR()
clf.verbose = 1
L = 1e-5
sx = 1000
sy = 10

clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
Y_pred_train = clf.predict(X_test=X_train)
Y_pred_test = clf.predict(X_test=X_test)
f1_train = f1_score(Y_pred_train, Y_train, average='samples')
f1_test = f1_score(Y_pred_test, Y_test, average='samples')
print("Train f1 score:", f1_train,"/", "Test f1 score:", f1_test)
```

You should get something like that:

![alt text](images/output-iokr.png)

<aside>
✅ **Great Job !**

</aside>
