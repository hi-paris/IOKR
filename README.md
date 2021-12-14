# IOKR

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

```python
--------------------------------
### Install Requirements (dependencies packages)
pip install -r requirements.txt

--------------------------------
### Upgrade your pip command to avoid problems
python -m pip install --upgrade pip

--------------------------------
### Upgrade setuptools to run setup file
pip install --upgrade setuptools

--------------------------------
### Create the local package
pip install -e .

--------------------------------
### Install IOKR
pip install IOKR
```

Once this is done you should see the following message

![Capture d’écran 2021-12-13 à 01.30.55.png](How%20to%20Run%20IOKR%20Locally%20a452af54962b45fc83bdd1a9b8174874/Capture_decran_2021-12-13_a_01.30.55.png)

## 03 Test packages

You can run the following commands in your python console in Pycharm

```python
from IOKR.data.data_load import load_bibtex
from IOKR.model.model import IOKR
from sklearn.model_selection import train_test_split

path = "IOKR/data/bibtex"
X, Y, _, _ = load_bibtex(path)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
	 test_size=0.33, random_state=42)

clf = IOKR()
clf.verbose = 1
L = 1e-5
sx = 1000
sy = 10

clf.fit(X=X_train, Y=Y_train, L=L, sx=sx, sy=sy)
Y_pred_train = clf.predict(X_train=X_train)
Y_pred_test = clf.predict(X_test=X_test)
f1_train = f1_score(Y_pred_train, Y_train, average='samples')
f1_test = f1_score(Y_pred_test, Y_test, average='samples')
```

You should get something like that:

![Capture d’écran 2021-12-13 à 09.16.18.png](How%20to%20Run%20IOKR%20Locally%20a452af54962b45fc83bdd1a9b8174874/Capture_decran_2021-12-13_a_09.16.18.png)

<aside>
✅ **Great Job !**

</aside>
