from IOKR.data.load_data import load_bibtex
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
