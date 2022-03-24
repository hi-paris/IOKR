# IOKR

![Build Status](https://github.com/hi-paris/IOKR/workflows/pytesting/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[Github All Releases](https://img.shields.io/github/downloads/hi-paris/IOKR/total.svg)]()




This open source Python library provide several methods for Output Kernelization.


Website and documentation: [https://IOKR.github.io/](https://IOKR.github.io/)

Source Code (MIT): [https://github.com/IOKR/IOKR](https://github.com/IOKR/IOKR)


## Installation

The library has been tested on Linux, MacOSX and Windows. It requires a C++ compiler for building/installing the EMD solver and relies on the following Python modules:

- Pandas (>=1.2)
- Numpy (>=1.16)
- Scipy (>=1.0)
- Scikit-learn (>=1.0) 

#### Pip installation


You can install the toolbox through PyPI with:

```console
pip install IOKR
```

#### Anaconda installation with conda-forge

If you use the Anaconda python distribution, POT is available in [conda-forge](https://conda-forge.org). To install it and the required dependencies:

```console
conda install -c conda-forge IOKR
```

#### Post installation check
After a correct installation, you should be able to import the module without errors:

```python
import IOKR
```

## Examples

### Short examples

* Import the toolbox

```python
import IOKR
```

* Run IOKR

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

### Examples and Notebooks

The examples folder contain several examples and use case for the library. The full documentation with examples and output is available on [https://IOKR.github.io/](https://IOKR.github.io/).


## Acknowledgements

This toolbox has been created and is maintained by

* [Hi! PARIS](https://www.hi-paris.fr/)


The contributors to this library are 

* [Florence d'Alché-Buc](https://perso.telecom-paristech.fr/fdalche/) (Researcher)
* [Luc Motte](https://www.telecom-paris.fr/fr/recherche/laboratoires/laboratoire-traitement-et-communication-de-linformation-ltci/les-equipes-de-recherche/signal-statistique-et-apprentissage-s2a/personnes) (Researcher)
* [Tamim El Ahmad](https://www.telecom-paris.fr/fr/recherche/laboratoires/laboratoire-traitement-et-communication-de-linformation-ltci/les-equipes-de-recherche/signal-statistique-et-apprentissage-s2a/personnes) (Researcher)
* [Gaëtan Brison](https://engineeringteam.hi-paris.fr/about-us/) (Engineer)
* [Danaël Schlewer-Becker](https://engineeringteam.hi-paris.fr/about-us/) (Engineer)
* [Awais Sani](https://engineeringteam.hi-paris.fr/about-us/) (Engineer)


## Contributions and code of conduct

Every contribution is welcome and should respect the [contribution guidelines](.github/CONTRIBUTING.md). Each member of the project is expected to follow the [code of conduct](.github/CODE_OF_CONDUCT.md).

## Support

You can ask questions and join the development discussion:

* On the IOKR [slack channel](https://IOKR-toolbox.slack.com)
* On the IOKR [gitter channel](https://gitter.im/IOKR/community)
* On the IOKR [mailing list](https://mail.python.org/mm3/mailman3/lists/IOKR.python.org/)

You can also post bug reports and feature requests in Github issues. Make sure to read our [guidelines](.github/CONTRIBUTING.md) first.

## References

[1] Brouard, C., d'Alché-Buc, F., Szafranski, M. (2013, November). [Semi-supervised Penalized Output Kernel Regression for
Link Prediction](https://hal.archives-ouvertes.fr/hal-00654123/document). 28th International Conference on Machine Learning (ICML 2011),
pp.593–600.

[2] Brouard, C., Szafranski, M., d'Alché-Buc, F. (2016). [Input Output Kernel Regression: Supervised and Semi-Supervised Structured Output Prediction with Operator-Valued Kernels](https://hal.archives-ouvertes.fr/hal-01601982/document).  Journal of Machine
Learning Research, Microtome Publishing.

[3] Brouard, C., Shen, H., Dührkop, K., d'Alché-Buc, F., Böcker, S., & Rousu, J. (2016) [Fast metabolite identification with input output kernel regression](https://hal.inrae.fr/hal-02637720/document).  Bioinformatics, Oxford University
Press (OUP)



![alt text](images/readme_iokr_equations.png)

