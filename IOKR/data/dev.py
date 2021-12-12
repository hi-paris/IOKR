import arff
import os
import numpy as np
import pandas as pd
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from line_profiler import LineProfiler


## bibtex
### files (sparse): Train and test sets along with their union and the XML header [bibtex.rar]
### source: I. Katakis, G. Tsoumakas, I. Vlahavas, "Multilabel Text Classification for Automated Tag Suggestion",
### Proceedings of the ECML/PKDD 2008 Discovery Challenge, Antwerp, Belgium, 2008.


# split dataset using

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_bibtex(dir_path: str):
    """
    Load the bibtex dataset.
    __author__ = "Michael Gygli, ETH Zurich"
    from https://github.com/gyglim/dvn/blob/master/mlc_datasets/__init__.py
    number of labels ("tags") = 159
    dimension of inputs = 1836
    Returns
    -------
    txt_labels (list)
        the 159 tags, e.g. 'TAG_system', 'TAG_social_nets'
    txt_inputs (list)
        the 1836 attribute words, e.g. 'dependent', 'always'
    labels (np.array)
        N x 159 array in one hot vector format
    inputs (np.array)
        N x 1839 array in one hot vector format
    """
    feature_idx = 1836

    dataset = arff.load(open('%s/bibtex.arff' % dir_path), "r")
    data = np.array(dataset['data'], np.int64)

    X = data[:, 0:feature_idx]
    Y = data[:, feature_idx:]

    X_txt = [t[0] for t in dataset['attributes'][:feature_idx]]
    Y_txt = [t[0] for t in dataset['attributes'][feature_idx:]]

    return X, Y, X_txt, Y_txt



path = "../data/bibtex/"
X, Y, _, _ = load_bibtex(path)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)


n_tr = X_train.shape[0]
n_te = X_test.shape[0]
input_dim = X_train.shape[1]
label_dim = Y_train.shape[1]

print(f'Train set size = {n_tr}')
print(f'Test set size = {n_te}')
print(f'Input dim. = {input_dim}')
print(f'Output dim. = {label_dim}')
print(len(Y_test))
print(len(Y_train))










# get_bibtex(dir_path: str, split: str)


# df = pd.DataFrame(dataset)
# data = np.array(dataset['data'], np.int64)
# print(type(data))


import pandas as pd
# fp = open('../data/bibtex/bibtex-train.arff')
# dataset = arff.load(fp)
# data = np.array(dataset['data'], np.int)
# dir_path = "../data/bibtex/"
# dataset = arff.load(open(os.path.join(dir_path, 'bibtex-train.arff'),"r"))
# dataset = list(dataset)
# data = np.array(dataset['data'], np.int)
# for x in dataset:
#    print(x)
# with open("../data/bibtex/bibtex-train.arff", 'r') as f:
#     dataset = arff.load(f)
#     for i in f:
#        print(i)
# with open('../data/bibtex/bibtex-train.arff') as f:
#     df = a2p.load(f)
#     print(df)

# import os
# dir_path = os.path.dirname(os.path.realpath(__file__))
# dataset = arff.load(open('%s/bibtex/bibtex.arff' % dir_path, 'rb'))
# data = np.array(dataset['data'], np.int)

# print(dataset.0)
# def get_bibtex(split='train'):
#     """Load the bibtex dataset."""
#     assert split in ['train', 'test']
#     feature_idx = 1836
#     if split == 'test':
#         dataset = arff.loadarff(open('%s/bibtex/bibtex-test.arff' % dir_path, 'rb')).decode('utf-8')
#     else:
#         dataset = arff.loadarff(open('%s/bibtex/bibtex-train.arff' % dir_path, 'rb')).decode('utf-8')
#     #data = np.array(dataset['data'], np.int)
#     return print(list(dataset))
#     #

# labels = data[:, feature_idx:]
# features = data[:, 0:feature_idx]
# txt_labels = [t[0] for t in dataset['attributes'][1836:]]
# txt_inputs = [t[0] for t in dataset['attributes'][:1836]]

# if split == 'train':
#    return labels, features, txt_labels
# else:
#    return labels, features, txt_labels, txt_inputs

# get_bibtex()


# Load the data using "arff.loadarff" then convert it to dataframe
# data = arff.loadarff('../data/bibtex/bibtex-test.arff')
# df = pd.DataFrame(data[0])


from torch.utils.data import Dataset

# def load_arff_data(data_file_name,path):
#
#     return "It is working"
#
#
# load_arff_data("yo","by")
#
# import torch
# from utils_data import get_bibtex
#
#
#
# df_arff = scipy.io.arff.loadarff("../data/bibtex/bibtex.arff")
# dir_path = "../data/bibtex/"
# dataset = arff.load(open(os.path.join(dir_path, 'bibtex-train.arff')), "r")
# df_arff.head(3)
#
# dataset = arff.load(open("../data/bibtex/bibtex-test.arff"),"r")
# dataset.head(3)
# Load the bibtex dataset
#
#
#
#     return "yo"


# #def load_csv_data(data_file_name,*,data_module=DATA_MODULE,descr_file_name=None,descr_module=DESCR_MODULE):
# #    """Loads `data_file_name` from `data_module with `importlib.resources`.
#     Parameters
#     ----------
#     data_file_name : str
#         Name of csv file to be loaded from `data_module/data_file_name`.
#         For example `'wine_data.csv'`.
#     data_module : str or module, default='sklearn.datasets.data'
#         Module where data lives. The default is `'sklearn.datasets.data'`.
#     descr_file_name : str, default=None
#         Name of rst file to be loaded from `descr_module/descr_file_name`.
#         For example `'wine_data.rst'`. See also :func:`load_descr`.
#         If not None, also returns the corresponding description of
#         the dataset.
#     descr_module : str or module, default='sklearn.datasets.descr'
#         Module where `descr_file_name` lives. See also :func:`load_descr`.
#         The default is `'sklearn.datasets.descr'`.
#     Returns
#     -------
#     data : ndarray of shape (n_samples, n_features)
#         A 2D array with each row representing one sample and each column
#         representing the features of a given sample.
#     target : ndarry of shape (n_samples,)
#         A 1D array holding target variables for all the samples in `data`.
#         For example target[0] is the target variable for data[0].
#     target_names : ndarry of shape (n_samples,)
#         A 1D array containing the names of the classifications. For example
#         target_names[0] is the name of the target[0] class.
#     descr : str, optional
#         Description of the dataset (the content of `descr_file_name`).
#         Only returned if `descr_file_name` is not None.
#     """
#     with resources.open_text(data_module, data_file_name) as csv_file:
#         data_file = csv.reader(csv_file)
#         temp = next(data_file)
#         n_samples = int(temp[0])
#         n_features = int(temp[1])
#         target_names = np.array(temp[2:])
#         data = np.empty((n_samples, n_features))
#         target = np.empty((n_samples,), dtype=int)
#
#         for i, ir in enumerate(data_file):
#             data[i] = np.asarray(ir[:-1], dtype=np.float64)
#             target[i] = np.asarray(ir[-1], dtype=int)
#
#     if descr_file_name is None:
#         return data, target, target_names
#     else:
#         assert descr_module is not None
#         descr = load_descr(descr_module=descr_module, descr_file_name=descr_file_name)
#         return data, target, target_names, descr

#
# def get_bibtex(dir_path: str, use_train: bool):
#     """
#     Load the bibtex dataset.
#     __author__ = "Michael Gygli, ETH Zurich"
#     from https://github.com/gyglim/dvn/blob/master/mlc_datasets/__init__.py
#     number of labels ("tags") = 159
#     dimension of inputs = 1836
#     Returns
#     -------
#     txt_labels (list)
#         the 159 tags, e.g. 'TAG_system', 'TAG_social_nets'
#     txt_inputs (list)
#         the 1836 attribute words, e.g. 'dependent', 'always'
#     labels (np.array)
#         N x 159 array in one hot vector format
#     inputs (np.array)
#         N x 1839 array in one hot vector format
#     """
#     feature_idx = 1836
#     if use_train:
#         dataset = arff.load(open(os.path.join(dir_path, 'bibtex-train.arff')), "r")
#     else:
#         dataset = arff.load(open(os.path.join(dir_path, 'bibtex-test.arff')), "r")
#
#     data = np.array(dataset['data'], np.int)
#
#     labels = data[:, feature_idx:]
#     inputs = data[:, 0:feature_idx]
#     txt_labels = [t[0] for t in dataset['attributes'][feature_idx:]]
#     txt_inputs = [t[0] for t in dataset['attributes'][:feature_idx]]
#     return labels, inputs, txt_labels, txt_inputs
#
# Y_tr, X_tr, _, _ = get_bibtex(PATH_BIBTEX, use_train=True)
# Y_te, X_te, _, _ = get_bibtex(PATH_BIBTEX, use_train=False)
# n_tr = X_tr.shape[0]
# n_te = X_te.shape[0]
# input_dim = X_tr.shape[1]
# label_dim = Y_tr.shape[1]
#
# print(f'Train set size = {n_tr}')
# print(f'Test set size = {n_te}')
# print(f'Input dim. = {input_dim}')
# print(f'Output dim. = {label_dim}')
