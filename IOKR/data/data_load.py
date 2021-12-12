
   
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on December 12, 2021
@author: Gaëtan Brison <gaetan.brison@ip-paris.fr>
"""

def load_wine(*, return_X_y=False, as_frame=False):
    """Load and return the wine dataset (classification).
    .. versionadded:: 0.18
    The wine dataset is a classic and very easy multi-class classification
    dataset.
    =================   ==============
    Classes                          3
    Samples per class        [59,71,48]
    Samples total                  178
    Dimensionality                  13
    Features            real, positive
    =================   ==============
    Read more in the :ref:`User Guide <wine_dataset>`.
    Parameters
    ----------
    return_X_y : bool, default=False
        If True, returns ``(data, target)`` instead of a Bunch object.
        See below for more information about the `data` and `target` object.
    as_frame : bool, default=False
        If True, the data is a pandas DataFrame including columns with
        appropriate dtypes (numeric). The target is
        a pandas DataFrame or Series depending on the number of target columns.
        If `return_X_y` is True, then (`data`, `target`) will be pandas
        DataFrames or Series as described below.
        .. versionadded:: 0.23
    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.
        data : {ndarray, dataframe} of shape (178, 13)
            The data matrix. If `as_frame=True`, `data` will be a pandas
            DataFrame.
        target: {ndarray, Series} of shape (178,)
            The classification target. If `as_frame=True`, `target` will be
            a pandas Series.
        feature_names: list
            The names of the dataset columns.
        target_names: list
            The names of target classes.
        frame: DataFrame of shape (178, 14)
            Only present when `as_frame=True`. DataFrame with `data` and
            `target`.
            .. versionadded:: 0.23
        DESCR: str
            The full description of the dataset.
    (data, target) : tuple if ``return_X_y`` is True
    The copy of UCI ML Wine Data Set dataset is downloaded and modified to fit
    standard format from:
    https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    Examples
    --------
    Let's say you are interested in the samples 10, 80, and 140, and want to
    know their class name.
    >>> from sklearn.datasets import load_wine
    >>> data = load_wine()
    >>> data.target[[10, 80, 140]]
    array([0, 1, 2])
    >>> list(data.target_names)
    ['class_0', 'class_1', 'class_2']
    """

    data, target, target_names, fdescr = load_csv_data(
        data_file_name="wine_data.csv", descr_file_name="wine_data.rst"
    )

    feature_names = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280/od315_of_diluted_wines",
        "proline",
    ]

    frame = None
    target_columns = [
        "target",
    ]
    if as_frame:
        frame, data, target = _convert_data_dataframe(
            "load_wine", data, target, feature_names, target_columns
        )

    if return_X_y:
        return data, target

    return Bunch(
        data=data,
        target=target,
        frame=frame,
        target_names=target_names,
        DESCR=fdescr,
        feature_names=feature_names,
    )