"""
Custom metrics.
"""

from collections import OrderedDict

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels


def confusion_dataframe(y_true, y_pred,
                        columns=['P', 'N', 'PP', 'NP', 'TP', 'TN', 'FP', 'FN'],
                        orderby='PP'):
    """Builds a confusion dataframe.

    Each row corresponds to a unique method name X. Values of the row
    contain elements of the confusion matrix of binary classification
    with labels ["X", "not X"]. The condition for this classification
    is "name == X".

    P - condition positive
    N - condition negative
    PP - Predicted condition positive
    PN - Predicted condition negative
    TP - True positives
    TN - True negatives
    FP - False positives (type I error)
    FN - False negatives (type II error)

    More information here: https://en.wikipedia.org/wiki/Confusion_matrix

    Parameters
    ----------

    y_true : array-like
        True labels

    y_pred : array-like
        Predicted labels

    columns : array-like
        List of columns to be included in a dataframe in a specified order

    orderby : str or array-like
        Column name or list of names to specify the column(s) by which
        the dataframe should be ordered.

    Returns
    -------

    confusion_df : pd.DataFrame
        Confusion dataframe

    Examples
    --------

    >>> from metrics import confusion_dataframe
    >>> y_true = ['cat', 'dog', 'mouse', 'cat']
    >>> y_pred = ['cat', 'mouse', 'dog', 'dog']
    >>> confusion_dataframe(y_true)
           P  N  PP  NP  TP  TN  FP  FN
    dog    1  3   2   2   0   1   2   1
    cat    2  2   1   3   1   2   0   1
    mouse  1  3   1   3   0   2   1   1

    Order by true positives and then predicted positives and show only
    false negatives and false positives in the specified order.

    >>> confusion_dataframe(y_true, y_pred, columns=['FN', 'FP'], orderby=['TP', 'PP'])
           FN  FP
    cat     1   0
    dog     1   2
    mouse   1   1
    """
    # Converting all labels to str
    y_true = np.array(y_true, dtype=str)
    y_pred = np.array(y_pred, dtype=str)

    confusion = confusion_matrix(y_true, y_pred)
    labels = unique_labels(y_true, y_pred)

    P = confusion.sum(axis=1)
    PP = confusion.sum(axis=0)
    TP = confusion.diagonal()
    N = len(y_true) - P
    NP = len(y_pred) - PP
    FP = PP - TP
    TN = N - FP
    FN = NP - TN

    confusion_df = pd.DataFrame(OrderedDict([
        ('name', labels),
        ('P', P),
        ('N', N),
        ('PP', PP),
        ('NP', NP),
        ('TP', TP),
        ('TN', TN),
        ('FP', FP),
        ('FN', FN)
    ]))

    confusion_df = confusion_df.set_index('name')
    del confusion_df.index.name

    return confusion_df.sort_values(orderby, ascending=False)[columns]