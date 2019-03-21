"""
cocitation_category_feature.py

This module creates feature vectors for articles which combines co-citation and
categories
"""

import numpy as np
from scipy.sparse import csr_matrix
from arxiv_public_data.oai_metadata import load_metadata

def cocitation_matrix(adj, train_list, test_list, normalize=True):
    """
    Parameters
    ----------
        adj : dict of list of strings
            adjacency matrix adj[key] = list of connected keys
        train_list : list of strings
            list of IDs in the training set
        test_set : list of strings
            list of IDs in the test set
        (optional)
        normalize: True/False
            replace 1 in each entry with 1/Number of citations
    Returns
    -------
        train_cocitation, test_cocitation : tuple of sparse matrices
    """
    N_train = len(train_list)
    N_test = len(test_list)
    train_idx = {i: idx for idx, i in enumerate(train_list)}
    test_idx = {i: idx for idx, i in enumerate(test_list)}

    data, row, col = [], [], []
    for idx, i in enumerate(train_list):
        n = 0
        for c in adj.get(i, ''):
            if (not c == i) and (not c in test_idx) and c in train_idx:
                n += 1
                row.append(idx)
                col.append(train_idx[c])
        if n:
            data.extend([1/n] * n if normalize else [1.] * n)
    train_cocitation = csr_matrix((data, (row, col)), shape=(N_train, N_train))

    data, row, col = [], [], []
    for idx, i in enumerate(test_list):
        n = 0
        for c in adj.get(i, ''):
            if (not c == i) and (not c in train_idx) and c in test_idx:
                n += 1
                row.append(idx)
                col.append(test_idx[c])
        if n:
            data.extend([1/n] * n if normalize else [1.] * n)
    test_cocitation = csr_matrix((data, (row, col)), shape=(N_test, N_test))
    return train_cocitation, test_cocitation

def category_matrix(target_train, target_test, normalize=False):
    data, row, col = [], [], []
    N_train = len(target_train)
    N_test = len(target_test)
    N_cats = max(target_train) + 1
    cat_train = csr_matrix(
        (np.ones(N_train), (np.arange(N_train), target_train)),
        shape = (N_train, N_cats)
    )
    cat_test = csr_matrix(
        (np.ones(N_test), (np.arange(N_test), target_test)),
        shape = (N_test, N_cats)
    )
    return cat_train, cat_test
