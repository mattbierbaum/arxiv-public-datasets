"""
cocitation_category_feature.py

This module creates feature vectors for articles which combine co-citation and
categories.

the cocitation matrix M[i,j] = 1 if article i cites article j, 0 else
the category matrix C[i,c] = 1 if article i is in category c

we can produce features of the data which include graph topology by then
contracting these matrices:
    F[i, c] = sum_j M[i,j] C[j, c]

Note: the training set cannot cite articles in the test set, as that would
contaminate the predictive information of the test set, however the test set can 
cite the training set, otherwise it would be very incomplete. Therefore in
`cocitation_matrix` the test cocitation matrix is rectangular with shape
(N_test,N_test+N_train).
"""

import numpy as np
from scipy.sparse import csr_matrix, vstack
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

    data, row, col = [], [], []
    for idx, i in enumerate(train_list):
        n = 0
        for c in adj.get(i, ''):
            if (not c == i) and (c in train_idx):
                n += 1
                row.append(idx)
                col.append(train_idx[c])
        if n:
            data.extend([1/n] * n if normalize else [1.] * n)
    train_cocitation = csr_matrix((data, (row, col)), shape=(N_train, N_train))

    # test set can cite training set, so this citation matrix is not square!
    train_idx.update({i: idx+N_train for idx, i in enumerate(test_list)})
    data, row, col = [], [], []
    for idx, i in enumerate(test_list):
        n = 0
        for c in adj.get(i, ''):
            if (not c == i) and (c in train_idx):
                n += 1
                row.append(idx)
                col.append(train_idx[c])
        if n:
            data.extend([1/n] * n if normalize else [1.] * n)
    test_cocitation = csr_matrix(
        (data, (row, col)), shape=(N_test, N_test+N_train)
    )
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

def cocitation_feature(adj, train_list, test_list, target_train, target_test,
                       normalize=True):
    m_train, m_test = cocitation_matrix(adj, train_list, test_list, normalize)
    c_train, c_test = category_matrix(target_train, target_test, normalize)
    return (m_train.dot(c_train).todense(), 
            m_test.dot(vstack([c_train, c_test])).todense())
