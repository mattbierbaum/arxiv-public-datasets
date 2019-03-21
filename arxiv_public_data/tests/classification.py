"""
Classification.py

"""

import os
import pickle
import numpy as np
from datetime import datetime
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from arxiv_public_data.oai_metadata import load_metadata
from arxiv_public_data.tests.intra_citation import loaddata
import arxiv_public_data.tests.cocitation_category_feature as features

def load_embeddings(filename):
    """
    Loads vector embeddings
    Parameters
    ----------
        filename : str
            path to vector embeddings saved by `create_save_embeddings`
    Returns
    -------
        embeddings : array_like
    """
    out = []
    with open(filename, 'rb') as fin:
        while True:
            try:
                out.extend(pickle.load(fin))
            except EOFError as e:
                break 
    return np.array(out)

def in_top_n(prob, target, n=5):
    intopn = 0
    labels = np.arange(prob.shape[1])
    for p, t in zip(prob, target):
        if t in sorted(labels, key = lambda i: p[i])[-n:]:
            intopn += 1
    return intopn/prob.shape[0]

def train_test(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    prec = np.mean(model.predict(X_test) == y_test)
    prob = model.predict_proba(X_test)
    loglikelihood = np.sum(np.log([p[t] for p, t in zip(prob, y_test)]))
    perplexity = 2 ** ( - loglikelihood / len(y_test) / np.log(2))
    top3 = in_top_n(prob, y_test, 3)
    top5 = in_top_n(prob, y_test, 5)
    return dict(top1=prec, top3=top3, top5=top5, loglikelihood=loglikelihood,
                perplexity=perplexity)

EMBDIR = '/pool0/arxiv/embeddings'
usel_abstract = os.path.join(EMBDIR, 'abstract-embedding-usel-2019-03-19.pkl')
usel_title = os.path.join(EMBDIR, 'title-embedding-usel-2019-03-19.pkl')
md_file = ('/home/colin/work/arxiv-public-datasets/data/'
           'oai-arxiv-metadata-2019-03-01.json.gz')
adj_file = ('/home/colin/work/arxiv-public-datasets/data/'
            'internal-references-pdftotext.json.gz')

def maincat(name):
    if '.' in name:
        return name.split('.')[0]
    return name

def shuffle(arr, seed=14850):
    """ Deterministic in-place shuffling """
    rng = np.random.RandomState(seed)
    rng.shuffle(arr)

def ids_cats(md_file, subcats=True):
    md = load_metadata(md_file)
    ids = np.array([m['id'] for m in md], dtype='object')
    shuffle(ids)
    if subcats:
        cats = np.array([m['categories'][0].split()[0] for m in md],
                        dtype='object')
    else:
        cats = np.array([maincat(m['categories'][0].split()[0]) for m in md],
                        dtype='object')
    shuffle(cats)
    return ids, cats

if __name__ == "__main__":
    
    adj = loaddata(adj_file)
    ids, cats = ids_cats(md_file, subcats=True) 
    
    scats = list(set(cats))
    labels = {c: l for l, c in enumerate(scats)}
    target = [labels[c] for c in cats]
    
    train_size = 1200000
    
    ids_train = ids[:train_size]
    ids_test = ids[train_size:]
    
    target_train = target[:train_size]
    target_test = target[train_size:]

    # Features containing cocitation category information
    mc_train, mc_test = features.cocitation_feature(adj, ids_train, ids_test,
                                                    target_train, target_test)

    lr = LogisticRegression(solver='lbfgs', multi_class='multinomial',
                            verbose=1, max_iter=200)
    results = {}
    
    ## First fit on just titles
    title_vec = load_embeddings(usel_title)
    shuffle(title_vec)
    title_train = title_vec[:train_size]
    title_test = title_vec[train_size:]

    print('Fitting title vectors')
    results['titles'] = train_test(lr, title_train, target_train, title_test,
                                   target_test)
    print(results['titles'])
    print('title vectors done!')
    
    ### Next fit on just the abstracts
    abstract_vec = load_embeddings(usel_abstract)
    shuffle(abstract_vec)
    abstract_train = abstract_vec[:train_size]
    abstract_test = abstract_vec[train_size:]

    print('Fitting abstract vectors')
    results['abstracts'] = train_test(lr, abstract_train, target_train,
                                      abstract_test, target_test)
    print(results['abstracts'])
    print('abstract vectors done!')
    
    # Now on the combination of titles and abstracts
    title_abstract_train = np.concatenate([title_train, abstract_train], axis=1)
    del title_train
    title_abstract_test = np.concatenate([title_test, abstract_test], axis=1)
    del title_test
    print('Fitting abstract + title vectors')
    results['abstract+titles'] = train_test(
        lr, title_abstract_train, target_train, title_abstract_test, target_test
    )
    print(results['abstract+titles'])
    print('abstract + title vectors done!')

    # Next fit on the cocitation features
    print('Fitting cocitation vectors')
    results['cocitation'] = train_test(lr, mc_train, target_train,
                                       mc_test, target_test)
    print(results['cocitation'])
    print('cocitation vectors done!')

    # Next fit on the cocitation features
    co_ti_ab_train = np.concatenate([title_abstract_train, mc_train], axis=1)
    del title_abstract_train
    co_ti_ab_test = np.concatenate([title_abstract_test, mc_test], axis=1)
    del title_abstract_test
    
    print('Fitting title+abstract+cocitation vectors')
    results['cocitation+title+abstracts'] = train_test(
        lr, co_ti_ab_train, target_train, co_ti_ab_test, target_test
    )
    print(results['cocitation+title+abstracts'])
    print('title+abstract+cocitation vectors done!')
    
    nowdate = str(datetime.now()).split()[0]
    filename = "logistic-regression-embedding-vectors-{}.pkl".format(nowdate)
    with open(filename, 'wb') as fout:
        pickle.dump(results, fout)
