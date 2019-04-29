"""
Classification.py

"""

import os
import json
import gzip
import json
import numpy as np
from datetime import datetime
from sklearn.linear_model import LogisticRegression, SGDClassifier

from arxiv_public_data.embeddings.util import load_embeddings, fill_zeros
import arxiv_public_data.tests.cocitation_category_feature as features
from arxiv_public_data.config import DIR_OUTPUT, DIR_BASE, LOGGER
from arxiv_public_data.oai_metadata import load_metadata

logger = LOGGER.getChild('lr-classify')

def loaddata(fname='data/internal-references.json.gz'):
    return json.load(gzip.open(fname, 'r'))

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

EMBDIR = os.path.join(DIR_OUTPUT, 'embeddings')
usel_abstract = os.path.join(EMBDIR, 'abstract-embedding-usel-2019-03-19.pkl')
usel_title = os.path.join(EMBDIR, 'title-embedding-usel-2019-03-19.pkl')
usel_fulltext = os.path.join(
    EMBDIR, 'fulltext-embedding-usel-2-headers-2019-04-05.pkl'
)

md_file = os.path.join(DIR_BASE, 'arxiv-metadata-oai-2019-03-01.json.gz')
adj_file = os.path.join(DIR_OUTPUT, 'internal-citations.json.gz')

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

    model_kwargs = dict(loss='log', tol=1e-6, max_iter=50, alpha=1e-7,
                        verbose=False, n_jobs=6)
    results = {}
    
    # JUST cocitation features
    logger.info('Fitting cocitation vectors')
    lr = SGDClassifier(**model_kwargs)
    results['cocitation'] = train_test(lr, mc_train, target_train,
                                       mc_test, target_test)
    logger.info(results['cocitation'])
    logger.info('cocitation vectors done!')

    # JUST full text
    fulltext_vec = fill_zeros(load_embeddings(usel_fulltext, headers=2))
    shuffle(fulltext_vec)
    fulltext_train = fulltext_vec[:train_size]
    fulltext_test = fulltext_vec[train_size:]

    logger.info('Fitting fulltext vectors')
    lr = SGDClassifier(**model_kwargs)
    results['fulltext'] = train_test(lr, fulltext_train, target_train,
                                     fulltext_test, target_test)
    logger.info(results['fulltext'])
    logger.info('fulltext vectors done!')

    # JUST titles
    title_vec = load_embeddings(usel_title)['embeddings']
    shuffle(title_vec)
    title_train = title_vec[:train_size]
    title_test = title_vec[train_size:]

    logger.info('Fitting title vectors')
    lr = SGDClassifier(**model_kwargs)
    results['titles'] = train_test(lr, title_train, target_train, title_test,
                                   target_test)
    logger.info(results['titles'])
    logger.info('title vectors done!')
    
    # JUST abstracts
    abstract_vec = load_embeddings(usel_abstract)['embeddings']
    shuffle(abstract_vec)
    abstract_train = abstract_vec[:train_size]
    abstract_test = abstract_vec[train_size:]

    logger.info('Fitting abstract vectors')
    lr = SGDClassifier(**model_kwargs)
    results['abstracts'] = train_test(lr, abstract_train, target_train,
                                      abstract_test, target_test)
    logger.info(results['abstracts'])
    logger.info('abstract vectors done!')
   
    # ALL features
    logger.info('Fitting all features')
    lr = SGDClassifier(**model_kwargs)
    results['all'] = train_test(
        lr,
        np.concatenate(
            [title_train, abstract_train, mc_train, fulltext_train], axis=1
        ),
        target_train,
        np.concatenate(
            [title_test, abstract_test, mc_test, fulltext_test], axis=1
        ),
        target_test
    )
    logger.info(results['all'])
    logger.info('all features done!')

    #
    # Now feature ablations (individual removals
    #
 
    # ALL - titles
    logger.info('Fitting all - titles')
    lr = SGDClassifier(**model_kwargs)
    results['all - titles'] = train_test(
        lr,
        np.concatenate(
            [abstract_train, mc_train, fulltext_train], axis=1
        ),
        target_train,
        np.concatenate(
            [abstract_test, mc_test, fulltext_test], axis=1
        ),
        target_test
    )
    logger.info(results['all - titles'])
    logger.info('all - titles done!')

    # ALL - abstracts
    logger.info('Fitting all - abstracts')
    lr = SGDClassifier(**model_kwargs)
    results['all - abstracts'] = train_test(
        lr,
        np.concatenate(
            [title_train, mc_train, fulltext_train], axis=1
        ),
        target_train,
        np.concatenate(
            [title_test, mc_test, fulltext_test], axis=1
        ),
        target_test
    )
    logger.info(results['all - abstracts'])
    logger.info('all - abstracts done!')

    # ALL - cocitation
    logger.info('Fitting all - cocitation')
    lr = SGDClassifier(**model_kwargs)
    results['all - cocitation'] = train_test(
        lr,
        np.concatenate(
            [title_train, abstract_train, fulltext_train], axis=1
        ),
        target_train,
        np.concatenate(
            [title_test, abstract_test, fulltext_test], axis=1
        ),
        target_test
    )
    logger.info(results['all - cocitation'])
    logger.info('all - cocitation done!')

    # ALL - fulltext
    logger.info('Fitting all features')
    lr = SGDClassifier(**model_kwargs)
    results['all - fulltext'] = train_test(
        lr,
        np.concatenate(
            [title_train, abstract_train, mc_train], axis=1
        ),
        target_train,
        np.concatenate(
            [title_test, abstract_test, mc_test], axis=1
        ),
        target_test
    )
    logger.info(results['all - fulltext'])
    logger.info('all - fulltext done!')

    # SAVE
    nowdate = str(datetime.now()).split()[0]
    filename = "logistic-regression-classification-{}.json".format(nowdate)
    with open(os.path.join(DIR_OUTPUT, filename), 'w') as fout:
        json.dump(results, fout)
