"""
Classification.py

"""

import os
import pickle
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from arxiv_public_data.oai_metadata import load_metadata
from arxiv_public_data.embeddings.tf_hub import load_embeddings

def in_top_n(prob, target, n=5):
    intopn = 0
    labels = np.arange(prob.shape[1])
    for p, t in zip(prob, target):
        if t in sorted(labels, key = lambda i: p[i])[-n:]:
            intopn += 1
    return intopn/prob.shape[0]

EMBDIR = '/pool0/arxiv/embeddings'
usel_abstract = os.path.join(EMBDIR, 'abstract-embedding-usel-2019-03-19.pkl')
usel_title = os.path.join(EMBDIR, 'title-embedding-usel-2019-03-19.pkl')
md_file = ('/home/colin/work/arxiv-public-datasets/data/'
           'oai-arxiv-metadata-2019-03-01.json.gz')

def maincat(name):
    if '.' in name:
        return name.split('.')[0]
    return name

def shuffle(arr, seed=14850):
    """ Deterministic in-place shuffling """
    rng = np.random.RandomState(seed)
    rng.shuffle(arr)

abstract_vec = load_embeddings(usel_abstract)
title_vec = load_embeddings(usel_title)
order = np.arange(len(title_vec))
shuffle(abstract_vec)
shuffle(title_vec)
shuffle(order)

md = load_metadata(md_file)

cats = [maincat(m['categories'][0].split()[0]) for m in md]
shuffle(cats)

scats = list(set(cats))
labels = {c: l for l, c in enumerate(scats)}
target = [labels[c] for c in cats]

train_size = 1200000
target_train = target[:train_size]
target_test = target[train_size:]
title_train = title_vec[:train_size]
title_test = title_vec[train_size:]
abstract_train = abstract_vec[:train_size]
abstract_test = abstract_vec[train_size:]

combined_train = np.concatenate([title_train, abstract_train], axis=1)
combined_test = np.concatenate([title_test, abstract_test], axis=1)

lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', verbose=1)

results = {}

## First fit on just titles
title_results = {}
print('Fitting title vectors')
lr.fit(title_train, target_train)
prec = np.mean(lr.predict(title_test) == target_test)
prob = lr.predict_proba(title_test)
loglikelihood = np.sum(np.log([p[t] for p, t in zip(prob, target_test)]))
top3 = in_top_n(prob, target_test, 3)
top5 = in_top_n(prob, target_test, 5)
results['titles'] = dict(top1=prec, top3=top3, top5=top5,
                         loglikelihood=loglikelihood)
print('title vectors done!')

## Next fit on just the abstracts
print('Fitting abstract vectors')
lr.fit(abstract_train, target_train)
prec = np.mean(lr.predict(abstract_test) == target_test)
prob = lr.predict_proba(abstract_test)
loglikelihood = np.sum(np.log([p[t] for p, t in zip(prob, target_test)]))
top3 = in_top_n(prob, target_test, 3)
top5 = in_top_n(prob, target_test, 5)
results['abstracts'] = dict(top1=prec, top3=top3, top5=top5,
                            loglikelihood=loglikelihood)
print('abstract vectors done!')

# Now on the combination of titles and abstracts
print('Fitting abstract + title vectors')
lr.fit(combined_train, target_train)
prec = np.mean(lr.predict(combined_test) == target_test)
prob = lr.predict_proba(combined_test)
loglikelihood = np.sum(np.log([p[t] for p, t in zip(prob, target_test)]))
top3 = in_top_n(prob, target_test, 3)
top5 = in_top_n(prob, target_test, 5)
results['abstracts+titles'] = dict(
    top1=prec, top3=top3, top5=top5, loglikelihood=loglikelihood
)
print('abstract + title vectors done!')

with open("logistic-regression-embedding-vectors-2019-03-19.pkl", 'wb') as fout:
    pickle.dump(results, fout)
