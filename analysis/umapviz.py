from arxiv_public_data.embeddings.util import load_embeddings, fill_zeros
from arxiv_public_data.oai_metadata import load_metadata
from arxiv_public_data.config import DIR_OUTPUT, LOGGER
from analysis.colors import MAIN_CATS_RGB_DICT, MAIN_CATS

import umap
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from collections import defaultdict

logger = LOGGER.getChild('umapviz')

def embed(features, min_dist=1e-7, **kwargs):
    reducer = umap.UMAP(min_dist=min_dist, **kwargs)
    reducer.fit(features)
    return reducer

def plot_embeddings(emb, colors, cats):
    index_by_cat = defaultdict(list)
    for i, c in enumerate(cats):
        index_by_cat[c].append(i)
    fig, axes = plt.subplots(figsize=(6,6))
    order = list(sorted(index_by_cat, key=lambda c: len(index_by_cat[c])))[::-1]
    for c in order:  # plot biggest categories first so they don't cover up
        idx = np.array(index_by_cat[c])
        axes.scatter(emb.embedding_[idx,0], emb.embedding_[idx,1],
                     c=np.array(colors)[idx], label=c, alpha=0.4, s=10, marker='.')
    axes.axis('off')
    return fig, axes

def legendplot(colors, cats):
    colordict = {}
    for col, cat in zip(colors, cats):
        if not cat in colordict:
            colordict[cat] = col
    maincat_dict = defaultdict(set)
    for cat in cats:
        maincat_dict[MAIN_CATS[cat]].add(cat)

    fig = plt.figure(figsize=(10, 6))
    patches = []
    labels = []
    for mcat in maincat_dict:
        patches.extend(
            [Patch(color=colordict[cat], label=cat) for cat in maincat_dict[mcat]]
        )
        labels.extend(maincat_dict[mcat])
    fig.legend(patches, labels, loc='center', frameon=False,
               ncol=6)
    return fig

EMBDIR = os.path.join(DIR_OUTPUT, 'embeddings')
fulltext_file = os.path.join(
    EMBDIR, 'fulltext-embedding-usel-2-headers-2019-04-05.pkl'
)
abstracts_file = os.path.join(
    EMBDIR, 'abstract-embedding-usel-2019-03-19.pkl'
)
titles_file = os.path.join(
    EMBDIR, 'title-embedding-usel-2019-03-19.pkl'
)

#logger.info('Loading metadata from default location')
#md = load_metadata()
#logger.info('Loaded metadata!')
#
#cats = [m['categories'][0].split()[0] for m in md]
#colors = [MAIN_CATS_RGB_DICT[c] for c in cats]
#
logger.info('Loading fulltext vectors')
fulltext = load_embeddings(fulltext_file, 2)
logger.info('Loaded fulltext vectors!')
#
#logger.info('Loading abstract vectors')
#abstracts = load_embeddings(abstracts_file)['embeddings']
#logger.info('Loaded abstract vectors!')

#logger.info('Loading title vectors')
#titles = load_embeddings(titles_file)['embeddings']
#logger.info('Loaded title vectors!')
