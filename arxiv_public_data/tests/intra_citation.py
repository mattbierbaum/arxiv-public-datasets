"""
intra_citation.py

author: Colin Clement

TODO
  x Num edges, num nodes
  x Degree distribution
  * Clustering coefficient
  * Average path length, network diameter
  * Betweenness distribution, mean betweenness
  * Connected components

"""

import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import networkx as nx
import numpy as np
import json
import gzip

from arxiv_public_data.oai_metadata import load_metadata
from arxiv_public_data.regex_arxiv import strip_version, format_cat

def loaddata(fname='data/internal-references.json.gz'):
    return json.load(gzip.open(fname, 'r'))

def clean_cite_name(name):
    """ 
    Attempt to clean arxiv id
    Parameters
    ----------
    name : str
        input arvix id
    Returns
    -------
    cleaned_name : str
        can fix names with version numbers,
        e.g. astro-ph/040842v1 -> astro-ph/040842
        and subcategory modifications,
        e.g. 
    """
    return strip_version(format_cat(name))

def bad_ids(data, clean=True):
    """ Return arxiv ids cited which do not exist """
    articles = set(data.keys())
    cited = set(chain(*[v for k, v in data.items()]))
    if clean:
        cited = set(map(clean_cite_name, cited))
    return cited.difference(articles)

def makegraph(data, clean=True, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(data.keys())
    for art in data.keys():
        for ref in data[art]:
            ref = clean_cite_name(ref) if clean else ref
            if not ref == art and ref in data:
                G.add_edge(art, ref)
    return G

def biggest_connected_subgraph(G):
    if G.is_directed():
        comps = nx.weakly_connected_components(G)
    else:
        comps = nx.connected_components(G)
    biggest = max(comps, key=len)
    return G.subgraph(biggest)

def plot_degree_distn(G, bins=None, numbins=50):
    if G.is_directed():  # plot in/out
        in_deg = [d for n, d in G.in_degree()]
        out_deg = [d for n, d in G.out_degree()]
        fig, axes = plt.subplots(1, 2, figsize=(12., 4.))
        if bins is None:
            bins = np.logspace(0, np.log(max(max(in_deg), max(out_deg))),
                               numbins+1, base=np.e)
        axes[0].hist(in_deg, bins=bins)
        axes[1].hist(out_deg, bins=bins)
        for ax in axes:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylabel("Counts")
        axes[0].set_xlabel('Number of Citations')
        axes[1].set_xlabel('Number of References')
    else:
        deg = [d for n, d in G.degree()]
        if bins is None:
            bins = np.logspace(0, np.log(max(max(deg), max(deg))),
                               numbins+1, base=np.e)
        fig, axes = plt.subplots(1)
        axes.hist(deg, bins=bins)
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_ylabel('Counts')
        axes.set_xlabel('Degree')

def category_bar_chart(categories):
    count = Counter(categories)
    sortcats = sorted(count.keys(), key=count.get)[::-1]
    plt.bar(range(len(sortcats)), height=[count[c] for c in sortcats])
    plt.xticks(range(len(sortcats)), sortcats, rotation=90)
    plt.ylabel('Number of articles per category')
    plt.tight_layout()
    return plt.gcf(), plt.gca()


if __name__ == '__main__':
    metafile = 'data/arxiv-oai-metadata-2019-02-26.json.gz'
