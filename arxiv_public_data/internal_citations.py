#! /usr/bin/env python
import time
import re
import sys
import glob
import os
import gzip
import numpy as np
from multiprocessing import Pool

from arxiv_public_data.regex_arxiv import REGEX_ARXIV_FLEXIBLE, clean


OUTDIR = '/pool0/arxiv/full-text'
RE_FLEX = re.compile(REGEX_ARXIV_FLEXIBLE)
RE_OLDNAME_SPLIT = re.compile(r"([a-z\-]+)(\d+)")

def path_to_id(path):
    """ Convert filepath name of ArXiv file to ArXiv ID """
    name = os.path.splitext(os.path.basename(path))[0]
    if '.' in name:  # new  ID
        return name 
    split = [a for a in RE_OLDNAME_SPLIT.split(name) if a]
    return "/".join(split)

def all_articles(directory=OUTDIR):
    """ Find all *.txt files in directory """
    out = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if 'txt' in f:
                out.append(os.path.join(root, f))
    return out

def extract_references(filename, pattern=RE_FLEX):
    """
    Parameters
    ----------
        filename : str
            name of file to search for pattern
        pattern : re pattern object
            compiled regex pattern

    Returns
    -------
        citations : list
            list of found arXiv IDs
    """
    out = []
    for matches in pattern.findall(open(filename, 'r').read()):
        out.extend([clean(a) for a in matches if a])
    return out

def citation_list_inner(articles):
    """ Find references in all the input articles
    Parameters
    ----------
        articles : list of str
            list of paths to article text
    Returns
    -------
        citations : dict[arXiv ID] = list of arXiv IDs
            dictionary of articles and their references
    """
    cites = {}
    for i, article in enumerate(articles):
        try:
            refs = extract_references(article)
            cites[path_to_id(article)] = refs
        except:
            print("Error in {}".format(articles))
            continue
    return cites

def citation_list_parallel(N=8):
    """
    Split the task of checking for citations across some number of processes
    Parameters
    ----------
        N : int
            number of processes
    Returns
    -------
        citations : dict[arXiv ID] = list of arXiv IDs
            all arXiv citations in all articles
    """
    articles = all_articles()

    pool = Pool(N)
    cites = pool.map(citation_list_inner, np.array_split(articles, N))

    allcites = {}
    for c in cites:
        allcites.update(c)
    return allcites
