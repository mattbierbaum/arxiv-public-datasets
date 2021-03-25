#! /usr/bin/env python
import time
import re
import sys
import glob
import os
import gzip
import json
import math
from multiprocessing import Pool,cpu_count

from arxiv_public_data.regex_arxiv import REGEX_ARXIV_FLEXIBLE, clean
from arxiv_public_data.config import DIR_FULLTEXT, DIR_OUTPUT, LOGGER

log = LOGGER.getChild('fulltext')
RE_FLEX = re.compile(REGEX_ARXIV_FLEXIBLE)
RE_OLDNAME_SPLIT = re.compile(r"([a-z\-]+)(\d+)")


def path_to_id(path):
    """ Convert filepath name of ArXiv file to ArXiv ID """
    name = os.path.splitext(os.path.basename(path))[0]
    if '.' in name:  # new  ID
        return name
    split = [a for a in RE_OLDNAME_SPLIT.split(name) if a]
    return "/".join(split)


def all_articles(directory=DIR_FULLTEXT):
    """ Find all *.txt files in directory """
    out = []
    # make sure the path is absolute for os.walk
    directory = os.path.abspath(os.path.expanduser(directory))

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
    with open(filename, 'r') as fn:
        txt = fn.read()

        for matches in pattern.findall(txt):
            out.extend([clean(a) for a in matches if a])
    return list(set(out))

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
        if i > 0 and i % 1000 == 0:
            log.info('Completed {} articles'.format(i))
        try:
            refs = extract_references(article)
            cites[path_to_id(article)] = refs
        except:
            log.error("Error in {}".format(article))
            continue
    return cites


def citation_list_parallel(N=cpu_count(), directory=DIR_FULLTEXT):
    """
    Split the task of checking for citations across some number of processes
    Parameters
    ----------
        N : int
            number of processes
        directory: str
            directory where full text files are stored
    Returns
    -------
        citations : dict[arXiv ID] = list of arXiv IDs
            all arXiv citations in all articles
    """
    articles = all_articles(directory)
    log.info('Calculating citation network for {} articles'.format(len(articles)))

    pool = Pool(N)

    A = len(articles)
    divs = list(range(0, A, math.ceil(A/N))) + [A]
    chunks = [articles[s:e] for s, e in zip(divs[:-1], divs[1:])]

    cites = pool.map(citation_list_inner, chunks)

    allcites = {}
    for c in cites:
        allcites.update(c)
    return allcites


def default_filename():
    return os.path.join(DIR_OUTPUT, 'internal-citations.json.gz')


def save_to_default_location(citations):
    filename = default_filename()

    log.info('Saving to "{}"'.format(filename))
    with gzip.open(filename, 'wb') as fn:
        fn.write(json.dumps(citations).encode('utf-8'))
