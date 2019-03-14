#! /usr/bin/env python
import time
import re
import sys
import glob
import os
import gzip
import numpy as np
from multiprocessing import Pool

from regex_arxiv import REGEX_ARXIV_STRICT, REGEX_ARXIV_FLEXIBLE

RE_REMOVE_SPLIT = re.compile(r'(.*)\/\d+\/(.*)')
version_pat = re.compile(r"v(\d+).txt")

def convert_filename_id(s, root):
    s = s.split(root)[1]
    s = '/'.join(RE_REMOVE_SPLIT.match(s).groups())
    if s.startswith('arxiv'):
        s = s.split('arxiv/')[1].split('v')[0]
    else:
        s = s.split('v')[0]
    return s

def all_articles(root):
    all_text_files = sorted(glob.glob(os.path.join(root, '*/*/*.txt*')))
    return [convert_filename_id(s) for s in all_text_files]

def read_article(a):
    fname = read_article(a)

    if not fname:
        return ''

    try:
        with gzip.open(fname) as fn:
            txt = fn.read()
    except:
        with open(fname) as fn:
            txt = fn.read()

    try:
        txt = txt.decode('utf-8')
    except:
        pass
    return txt

def extract_references(txt):
    out = []
    for matches in re.findall(REGEX_ARXIV_FLEXIBLE, txt):
        out.extend([a for a in matches if a])
    return list(set(out))

def citation_list():
    cites = {}
    articles = all_articles()

    t_start = time.time()
    for i, article in enumerate(articles):
        if i % 10 == 0:
            t_end = time.time()
            print(i, article, t_end - t_start)
        text = read_article(article)
        refs = extract_references(text)
        cites[article] = refs

    return cites

def citation_list_inner(articles):
    cites = {}
    for i, article in enumerate(articles):
        try:
            text = read_article(article)
            if text:
                refs = extract_references(text)
                cites[article] = refs
        except:
            continue
    return cites

def citation_list_parallel(N):
    articles = all_articles()

    pool = Pool(N)
    cites = pool.map(citation_list_inner, np.array_split(articles, N))

    allcites = {}
    for c in cites:
        allcites.update(c)
    return allcites

