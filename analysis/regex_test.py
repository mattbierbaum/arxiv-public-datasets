"""
regex_test.py

author: Colin Clement
date: 2019-03-16

This module samples the fulltext of the arxiv, pulls out some arxiv IDs, and
then checks these IDs against valid ones in our set of metadata, producing
a report of bad id's found so that we can improve the citation extraction.
"""


import os
import re
import numpy as np
import arxiv_public_data.regex_arxiv as ra

RE_FLEX = re.compile(ra.REGEX_ARXIV_FLEXIBLE)

def strip_version(name):
    return name.split('v')[0]

def format_cat(name):
    """ Strip subcategory, add hyphen to category if missing """
    if '/' in name:  # OLD ID, names contains subcategory 
        catsubcat, aid = name.split('/')
        cat = catsubcat.split('.')[0] 
        return ra.dashdict.get(cat, cat) + "/" + aid
    else:
        return name

def zeropad_1501(name):
    """ Arxiv IDs after yymm=1501 are padded to 5 zeros """
    if not '/' in name:  # new ID
        yymm, num = name.split('.')
        if int(yymm) > 1500 and len(num) < 5:
            return yymm + ".0" + num
    return name

def clean(name):
    funcs = [strip_version, format_cat, zeropad_1501]
    for func in funcs:
        name = func(name)
    return name

def get_alltxt(directory='/pool0/arxiv/full-text'):
    out = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if 'txt' in f:
                out.append(os.path.join(root, f))
    return out

def sample(out, num):
    return [out[s] for s in np.random.randint(0, len(out)-1, num)]

def all_matches(filename, pattern=RE_FLEX):
    out = []
    matches = pattern.findall(open(filename, 'r').read())
    for match in matches:
        for group in match:
            if group:
                out.append(group)
    return out

def all_matches_context(filename, pattern=RE_FLEX, pad=10):
    out = []
    contents = open(filename, 'r').read()
    match = pattern.search(contents)
    while match is not None:
        s, e = match.start(), match.end()
        out.append(contents[max(s-pad,0):e+pad])
        contents = contents[e:]
        match = pattern.search(contents)
    return out 

def test_samples(samples, valid_ids, directory='/pool0/arxiv/full-text', 
                 pattern=RE_FLEX, showpasses=False):
    failures = dict()
    n_matches = 0
    n_failures = 0
    for i, s in enumerate(samples):
        matches = all_matches(s, pattern)
        n_matches += len(matches)
        valid = [clean(m) in valid_ids for m in matches]
        if not all(valid):
            failures[s] = all_matches_context(s, RE_FLEX)
            print("{}: BAD match in {}".format(i, s))
            for v, c in zip(valid, failures[s]):
                if not v:
                    n_failures += 1
                    print("\t{}".format(c))
        else:
            if showpasses:
                print("{}: {} had no match errors".format(i, s))
    error_rate = n_failures/n_matches
    print("Error rate from {} matches is {}".format(n_matches, error_rate))
    return failures


if __name__=="__main__":
    from arxiv_public_data.oai_metadata import load_metadata
    md_file = 'data/oai-arxiv-metadata-2019-03-01.json.gz'
    valid_ids = [m['id'] for m in load_metadata(md_file)]
    samples = sample(get_alltxt(), 10000)
    failures = test_samples(samples, valid_ids)
