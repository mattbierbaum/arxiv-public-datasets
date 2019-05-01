"""
util.py

author: Colin Clement
date: 2019-04-05

This module contains helper functions for loading embeddings and batch
loading the full text, since many computers cannot contain the whole
fulltext in memory.
"""

import os
import re
import numpy as np
import pickle

from arxiv_public_data.config import DIR_FULLTEXT, DIR_OUTPUT
from arxiv_public_data.oai_metadata import load_metadata

def id_to_pathname(aid):
    """ 
    Make filename path for text document, matching the format of fulltext 
    creation in `s3_bulk_download`
    Parameters
    ----------
        aid : str
            string of arXiv article id as found in metadata
    Returns
    -------
        pathname : str
            pathname in which to store the article following
    Examples
    --------
    >>> id_to_pathname('hep-ph/0001001')  #doctest: +ELLIPSIS
    '.../hep-ph/0001/hep-ph0001001.txt'

    >>> id_to_pathname('1501.13851')  #doctest: +ELLIPSIS
    '.../arxiv/1501/1501.13851.txt'
    """
    if '.' in aid:  # new style ArXiv ID
        yymm = aid.split('.')[0]
        return os.path.join(DIR_FULLTEXT, 'arxiv', yymm, aid + '.txt')

    # old style ArXiv ID
    cat, arxiv_id = re.split(r'(\d+)', aid)[:2]
    yymm = arxiv_id[:4]
    return os.path.join(DIR_FULLTEXT, cat, yymm, aid.replace('/', '') + '.txt')

def load_generator(paths, batchsize):
    """
    Creates a generator object for batch loading files from paths
    Parameters
    ----------
        paths : list of filepaths
        batchsize : int
    Returns
    -------
        file_contents : list of strings of contents of files in path
    """
    assert type(paths) is list, 'Requires a list of paths'
    assert type(batchsize) is int, 'batchsize must be an int'
    assert batchsize > 0, 'batchsize must be positive'

    out = []
    for p in paths:
        with open(p, 'r') as fin:
            out.append(fin.read())
        if len(out) == batchsize:
            yield np.array(out, dtype='object')
            out = []
    yield out

def batch_fulltext(batchsize=32, maxnum=None):
    """
    Read metadata and find corresponding files in the fulltext
    Parameters
    ----------
        (optional)
        batchsize : int
            number of fulltext files to load into a batch
        maxnum : int
            the maximum number of paths to feed the generator, for
            testing purposes
    Returns
    -------
        md_index, all_ids, load_gen : tuple of (list, list, generator)
           md_index is a mapping of existing fulltext files, in order
           of their appearance, and containing the index of corresponding
           metadata. all_ids is a list of all arXiv IDs in the metadata.
           load_gen is a generator which allows batched loading of the
           full-text, as defined by `load_generator`
    """
    all_ids = [m['id'] for m in load_metadata()]
    all_paths = [id_to_pathname(aid) for aid in all_ids]
    exists = [os.path.exists(p) for p in all_paths]
    existing_paths = [p for p, e in zip(all_paths, exists) if e][:maxnum]
    md_index = [i for i, e in enumerate(exists) if e] 
    return md_index, all_ids, load_generator(existing_paths, batchsize)

def load_embeddings(filename, headers=0):
    """
    Loads vector embeddings
    Parameters
    ----------
        filename : str
            path to vector embeddings saved by `create_save_embeddings`
        (optional)
        headers : int
            number of pickle calls containing metadata separate from the graphs
    Returns
    -------
        embeddings : dict
            keys 'embeddings' containing vector embeddings and
            'headers' containining metadata
    """
    out = {'embeddings': [], 'headers': []}
    N = 0
    with open(filename, 'rb') as fin:
        while True:
            try:
                if N < headers:
                    out['headers'].append(pickle.load(fin))
                else:
                    out['embeddings'].extend(pickle.load(fin))
            except EOFError as e:
                break 
            N += 1
    out['embeddings'] = np.array(out['embeddings'])
    return out

def fill_zeros(loaded_embedding):
    """
    Fill out zeros in the full-text embedding where full-text is missing
    Parameters
    ----------
        loaded_embedding : dict
            dict as saved from with `load_embeddings` with 2 headers
            of the list of the metadata_index each embedding vector corresponds
            to, the list of all article ids
    Returns
    -------
        embeddings : array_like
            vector embeddings of shape (number of articles, embedding dimension)
    """
    md_index = loaded_embedding['headers'][0]
    all_ids = loaded_embedding['headers'][1]
    vectors = loaded_embedding['embeddings']
    output = np.zeros((len(all_ids), vectors.shape[1]))
    for idx, v in zip(md_index, vectors):
        output[idx,:] = v
    return output
