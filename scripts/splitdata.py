"""
splitdata.py
"""

import os
import logging
import random
from pathlib import Path

from arxiv_public_data.config import LOGGER, DIR_BASE, DIR_OUTPUT
from arxiv_public_data.oai_metadata import load_metadata

LOGGER = LOGGER.getChild('splitdata')
LOGGER.setLevel(logging.INFO)
SEED = 1279724012
random.seed(SEED)


def testtrain_split(metadata, proportions=(0.9, 0.05), labels=('train', 'test', 'val'), seed=SEED):
    """ test train val split """
    title_paths = [Path(DIR_OUTPUT) / f'titles_{seed}.{lab}' for lab in labels]
    title_split = [[] for lab in labels]
    abstract_paths = [Path(DIR_OUTPUT) / f'abstract_{seed}.{lab}' for lab in labels]
    abstract_split = [[] for lab in labels]
    cumprop = [ sum(proportions[:i+1]) for i in range(len(proportions)) ]

    for md in metadata:
        rnd = random.random()
        if rnd < cumprop[0]:
            title_split[0].append(md['title'].replace('\n', ''))
            abstract_split[0].append(md['abstract'].replace('\n', ''))
        elif rnd < cumprop[1]:
            title_split[1].append(md['title'].replace('\n', ''))
            abstract_split[1].append(md['abstract'].replace('\n', ''))
        else:
            title_split[2].append(md['title'].replace('\n', ''))
            abstract_split[2].append(md['abstract'].replace('\n', ''))

    for lab, split in zip(labels, title_split):
        LOGGER.info(f'{lab} split contains {len(split)} examples out of {len(metadata)}')

    for path, split in zip(title_paths, title_split):
        LOGGER.info(f'Saving {path}')
        with path.open('w') as fout:
            fout.write('\n'.join(split))

    for path, split in zip(abstract_paths, abstract_split):
        LOGGER.info(f'Saving {path}')
        with path.open('w') as fout:
            fout.write('\n'.join(split))


def split_fulltext(directory, proportions=(0.9, 0.05), labels=('train', 'test', 'val'), seed=SEED):
    directory = Path(directory)
    files = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for fil in filenames:
            if Path(fil).match("*?.txt"):
                files.append(Path(dirpath) / fil)

    LOGGER.info(f'Found {len(files)} .txt files')

    split_paths = [Path(DIR_OUTPUT) / f'fulltext_{seed}.{lab}' for lab in labels]
    fulltext_split = [[] for lab in labels]

    cumprop = [ sum(proportions[:i+1]) for i in range(len(proportions)) ]
    for fil in files:
        rnd = random.random()
        if rnd < cumprop[0]:
            fulltext_split[0].append(fil)
        elif rnd < cumprop[1]:
            fulltext_split[1].append(fil)
        else:
            fulltext_split[2].append(fil)

    for lab, split in zip(labels, fulltext_split):
        LOGGER.info(f'{lab} split contains {len(split)} examples out of {len(files)}')

    for path, split in zip(split_paths, fulltext_split):
        LOGGER.info(f'Saving {path}')
        with path.open('w') as fout:
            for fil in split:
                with fil.open('r') as fin:
                    fout.write(fin.read())

if __name__ == '__main__':
    #pass
    #testtrain_split(load_metadata())
    split_fulltext('/smartml-athena/arxiv/fulltext/')
