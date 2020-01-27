"""
splitdata.py
"""

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

if __name__ == '__main__':

    testtrain_split(load_metadata())
