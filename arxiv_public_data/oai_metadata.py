"""
oia_metadata.py

authors: Matt Bierbaum and Colin Clement
date: 2019-02-25

This module interacts with the Open Archive Initiative API, downloading
the metadata for all Arxiv articles.

Usage
=====

python oia_metadata.py data/<savefile>.json

Notes
=====
The save file is not technically JSON, but individual streamed lines of JSON,
each of which is compressed by gzip. Use the helper function load_metadata
to be sure to open it without error.

Resources
=========
* http://www.openarchives.org/OAI/2.0/openarchivesprotocol.htm
* https://arxiv.org/help/oa/index
"""

import os
import gzip
import glob
import json
import time
import hashlib
import datetime
import requests
import xml.etree.ElementTree as ET

from arxiv_public_data.config import LOGGER, DIR_BASE

log = LOGGER.getChild('metadata')

URL_ARXIV_OAI = 'https://export.arxiv.org/oai2'
URL_CITESEER_OAI = 'http://citeseerx.ist.psu.edu/oai2'
OAI_XML_NAMESPACES = {
    'OAI': 'http://www.openarchives.org/OAI/2.0/',
    'arXiv': 'http://arxiv.org/OAI/arXivRaw/'
}

def get_list_record_chunk(resumptionToken=None, harvest_url=URL_ARXIV_OAI,
                          metadataPrefix='arXivRaw'):
    """
    Query OIA API for the metadata of 1000 Arxiv article

    Parameters
    ----------
        resumptionToken : str
            Token for the API which triggers the next 1000 articles

    Returns
    -------
        record_chunks : str
            metadata of 1000 arXiv articles as an XML string
    """
    parameters = {'verb': 'ListRecords'}

    if resumptionToken:
        parameters['resumptionToken'] = resumptionToken
    else:
        parameters['metadataPrefix'] = metadataPrefix

    response = requests.get(harvest_url, params=parameters)

    if response.status_code == 200:
        return response.text

    if response.status_code == 503:
        secs = int(response.headers.get('Retry-After', 20)) * 1.5
        log.info('Requested to wait, waiting {} seconds until retry...'.format(secs))

        time.sleep(secs)
        return get_list_record_chunk(resumptionToken=resumptionToken)
    else:
        raise Exception(
            'Unknown error in HTTP request {}, status code: {}'.format(
                response.url, response.status_code
            )
        )

def _record_element_text(elm, name):
    """ XML helper function for extracting text from leaf (single-node) elements """
    item = elm.find('arXiv:{}'.format(name), OAI_XML_NAMESPACES)
    return item.text if item is not None else None

def _record_element_all(elm, name):
    """ XML helper function for extracting text from queries with multiple nodes """
    return elm.findall('arXiv:{}'.format(name), OAI_XML_NAMESPACES)

def parse_record(elm):
    """
    Parse the XML element of a single ArXiv article into a dictionary of
    attributes

    Parameters
    ----------
        elm : xml.etree.ElementTree.Element
            Element of the record of a single ArXiv article

    Returns
    -------
        output : dict
            Attributes of the ArXiv article stored as a dict with the keys
            id, submitter, authors, title, comments, journal-ref, doi, abstract,
            report-no, categories, and version
    """
    text_keys = [
        'id', 'submitter', 'authors', 'title', 'comments',
        'journal-ref', 'doi', 'abstract', 'report-no'
    ]
    output = {key: _record_element_text(elm, key) for key in text_keys}
    output['categories'] = [
        i.text for i in (_record_element_all(elm, 'categories') or [])
    ]
    output['versions'] = [
        i.attrib['version'] for i in _record_element_all(elm, 'version')
    ]
    return output

def parse_xml_listrecords(root):
    """
    Parse XML of one chunk of the metadata of 1000 ArXiv articles
    into a list of dictionaries

    Parameters
    ----------
        root : xml.etree.ElementTree.Element
            Element containing the records of an entire chunk of ArXiv queries

    Returns
    -------
        records, resumptionToken : list, str
            records is a list of 1000 dictionaries, each containing the
            attributes of a single arxiv article
            resumptionToken is a string which is fed into the subsequent query
    """
    resumptionToken = root.find(
        'OAI:ListRecords/OAI:resumptionToken',
        OAI_XML_NAMESPACES
    )
    resumptionToken = resumptionToken.text if resumptionToken is not None else ''

    records = root.findall(
        'OAI:ListRecords/OAI:record/OAI:metadata/arXiv:arXivRaw',
        OAI_XML_NAMESPACES
    )
    records = [parse_record(p) for p in records]

    return records, resumptionToken

def check_xml_errors(root):
    """ Check for, log, and raise any OAI service errors in the XML """
    error = root.find('OAI:error', OAI_XML_NAMESPACES)

    if error is not None:
        raise RuntimeError(
            'OAI service returned error: {}'.format(error.text)
        )

def find_default_locations():
    outfile = os.path.join(DIR_BASE, 'arxiv-metadata-oai-*.json.gz')
    resume = os.path.join(
        DIR_BASE, 'arxiv-metadata-oai-*.json.gz-resumptionToken.txt'
    )
    fn_outfile = sorted(glob.glob(outfile))
    fn_resume = sorted(glob.glob(resume))

    if len(fn_outfile) > 0:
        return fn_outfile[-1]
    return None

def all_of_arxiv(outfile=None, resumptionToken=None, autoresume=True):
    """
    Download the metadata for every article in the ArXiv via the OAI API

    Parameters
    ----------
        outfile : str (default './arxiv-metadata-oai-<date>.json')
            name of file where data is stored, appending each chunk of 1000
            articles.
        resumptionToken : str (default None)
            token which instructs the OAI server to continue feeding the next
            chunk
        autoresume : bool
            If true, it looks for a saved resumptionToken in the file
            <outfile>-resumptionToken.txt
    """
    date = str(datetime.datetime.now()).split(' ')[0]

    outfile = (
        outfile or # user-supplied
        find_default_locations() or # already in progress 
        os.path.join(
            DIR_BASE, 'arxiv-metadata-oai-{}.json.gz'.format(date)
        ) # new file
    )

    directory = os.path.split(outfile)[0]
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    tokenfile = '{}-resumptionToken.txt'.format(outfile)
    chunk_index = 0
    total_records = 0

    log.info('Saving metadata to "{}"'.format(outfile))

    resumptionToken = None
    if autoresume:
        try:
            resumptionToken = open(tokenfile, 'r').read()
        except Exception as e:
            log.warn("No tokenfile found '{}'".format(tokenfile))
            log.info("Starting download from scratch...")

    while True:
        log.info('Index {:4d} | Records {:7d} | resumptionToken "{}"'.format(
            chunk_index, total_records, resumptionToken)
        )
        xml_root = ET.fromstring(get_list_record_chunk(resumptionToken))
        check_xml_errors(xml_root)
        records, resumptionToken = parse_xml_listrecords(xml_root)

        chunk_index = chunk_index + 1
        total_records = total_records + len(records)

        with gzip.open(outfile, 'at', encoding='utf-8') as fout:
            for rec in records:
                fout.write(json.dumps(rec) + '\n')
        if resumptionToken:
            with open(tokenfile, 'w') as fout:
                fout.write(resumptionToken)
        else:
            log.info('No resumption token, query finished')
            return

        time.sleep(12)  # OAI server usually requires a 10s wait

def load_metadata(infile=None):
    """
    Load metadata saved by all_of_arxiv, as a list of lines of gzip compressed
    json.

    Parameters
    ----------
        infile : str or None
            name of file saved by gzip. If None, one is attempted to be found
            in the expected location with the expected name.

    Returns
    -------
        article_attributes : list
            list of dicts, each of which contains the metadata attributes of
            the ArXiv articles
    """
    fname = infile or find_default_locations()
    with gzip.open(fname, 'rt', encoding='utf-8') as fin:
        return [json.loads(line) for line in fin.readlines()]

def hash_abstracts(metadata):
    """ Replace abstracts with their MD5 hash for legal distribution """
    metadata_no_abstract = []
    for i in range(len(metadata)):
        m = metadata[i].copy()
        m['abstract_md5'] = hashlib.md5(m['abstract'].encode()).hexdigest()
        del m['abstract']
        metadata_no_abstract.append(m)
    return metadata_no_abstract

def validate_abstract_hashes(metadata, metadata_no_abstract):
    """ Validate that abstracts match the hashes """
    for m, n in zip(metadata, metadata_no_abstract):
        md5 = hashlib.md5(m['abstract'].encode()).hexdigest()
        if not md5 == n['abstract_md5']:
            return False
    return True
