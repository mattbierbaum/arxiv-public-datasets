"""
oia_metadata.py

authors: Matt Bierbaum and Colin Clement
date: 2019-02-25

Resources
=========
* http://www.openarchives.org/OAI/2.0/openarchivesprotocol.htm
* https://arxiv.org/help/oa/index
"""

import json
import time
import datetime
import requests
import xml.etree.ElementTree as ET

URL_ARXIV_OAI = 'https://export.arxiv.org/oai2'
OAI_XML_NAMESPACES = {
    'OAI': 'http://www.openarchives.org/OAI/2.0/',
    'arXiv': 'http://arxiv.org/OAI/arXivRaw/'
}

def get_list_record_chunk(resumptionToken=None):
    parameters = {'verb': 'ListRecords'}

    if resumptionToken:
        parameters['resumptionToken'] = resumptionToken
    else:
        parameters['metadataPrefix'] = 'arXivRaw'

    response = requests.get(URL_ARXIV_OAI, params=parameters)

    if response.status_code == 200:
        return response.text
    elif response.status_code == 503:
        secs = int(response.headers.get('Retry-After', 20)) * 1.5
        print('Requested to wait, waiting {} seconds until retry...'.format(secs))

        time.sleep(secs)
        return get_list_record_chunk(resumptionToken=resumptionToken)
    else:
        raise Exception(
            'Unknown error in HTTP request {}, status code: {}'.format(
                response.url, response.status_code
            )
        )

def _record_element_text(elm, name):
    item = elm.find('arXiv:{}'.format(name), OAI_XML_NAMESPACES)
    return item.text if item is not None else None

def _record_element_all(elm, name):
    return elm.findall('arXiv:{}'.format(name), OAI_XML_NAMESPACES)

def parse_record(elm):
    text_keys = [
        'id', 'submitter', 'authors', 'title', 'comments',
        'journal-ref', 'doi', 'abstract', 'report-no'
    ]

    output = {key: _record_element_text(elm, key) for key in text_keys}
    output['categories'] = [
        i.text for i in (_record_element_all(elm, 'categories') or [])]
    output['versions'] = [
        i.attrib['version'] for i in _record_element_all(elm, 'version')
    ]

    return output

def parse_xml_listrecords(text):
    root = ET.fromstring(text)
    resumptionToken = root.find(
        'OAI:ListRecords/OAI:resumptionToken',
        OAI_XML_NAMESPACES
    ).text

    records = root.findall(
        'OAI:ListRecords/OAI:record/OAI:metadata/arXiv:arXivRaw',
        OAI_XML_NAMESPACES
    )
    records = [parse_record(p) for p in records]

    return records, resumptionToken

def check_xml_errors(text):
    root = ET.fromstring(text)
    error = root.find('OAI:error', OAI_XML_NAMESPACES)

    if error is not None:
        raise RuntimeError(
            'OAI service returned error: {}'.format(error.text)
        )

def all_of_arxiv(outfile=None, resumptionToken=None, autoresume=True):
    date = str(datetime.datetime.now()).split(' ')[0]
    outfile = outfile or './arxiv-metadata-oai-{}.json'.format(date)
    tokenfile = '{}-resumptionToken.txt'.format(outfile)
    chunk_index = 0
    total_records = 0

    resumptionToken = None
    if autoresume:
        try:
            resumptionToken = open(tokenfile, 'r').read()
        except Exception as e:
            print("No tokenfile found '{}'".format(tokenfile))

    while True:
        print('Index {:4d} | Records {:7d} | resumptionToken "{}"'.format(
            chunk_index, total_records, resumptionToken)
        )
        xml = get_list_record_chunk(resumptionToken=resumptionToken)
        check_xml_errors(xml)
        records, resumptionToken = parse_xml_listrecords(xml)

        chunk_index = chunk_index + 1
        total_records = total_records + len(records)

        with open(outfile, 'a') as fout:
            for rec in records:
                json.dump(rec, fout)
                fout.write('\n')
        if resumptionToken:
            with open(tokenfile, 'w') as fout:
                fout.write(resumptionToken)
        else:
            print('No resumption token, query finished')
            return

        time.sleep(15)


if __name__ == "__main__":
    import sys
    outfile = sys.argv[1] if len(sys.argv) > 1 else None
    all_of_arxiv(outfile)
