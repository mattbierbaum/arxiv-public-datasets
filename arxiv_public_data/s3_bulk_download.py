"""
s3_bulk_download.py

authors: Matt Bierbaum and Colin Clement
date: 2019-02-27

This module uses AWS to request a signed key url, which requests files
from the ArXiv S3 bucket. It then unpacks and converts the pdfs into text.

Note that at the time of writing the ArXiv manifest, it contains 1.15 TB
of PDFs, which would cost $103 to receive from AWS S3.

see: https://arxiv.org/help/bulk_data_s3

Usage
-----

Set DIR_FULLTEXT as the directory where the text parsed from pdfs should be placed.
Set DIR_PDFTARS as the directory where the raw pdf tars should be placed.

```
import arxiv_public_data.s3_bulk_download as s3

# Download manifest file (or load if already downloaded)
>>> manifest = s3.get_manifest()

# Download tar files and convert pdf to text
# Costs money! Will only download if it does not find files
>>> s3.process_manifest_files(manifest)

# If you just want to download the PDFs and not convert to text use
>>> s3.download_check_tarfiles(manifest)
```
"""

import os
import re
import gzip
import json
import glob
import shlex
import shutil
import tarfile
import boto3
import hashlib
import requests
import subprocess

from functools import partial
from multiprocessing import Pool
from collections import defaultdict
import xml.etree.ElementTree as ET

from arxiv_public_data import fulltext
from arxiv_public_data.config import DIR_FULLTEXT, DIR_PDFTARS, LOGGER

logger = LOGGER.getChild('s3')

CHUNK_SIZE = 2**20  # 1MB
BUCKET_NAME = 'arxiv'
S3_PDF_MANIFEST = 'pdf/arXiv_pdf_manifest.xml'
S3_TEX_MANIFEST = 'src/arXiv_src_manifest.xml'
HEADERS = {'x-amz-request-payer': 'requester'}

s3 = boto3.client('s3', region_name='us-east-1')

def download_file(filename, outfile, chunk_size=CHUNK_SIZE, redownload=False,
                  dryrun=False):
    """
    Downloads filename from the ArXiv AWS S3 bucket, and returns streaming md5
    sum of the content
    Parameters
    ----------
        filename : str
            KEY corresponding to AWS bucket file
        outfile : stf
            name and path of local file in which downloaded file will be stored
        (optional)
        chunk_size : int
            requests byte streaming size (so 500MB are not stored in memory
            prior to processing)
        redownload : bool
            Look to see if file is already downloaded, and simply return md5sum
            if it it exists, unless redownload is True
        dryrun : bool
            If True, only log activity
    Returns
    -------
        md5sum : str
            md5 checksum of the contents of filename
    """
    if os.path.exists(outfile) and not redownload:
        md5 = hashlib.md5()
        md5.update(gzip.open(outfile, 'rb').read())
        return md5.hexdigest()

    md5 = hashlib.md5()
    url = s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": BUCKET_NAME, "Key": filename, "RequestPayer": 'requester'
        }
    )
    if not dryrun:
        logger.info('Requesting "{}" (costs money!)'.format(filename))
        request = requests.get(url, stream=True)
        response_iter = request.iter_content(chunk_size=chunk_size)
        logger.info("\t Writing {}".format(outfile))
        with gzip.open(outfile, 'wb') as fout:
            for i, chunk in enumerate(response_iter):
                fout.write(chunk)
                md5.update(chunk)
    else:
        logger.info('Requesting "{}" (free!)'.format(filename))
        logger.info("\t Writing {}".format(outfile))
    return md5.hexdigest()

def default_manifest_filename():
    return os.path.join(DIR_PDFTARS, 'arxiv-manifest.xml.gz')

def get_manifest(filename=None, redownload=False):
    """
    Get the file manifest for the ArXiv
    Parameters
    ----------
        redownload : bool
            If true, forces redownload of manifest even if it exists
    Returns
    -------
        file_information : list of dicts
            each dict contains the file metadata
    """
    manifest_file = filename or default_manifest_filename()
    md5 = download_file(
        S3_PDF_MANIFEST, manifest_file, redownload=redownload, dryrun=False
    )
    manifest = gzip.open(manifest_file, 'rb').read()
    return parse_manifest(manifest)

def parse_manifest(manifest):
    """
    Parse the XML of the ArXiv manifest file.

    Parameters
    ----------
        manifest : str
            xml string from the ArXiv manifest file

    Returns
    -------
        file_information : list of dicts
            One dict for each file, containing the filename, size, md5sum,
            and other metadata
    """
    root = ET.fromstring(manifest)
    return [
        {c.tag: f.find(c.tag).text for c in f.getchildren()}
        for f in root.findall('file')
    ]

def _tar_to_filename(filename):
    return os.path.join(DIR_PDFTARS, os.path.basename(filename)) + '.gz'

def download_check_tarfile(filename, md5_expected, dryrun=False, redownload=False):
    """ Download filename, check its md5sum, and form the output path """
    outname = _tar_to_filename(filename)
    md5_downloaded = download_file(
        filename, outname, dryrun=dryrun, redownload=redownload
    )

    if not dryrun:
        if md5_expected != md5_downloaded:
            msg = "MD5 '{}' does not match expected '{}' for file '{}'".format(
                md5_downloaded, md5_expected, filename
            )
            raise AssertionError(msg)

    return outname

def download_check_tarfiles(list_of_fileinfo, dryrun=False):
    """
    Download tar files from the ArXiv manifest and check that their MD5sums
    match

    Parameters
    ----------
        list_of_fileinfo : list
            Some elements of results of get_manifest
        (optional)
        dryrun : bool
            If True, only log activity
    """
    for fileinfo in list_of_fileinfo:
        download_check_tarfile(fileinfo['filename'], fileinfo['md5sum'], dryrun=dryrun)

def call(cmd, dryrun=False, debug=False):
    """ Spawn a subprocess and execute the string in cmd """
    if dryrun:
        logger.info(cmd)
        return 0
    else:
        return subprocess.check_call(
            shlex.split(cmd), stderr=None if debug else open(os.devnull, 'w')
        )

def _make_pathname(filename):
    """
    Make filename path for text document, sorted like on arXiv servers.
    Parameters
    ----------
        filename : str
            string filename of arXiv article
        (optional)
    Returns
    -------
        pathname : str
            pathname in which to store the article following
            * Old ArXiv IDs: e.g. hep-ph0001001.txt returns
                DIR_PDFTARS/hep-ph/0001/hep-ph0001001.txt
            * New ArXiv IDs: e.g. 1501.13851.txt returns
                DIR_PDFTARS/arxiv/1501/1501.13851.txt
    """
    basename = os.path.basename(filename)
    fname = os.path.splitext(basename)[0]
    if '.' in fname:  # new style ArXiv ID
        yearmonth = fname.split('.')[0]
        return os.path.join(DIR_FULLTEXT, 'arxiv', yearmonth, basename)
    # old style ArXiv ID
    cat, aid = re.split(r'(\d+)', fname)[:2]
    yearmonth = aid[:4]
    return os.path.join(DIR_FULLTEXT, cat, yearmonth, basename)

def process_tarfile_inner(filename, pdfnames=None, processes=1, dryrun=False,
                          timelimit=fulltext.TIMELIMIT):
    outname = _tar_to_filename(filename)

    if not os.path.exists(outname):
        msg = 'Tarfile from manifest not found {}, skipping...'.format(outname)
        logger.error(msg)
        return

    # unpack tar file
    if pdfnames:
        namelist = ' '.join(pdfnames)
        cmd = 'tar --one-top-level -C {} -xf {} {}'
        cmd = cmd.format(DIR_PDFTARS, outname, namelist)
    else:
        cmd = 'tar --one-top-level -C {} -xf {}'.format(DIR_PDFTARS, outname)
    _call(cmd, dryrun)

    basename = os.path.splitext(os.path.basename(filename))[0]
    pdfdir = os.path.join(DIR_PDFTARS, basename, basename.split('_')[2])

    # Run fulltext to convert pdfs in tardir into *.txt
    converts = fulltext.convert_directory_parallel(
        pdfdir, processes=processes, timelimit=timelimit
    )

    # move txt into final file structure
    txtfiles = glob.glob('{}/*.txt'.format(pdfdir))
    for tf in txtfiles:
        mvfn = _make_pathname(tf)
        dirname = os.path.dirname(mvfn)
        if not os.path.exists(dirname):
            _call('mkdir -p {}'.format(dirname), dryrun)

        if not dryrun:
            shutil.move(tf, mvfn)

    # clean up pdfs
    _call('rm -rf {}'.format(os.path.join(DIR_PDFTARS, basename)), dryrun)

def process_tarfile(fileinfo, pdfnames=None, dryrun=False, debug=False, processes=1):
    """
    Download and process one of the tar files from the ArXiv manifest.
    Download, unpack, and spawn the Docker image for converting pdf2text.
    It will only try to download the file if it does not already exist.

    The tar file will be stored in DIR_FULLTEXT/<fileinfo[filename](tar)> and the
    resulting arXiv articles will be stored in the subdirectory
    DIR_FULLTEXT/arxiv/<yearmonth>/<aid>.txt for old style arXiv IDs and
    DIR_FULLTEXT/<category>/<yearmonth>/<aid>.txt for new style arXiv IDs.

    Parameters
    ----------
        fileinfo : dict
            dictionary of file information from parse_manifest
        (optional)
        dryrun : bool
            If True, only log activity
        debug : bool
            Silence stderr of Docker _call if debug is False
    """
    filename = fileinfo['filename']
    md5sum = fileinfo['md5sum']

    if check_if_any_processed(fileinfo):
        logger.info('Tar file appears processed, skipping {}...'.format(filename))
        return

    logger.info('Processing tar "{}" ...'.format(filename))
    process_tarfile_inner(filename, pdfnames=None, processes=processes, dryrun=dryrun)

def process_manifest_files(list_of_fileinfo, processes=1, dryrun=False):
    """
    Download PDFs from the ArXiv AWS S3 bucket and convert each pdf to text
    Parameters. If files are already downloaded, it will only process them.
    ----------
        list_of_fileinfo : list
            Some elements of results of get_manifest
        (optional)
        processes : int
            number of paralell workers to spawn (roughly as many CPUs as you have)
        dryrun : bool
            If True, only log activity
    """
    for fileinfo in list_of_fileinfo:
        process_tarfile(fileinfo, dryrun=dryrun, processes=processes)

def check_if_any_processed(fileinfo):
    """
    Spot check a tarfile to see if the pdfs have been converted to text,
    given an element of the s3 manifest
    """
    first = _make_pathname(fileinfo['first_item']+'.txt')
    last = _make_pathname(fileinfo['last_item']+'.txt')
    return os.path.exists(first) and os.path.exists(last)

def generate_tarfile_indices(manifest):
    """
    Go through the manifest and for every tarfile, get a list of the PDFs
    that should be contained within it. This is a separate function because
    even checking the tars is rather slow.

    Returns
    -------
    index : dictionary
        keys: tarfile, values: list of pdfs
    """
    index = {}

    for fileinfo in manifest:
        name = fileinfo['filename']
        logger.info("Indexing {}...".format(name))

        tarname = os.path.join(DIR_PDFTARS, os.path.basename(name))+'.gz'
        files = [i for i in tarfile.open(tarname).getnames() if i.endswith('.pdf')]

        index[name] = files
    return index

def check_missing_txt_files(index):
    """
    Use the index file from `generate_tarfile_indices` to check which pdf->txt
    conversions are outstanding.
    """
    missing = defaultdict(list)
    for tar, pdflist in index.items():
        logger.info("Checking {}...".format(tar))
        for pdf in pdflist:
            txt = _make_pathname(pdf).replace('.pdf', '.txt')

            if not os.path.exists(txt):
                missing[tar].append(pdf)

    return missing

def rerun_missing(missing, processes=1):
    """
    Use the output of `check_missing_txt_files` to attempt to rerun the text
    files which are missing from the conversion. There are various reasons
    that they can fail.
    """
    sort = list(reversed(
        sorted([(k, v) for k, v in missing.items()], key=lambda x: len(x[1]))
    ))

    for tar, names in sort:
        logger.info("Running {} ({} to do)...".format(tar, len(names)))
        process_tarfile_inner(
            tar, pdfnames=names, processes=processes,
            timelimit=5 * fulltext.TIMELIMIT
        )

def process_missing(manifest, processes=1):
    """
    Do the full process of figuring what is missing and running them
    """
    indexfile = os.path.join(DIR_PDFTARS, 'manifest-index.json')

    if not os.path.exists(indexfile):
        index = generate_tarfile_indices(manifest)
        json.dump(index, open(indexfile, 'w'))

    index = json.load(open(indexfile))
    missing = check_missing_txt_files(index)
    rerun_missing(missing, processes=processes)
