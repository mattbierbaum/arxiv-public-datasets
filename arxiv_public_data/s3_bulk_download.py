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

Set OUTDIR as the directory where the text parsed from pdfs should be placed.
Set TARDIR as the directory where the raw pdf tars should be placed.

```
import arxiv_public_data.s3_bulk_download as s3

# Download manifest file (or load if already downloaded)
manifest = s3.get_manifest()

# Download tar files and convert pdf to text
# Costs money! Will only download if it does not find files
s3.download_and_pdf2text(manifest)
```
"""

import os
import re
import gzip
import glob
import shlex
import shutil
import boto3
import hashlib
import botocore
import requests
import subprocess 
from functools import partial
from multiprocessing import Pool

import xml.etree.ElementTree as ET

# Location of text files extracted from PDFs
OUTDIR = os.path.abspath('/pool0/arxiv/full-text')
# Location of TAR files
TARDIR = os.path.abspath('/pool0/arxiv/full-text/rawpdfs')

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
            If True, only print activity
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
            "Bucket": BUCKET_NAME, "Key": filename, "RequestPayer":'requester'
        }
    )
    if not dryrun:
        print('Requesting "{}" (costs money!)'.format(filename))
        request = requests.get(url, stream=True)
        response_iter = request.iter_content(chunk_size=chunk_size)
        print("\t Writing {}".format(outfile))
        with gzip.open(outfile, 'wb') as fout:
            for i, chunk in enumerate(response_iter):
                fout.write(chunk)
                md5.update(chunk)
    else:
        print('Requesting "{}" (free!)'.format(filename))
        print("\t Writing {}".format(outfile))
    return md5.hexdigest()

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

def download_check_manifest_file(filename, md5_expected, savedir=TARDIR,
                                 dryrun=False):
    """ Download filename, check its md5sum, and form the output path """
    outname = os.path.join(savedir, os.path.basename(filename)) + '.gz'
    md5_downloaded = download_file(filename, outname, dryrun=dryrun)
    
    if not dryrun:
        if md5_expected != md5_downloaded:
            msg = "MD5 '{}' does not match expected '{}' for file '{}'".format(
                md5_downloaded, md5_expected, filename
            )
            raise AssertionError(msg)

    return outname

def _call(cmd, dryrun=False, debug=False):
    """ Spawn a subprocess and execute the string in cmd """
    if dryrun:
        print(cmd)
        return 0
    else:
        return subprocess.check_call(
            shlex.split(cmd), stderr=None if debug else open(os.devnull, 'w')
        )

def _make_pathname(filename, savedir=OUTDIR):
    """ 
    Make filename path for text document, sorted like on arXiv servers.
    Parameters
    ----------
        filename : str
            string filename of arXiv article
        (optional)
        savedir : str
            the directory in which to store the file
    Returns
    -------
        pathname : str
            pathname in which to store the article following
            * Old ArXiv IDs: e.g. hep-ph0001001.txt returns
                savedir/hep-ph/0001/hep-ph0001001.txt
            * New ArXiv IDs: e.g. 1501.13851.txt returns
                savedir/arxiv/1501/1501.13851.txt
    """
    basename = os.path.basename(filename)
    fname = os.path.splitext(basename)[0]
    if '.' in fname:  # new style ArXiv ID
        yearmonth = fname.split('.')[0]
        return os.path.join(savedir, 'arxiv', yearmonth, basename)
    # old style ArXiv ID
    cat, aid = re.split(r'(\d+)', fname)[:2]
    yearmonth = aid[:4]
    return os.path.join(savedir, cat, yearmonth, basename)


def _id_to_txtpath(aid, savedir=OUTDIR):
    fname = aid
    if '.' in aid:  # new style ArXiv ID
        yearmonth = aid.split('.')[0]
        return os.path.join(savedir, 'arxiv', yearmonth, fname)+'.txt'
    # old style ArXiv ID
    cat, aid = re.split(r'(\d+)', aid)[:2]
    yearmonth = aid[:4]
    return os.path.join(savedir, cat, yearmonth, fname)+'.txt'


def download_and_pdf2text(fileinfo, savedir=TARDIR, outdir=OUTDIR, dryrun=False,
                          debug=False, processes=1):
    """
    Download and process one of the tar files from the ArXiv manifest.
    Download, unpack, and spawn the Docker image for converting pdf2text.
    It will only try to download the file if it does not already exist.

    The tar file will be stored in OUTDIR/<fileinfo[filename](tar)> and the
    resulting arXiv articles will be stored in the subdirectory
    OUTDIR/arxiv/<yearmonth>/<aid>.txt for old style arXiv IDs and
    OUTDIR/<category>/<yearmonth>/<aid>.txt for new style arXiv IDs.

    Parameters
    ----------
        fileinfo : dict
            dictionary of file information from parse_manifest
        (optional)
        savedir : str
            directory in which to store saved tar files
        outdir : str
            directory in which to store processed text from pdfs
        dryrun : bool
            If True, only print activity
        debug : bool
            Silence stderr of Docker _call if debug is False
    """
    filename = fileinfo['filename']
    md5sum = fileinfo['md5sum']

    file0 = _id_to_txtpath(fileinfo['first_item'], outdir)
    file1 = _id_to_txtpath(fileinfo['last_item'], outdir)

    if os.path.exists(file0) and os.path.exists(file1):
        print('Tar file appears processed, skipping {}...'.format(filename))
        return

    print('Processing tar "{}" ...'.format(filename))
    outname = download_check_manifest_file(filename, md5sum, savedir,
                                           dryrun=dryrun)
    # unpack tar file
    cmd = 'tar --one-top-level -C {} -xf {}'.format(savedir, outname)
    _call(cmd, dryrun)
    basename = os.path.splitext(os.path.basename(filename))[0]
    pdfdir = os.path.join(savedir, basename, basename.split('_')[2])

    # Run docker image to convert pdfs in tardir into *.txt
    cmd = 'docker run --rm -v {}:/pdfs fulltext {}'.format(pdfdir, processes)
    _call(cmd, dryrun, debug)

    # move txt into final file structure
    txtfiles = glob.glob('{}/*.txt'.format(pdfdir))
    for tf in txtfiles:
        mvfn = _make_pathname(tf, outdir)
        dirname = os.path.dirname(mvfn)
        if not os.path.exists(dirname):
            _call('mkdir -p {}'.format(dirname, dryrun))

        if not dryrun:
            shutil.move(tf, mvfn)

    # clean up pdfs
    _call('rm -rf {}'.format(os.path.join(savedir, basename)), dryrun)

def get_manifest(redownload=False):
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
    manifest_file = os.path.join(TARDIR, 'arxiv-manifest.xml.gz')
    md5 = download_file(S3_PDF_MANIFEST, manifest_file, redownload=redownload,
                        dryrun=False)
    manifest = gzip.open(manifest_file, 'rb').read()
    return parse_manifest(manifest)

def download_manifest_files(list_of_fileinfo, savedir=TARDIR, dryrun=False):
    """
    Download tar files from the ArXiv manifest and check that their MD5sums
    match
    Parameters
    ----------
        list_of_fileinfo : list
            Some elements of results of get_manifest
        (optional)
        savedir : str
            Directory in which tar files will be saved
        dryrun : bool
            If True, only print activity
    """
    for fileinfo in list_of_fileinfo:
        download_check_manifest_file(fileinfo['filename'], fileinfo['md5sum'],
                                     savedir, dryrun)

def download_and_process_manifest_files(list_of_fileinfo, processes=1,
                                        savedir=TARDIR, outdir=OUTDIR,
                                        dryrun=False):
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
            If True, only print activity
    """
    for fileinfo in list_of_fileinfo:
        download_and_pdf2text(
            fileinfo, savedir=savedir, outdir=outdir,
            dryrun=dryrun, processes=processes
        )

def check_if_any_processed(fileinfo, savedir=TARDIR, outdir=OUTDIR):
    first = _make_pathname(fileinfo['first_item']+'.txt', outdir)
    last = _make_pathname(fileinfo['last_item']+'.txt', outdir)
    return os.path.exists(first) and os.path.exists(last)
