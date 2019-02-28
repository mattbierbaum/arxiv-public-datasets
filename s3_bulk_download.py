"""
s3_bulk_download.py

authors: Matt Bierbaum and Colin Clement
date: 2019-02-27

This module uses AWS to request a signed key url, which requests files
from the ArXiv S3 bucket. It then unpacks and converts the pdfs into text.

Note that at the time of writing the ArXiv manifest, it contains 1.15 TB
of PDFs, which would cost $103 to receive from AWS S3.

see: https://arxiv.org/help/bulk_data_s3
"""

import os
import gzip
import glob
import shutil
import shlex
import boto3
import hashlib
import botocore
import requests
import subprocess 

import xml.etree.ElementTree as ET

OUTDIR = os.path.abspath('./data')
CHUNK_SIZE = 2**20  # 1MB

BUCKET_NAME = 'arxiv'
S3_PDF_MANIFEST = 'pdf/arXiv_pdf_manifest.xml'
S3_TEX_MANIFEST = 'src/arXiv_src_manifest.xml'
HEADERS = {'x-amz-request-payer': 'requester'}

s3 = boto3.client('s3', region_name='us-east-1')

def download_file(filename, outfile, chunk_size=CHUNK_SIZE, redownload=False):
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
    print('Requesting "{}"'.format(filename))
    request = requests.get(url, stream=True)
    response_iter = request.iter_content(chunk_size=chunk_size)
    with gzip.open(outfile, 'wb') as fout:
        for i, chunk in enumerate(response_iter):
            fout.write(chunk)
            md5.update(chunk)

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

def _download_tar(filename, md5_expected):
    """ Download filename, check its md5sum, and form the output path """
    outname = os.path.join(OUTDIR, os.path.basename(filename)) + '.gz'
    md5_downloaded = download_file(filename, outname)
    
    if md5_expected != md5_downloaded:
        msg = "MD5 '{}' does not match expected '{}' for file '{}'".format(
            md5_downloaded, md5_expected, filename
        )
        raise AssertionError(msg)

    return outname

def _call(cmd):
    """ Spawn a subprocess and execute the string in cmd """
    return subprocess.check_call(shlex.split(cmd))

def download_tar_pdf2text(fileinfo):
    """
    Download and process one of the tar files from the ArXiv manifest.
    Download, unpack, and spawn the Docker image for converting pdf2text

    The tar file will be stored in OUTDIR/<fileinfo[filename]>(.tar) and the
    resulting arXiv articles will be stored in the subdirectory
    OUTDIR/<fileinfo[filename]>/*.txt

    Parameters
    ----------
        fileinfo : dict
            dictionary of file information from parse_manifest
    """
    filename = fileinfo['filename']
    md5sum = fileinfo['md5sum']

    print('Processing tar "{}" ...'.format(filename))
    outname = _download_tar(filename, md5sum)

    tardir = os.path.join(OUTDIR, outname.split('.')[0])
    if not os.path.exists(tardir):
        _call('mkdir -p {}'.format(tardir))

    # unpack tar file
    _call('tar -C {} -xf {}'.format(tardir, outname))

    # ArXiv tar file has one directory, move contents up to tardir
    for fn in glob.glob('{}/*/*.pdf'.format(tardir)):
        shutil.move(fn, tardir)

    # Run docker image to convert pdfs in tardir into *.txt
    _call('docker run --rm -v {}:/pdfs fulltext'.format(tardir))
    # clean up expanded pdf files
    _call('rm {}/*.pdf'.format(tardir))

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
    manifest_file = os.path.join(OUTDIR, 'arxiv-manifest.xml.gz')
    md5 = download_file(S3_PDF_MANIFEST, manifest_file, redownload=redownload)
    manifest = gzip.open(manifest_file, 'rb').read()
    return parse_manifest(manifest)

def download_and_process_manifest_files(list_of_fileinfo):
    """
    Download PDFs from the ArXiv AWS S3 bucket and convert each pdf to text
    Parameters
    ----------
    list_of_fileinfo : list
        Some elements of results of get_manifest
    """
    for fileinfo in list_of_fileinfo:
        download_tar_pdf2text(fileinfo)
