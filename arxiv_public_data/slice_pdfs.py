import os
import subprocess
import shlex
from collections import defaultdict

from arxiv_public_data.config import DIR_FULLTEXT, DIR_PDFTARS, LOGGER

def id_to_tarpdf(n):
    if '.' in n:
        ym = n.split('.')[0]
        return '{}/{}.pdf'.format(ym, n)
    else:
        ym = n.split('/')[1][:4]
        return '{}/{}.pdf'.format(ym, n.replace('/', ''))

def _call(cmd, dryrun=False, debug=False):
    """ Spawn a subprocess and execute the string in cmd """
    return subprocess.check_call(
        shlex.split(cmd), stderr=None if debug else open(os.devnull, 'w')
    )

def _tar_to_filename(filename):
    return os.path.join(DIR_PDFTARS, os.path.basename(filename)) + '.gz'

def extract_files(tarfile, pdfs, outdir):
    """
    Extract the list of `pdfs` filenames from `tarfile` into the `outdir`
    """
    filename = tarfile
    namelist = ' '.join([id_to_tarpdf(i) for i in pdfs])

    outname = _tar_to_filename(filename)
    basename = os.path.splitext(os.path.basename(filename))[0]
    tdir = os.path.join(DIR_PDFTARS, basename)
    outpdfs = ' '.join([os.path.join(tdir, id_to_tarpdf(i)) for i in pdfs])

    cmd0 = 'tar --one-top-level -C {} -xf {} {}'.format(DIR_PDFTARS, outname, namelist)
    cmd1 = 'cp -a {} {}'.format(outpdfs, outdir)
    cmd2 = 'rm -rf {}'.format(tdir)

    _call(cmd0)
    _call(cmd1)
    _call(cmd2)

def call_list(ai, manifest):
    """
    Convert a list of articles and the tar manifest into a dictionary
    of the tarfiles and the pdfs needed from them.
    """
    inv = {}
    for tar, pdfs in manifest.items():
        for pdf in pdfs:
            inv[pdf] = tar

    tars = defaultdict(list)
    num = 0
    for i in ai:
        aid = i.get('id')
    
        tar = id_to_tarpdf(aid)
        if not tar in inv:
            continue
        tars[inv[id_to_tarpdf(aid)]].append(aid)

    return tars

def extract_by_filter(oai, tarmanifest, func, outdir):
    """
    User-facing function that deals extracts a section of articles from
    the entire arxiv.

    Parameters
    ----------
    oai : list of dicts
        The OAI metadata from `oai_metadata.load_metadata`

    tarmanifest : list of dicts
        Dictionary describing the S3 downloads, `s3_bulk_download.get_manifest`

    func : function
        Filter to apply to OAI metadata to get list of articles

    outdir : string
        Directory in which to place the PDFs and metadata for the slice
    """
    articles = func(oai)
    tarmap = call_list(articles, tarmanifest)

    for tar, pdfs in tarmap.items():
        extract_files(tar, pdfs, outdir=outdir)

    with open(os.path.join(outdir, 'metadata.json'), 'w') as f:
        json.dump(articles, f)
