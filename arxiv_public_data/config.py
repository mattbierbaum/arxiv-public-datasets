import os
import json

DEFAULT_PATH = os.path.abspath('./')
JSONFILE = './config.json'
KEY = 'ARXIV_OUTDIR'

def get_outdir():
    if os.environ.get(KEY):
        out = os.environ.get(KEY)
    else:
        if os.path.exists(JSONFILE):
            js = json.load(open(JSONFILE))
            out = js.get(KEY)
        else:
            print("WARNING: default output directory is {}".format(DEFAULT_PATH))
            out = DEFAULT_PATH
    return out

try:
    ARXIV_DIR = get_outdir()
except Exception as e:
    print("Error attempting to get path from ENV or json conf, defaulting to current directory")
    ARXIV_DIR = DEFAULT_PATH

OUTDIR = os.path.join(ARXIV_DIR, 'full-text')
TARDIR = os.path.join(ARXIV_DIR, 'full-text', 'rawpdfs')
