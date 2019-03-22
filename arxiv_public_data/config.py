import os
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
)
baselog = logging.getLogger('openarxiv')
logger = baselog.getChild('config')

DEFAULT_PATH = os.path.join(os.path.abspath('./'), 'openarxiv-data')
JSONFILE = './config.json'
KEY = 'OPENARXIV_DATA'

def get_outdir():
    """
    Grab the outdir from:
    1) Environment
    2) config.json
    3) default ($PWD/openarxiv-data)
    """
    if os.environ.get(KEY):
        out = os.environ.get(KEY)
    else:
        if os.path.exists(JSONFILE):
            js = json.load(open(JSONFILE))
            if not KEY in js:
                logger.warn('Configuration in "{}" invalid, using default'.format(JSONFILE))
                logger.warn("default output directory is {}".format(DEFAULT_PATH))
                out = DEFAULT_PATH
            else:
                out = js[KEY]
        else:
            logger.warn("default output directory is {}".format(DEFAULT_PATH))
            out = DEFAULT_PATH
    return out

try:
    DIR_BASE = get_outdir()
except Exception as e:
    logger.error(
        "Error attempting to get path from ENV or json conf, "
        "defaulting to current directory"
    )
    DIR_BASE = DEFAULT_PATH

DIR_FULLTEXT = os.path.join(DIR_BASE, 'fulltext')
DIR_PDFTARS = os.path.join(DIR_BASE, 'tarpdfs')
DIR_OUTPUT = os.path.join(DIR_BASE, 'output')
LOGGER = baselog
