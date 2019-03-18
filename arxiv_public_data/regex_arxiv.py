"""
regex_arxiv.py

author: Matt Bierbaum
date: 2019-03-14

RegEx patterns for finding arXiv id citations in fulltext articles.
"""

import re

# These are all the primary categories present in the OAI ArXiv metadata
CATEGORIES = [
    "acc-phys", "adap-org", "alg-geom", "ao-sci", "astro-ph", "atom-ph",
    "bayes-an", "chao-dyn", "chem-ph", "cmp-lg", "comp-gas", "cond-mat", "cs",
    "dg-ga", "funct-an", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math", "math-ph", "mtrl-th", "nlin", "nucl-ex", "nucl-th", "patt-sol",
    "physics", "plasm-ph", "q-alg", "q-bio", "quant-ph", "solv-int",
    "supr-con", "eess", "econ", "q-fin", "stat"
]

#  All subcategories with more than 2 capital letters (not SG, SI, SP, etc)
SUB_CATEGORIES = [
     'acc-ph', 'ao-ph', 'app-ph', 'atm-clus', 'atom-ph', 'bio-ph', 'chem-ph',
     'class-ph', 'comp-ph', 'data-an', 'dis-nn', 'ed-ph', 'flu-dyn', 'gen-ph',
     'geo-ph', 'hist-ph', 'ins-det', 'med-ph', 'mes-hall', 'mtrl-sci', 'optics',
     'other', 'plasm-ph', 'pop-ph', 'quant-gas', 'soc-ph', 'soft', 'space-ph',
     'stat-mech', 'str-el', 'supr-con'
]

__all__ = (
    'REGEX_ARXIV_SIMPLE',
    'REGEX_ARXIV_STRICT',
    'REGEX_ARXIV_FLEXIBLE'
)

dashdict = {c.replace('-', ''): c for c in CATEGORIES if '-' in c}
dashdict.update({c.replace('-', ''): c for c in SUB_CATEGORIES if '-' in c})

REGEX_VERSION_SPLITTER = re.compile(r'([vV][1-9]\d*)')

def strip_version(name):
    """ 1501.21981v1 -> 1501.21981 """
    return REGEX_VERSION_SPLITTER.split(name)[0]

def format_cat(name):
    """ Strip subcategory, add hyphen to category name if missing """
    if '/' in name:  # OLD ID, names contains subcategory 
        catsubcat, aid = name.split('/')
        cat = catsubcat.split('.')[0] 
        return dashdict.get(cat, cat) + "/" + aid
    else:
        return name

def zeropad_1501(name):
    """ Arxiv IDs after yymm=1501 are padded to 5 zeros """
    if not '/' in name:  # new ID
        yymm, num = name.split('.')
        if int(yymm) > 1500 and len(num) < 5:
            return yymm + ".0" + num
    return name

def clean(name):
    """ Correct common errors in ArXiv IDs to improve matching """
    funcs = [strip_version, format_cat, zeropad_1501]
    for func in funcs:
        name = func(name)
    return name

# A common typo is to exclude the hyphen in the category.
categories = list(set(CATEGORIES + [cat.replace('-', '') for cat in
                                    CATEGORIES]))
subcategories = list(set(SUB_CATEGORIES + [cat.replace('-', '') for cat in
                                           SUB_CATEGORIES]))

#  capture possible minor categories
RE_CATEGORIES = r'(?:{})(?:(?:[.][A-Z]{{2}})|(?:{}))?'.format(
    r'|'.join(categories), r'|'.join(subcategories)
)

# valid YYMM date, NOT preceded by any digits
# NOTE: at the date of writing, it is 2019, so we do not allow
# proper dates for YY 20 or larger
RE_DATE = r'(?:(?:[0-1][0-9])|(?:9[1-9]))(?:0[1-9]|1[0-2])'
RE_VERSION = r'(?:[vV][1-9]\d*)?'

# =============================================================================
RE_NUM_NEW = RE_DATE + r'(?:[.]\d{4,5})' + RE_VERSION
RE_NUM_OLD = RE_DATE + r'(?:\d{3})' + RE_VERSION

# matches: 1612.00001 1203.0023v2
RE_ID_NEW = r'(?:{})'.format(RE_NUM_NEW)

# matches: hep-th/11030234 cs/0112345v2 cs.AI/0112345v2
RE_ID_OLD = r'(?:{}/{})'.format(RE_CATEGORIES, RE_NUM_OLD)

# =============================================================================
# matches: https://arxiv.org/abs/ abs/ arxiv.org/abs/
#   3. e-print: eprints
RE_PREFIX_URL = (
    r'(?:'
      r'(?i:http[s]?\://)?'  # we could have a url prefix
      r'(?i:arxiv\.org/)?'   # maybe with the arxiv.org bit
      r'(?i:abs/|pdf/)'      # at least it has the abs/ part
    r')'
)

# matches: arXiv: arxiv/ arxiv
RE_PREFIX_ARXIV = r'(?i:arxiv\s*[:/\s,.]*\s*)'

# matches:  cs.AI/ cs.AI nucl-th
RE_PREFIX_CATEGORIES = r'(?i:{})'.format(RE_CATEGORIES)

# matches: e-prints: e-print eprints:
RE_PREFIX_EPRINT = r'(?i:e[-]?print[s]?.{1,3})'

# =============================================================================
# matches simple old or new identifiers, no fancy business
REGEX_ARXIV_SIMPLE = r'(?:{}|{})'.format(RE_ID_OLD, RE_ID_NEW)

# this one follows the guide set forth by:
#   https://arxiv.org/help/arxiv_identifier
REGEX_ARXIV_STRICT = (
    r'(?:{})'.format(RE_PREFIX_ARXIV) +
    r'(?:'
      r'({})'.format(RE_ID_OLD) +
    r'|'
      r'({})'.format(RE_ID_NEW) +
    r')'
)

# this regex essentially accepts anything that looks like an arxiv id and has
# the slightest smell of being one as well. that is, if it is an id and
# mentions anything about the arxiv before hand, then it is an id.
REGEX_ARXIV_FLEXIBLE = (
    r'(?:'
      r'({})'.format(REGEX_ARXIV_SIMPLE) +  # capture
    r')|(?:'
      r'(?:'
        r'(?:{})?'.format(RE_PREFIX_URL) +
        r'(?:{})?'.format(RE_PREFIX_EPRINT) +
        r'(?:'
          r'(?:{})?'.format(RE_PREFIX_ARXIV) +
          r'({})'.format(RE_ID_OLD) +  # capture
        r'|'
          r'(?:{})'.format(RE_PREFIX_ARXIV) +
          r'(?:{}/)?'.format(RE_CATEGORIES) +
          r'({})'.format(RE_ID_NEW) +  # capture
        r')'
      r')'
    r'|'
      r'(?:'
        r'(?:{})|'.format(RE_PREFIX_URL) +
        r'(?:{})|'.format(RE_PREFIX_EPRINT) +
        r'(?:{})|'.format(RE_PREFIX_CATEGORIES) +
        r'(?:{})'.format(RE_PREFIX_ARXIV) +
      r')'
      r'.*?'
      r'({})'.format(REGEX_ARXIV_SIMPLE) +  # capture
    r')|(?:'
      r'(?:[\[\(]\s*)'
        r'({})'.format(REGEX_ARXIV_SIMPLE) +  # capture
      r'(?:\s*[\]\)])'
    r')'
)

TEST_POSITIVE = [
    'arXiv:quant-ph 1503.01017v3',
    'math. RT/0903.2992',
    'arXiv, 1511.03262',
    'tions. arXiv preprint arXiv:1607.00021, 2016',
    'Math. Phys. 255, 577 (2005), hep-th/0306165',
    'Kuzovlev, arXiv:cond-mat/9903350 ',
    'arXiv:math.RT/1206.5933,',
    'arXiv e-prints 1306.1595',
    'ays, JHEP 07 (2009) 055, [ 0903.0883]',
    ' Rev. D71 (2005) 063534, [ astro-ph/0501562]',
    'e-print arXiv:1506.02215v1',
    'available at: http://arxiv.org/abs/1511.08977',
    'arXiv e-print: 1306.2144',
    'Preprint arXiv:math/0612139',
    'Vertices in a Digraph. arXiv preprint 1602.02129 ',
    'cond-mat/0309488.'
    'decays, 1701.01871 LHCB-PAPE',
    'Distribution. In: 1404.2485v3 (2015)',
    '113005 (2013), 1307.4331,',
    'scalar quantum 1610.07877v1',
    'cond-mat/0309488.'
    'cond-mat/0309488.8383'
]

TEST_NEGATIVE = [
    'doi: 10.1145/ 321105.321114 ',
    'doi: 10.1145/ 1105.321114 ',
]
