"""RegEx patterns for arXiv identifiers in citations."""

CATEGORIES = [
    "acc-phys", "adap-org", "alg-geom", "ao-sci", "astro-ph", "atom-ph",
    "bayes-an", "chao-dyn", "chem-ph", "cmp-lg", "comp-gas", "cond-mat", "cs",
    "dg-ga", "funct-an", "gr-qc", "hep-ex", "hep-lat", "hep-ph", "hep-th",
    "math", "math-ph", "mtrl-th", "nlin", "nucl-ex", "nucl-th", "patt-sol",
    "physics", "plasm-ph", "q-alg", "q-bio", "quant-ph", "solv-int",
    "supr-con", "eess", "econ"
]

__all__ = (
    'REGEX_ARXIV_SIMPLE',
    'REGEX_ARXIV_STRICT',
    'REGEX_ARXIV_FLEXIBLE'
)


# A common typo is to exclude the hyphen in the category.
categories = CATEGORIES + [cat.replace('-', '') for cat in CATEGORIES]

RE_SEPS = r'.{1,5}'
RE_CATEGORIES = r'(?:{})(?:[.][A-Z]{{2}})?'.format(r'|'.join(categories))
RE_DATE = r'[0-9]{2}(?:0[1-9]|1[0-2])'
RE_VERSION = r'(?:[vV][1-9]\d*)?'

# =============================================================================
RE_NUM_NEW = RE_DATE + r'[.]\d{4,5}' + RE_VERSION
RE_NUM_OLD = RE_DATE + r'\d{3}' + RE_VERSION

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
REGEX_ARXIV_SIMPLE = r'({}|{})'.format(RE_ID_OLD, RE_ID_NEW)

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
      r'(?:'
        r'(?:{})?'.format(RE_PREFIX_URL) +
        r'(?:{})?'.format(RE_PREFIX_EPRINT) +
        r'(?:'
          r'(?:{})?'.format(RE_PREFIX_ARXIV) +
          r'({})'.format(RE_ID_OLD) +
        r'|'
          r'(?:{})'.format(RE_PREFIX_ARXIV) +
          r'(?:{}/)?'.format(RE_CATEGORIES) +
          r'({})'.format(RE_ID_NEW) +
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
      r'({})'.format(REGEX_ARXIV_SIMPLE) +
    r')|(?:'
      r'(?:[\[\(]\s*)'
        r'({})'.format(REGEX_ARXIV_SIMPLE) +
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
]

TEST_NEGATIVE = [
    'decays, 1701.01871 LHCB-PAPE',
    'Distribution. In: 1404.2485v3 (2015)',
    '113005 (2013), 1307.4331,',
    'doi: 10.1145/ 321105.321114 ',
    'scalar quantum 1610.07877v1'
]
