import re

SPACE_DIGIT = r'\s*\d\s*'
SPACE_NUMBER = r'(?:{})+'.format(SPACE_DIGIT)
SPACE_CHAR = r'\s*[a-zA-Z\.-]\s*'
SPACE_WORD = r'(?:{})+'.format(SPACE_CHAR)

# old style ID, 7 digits in a row
RE_NUM_OLD = SPACE_DIGIT*7

# new style ID, 4 digits, ., 4,5 digits
RE_NUM_NEW = (
    SPACE_DIGIT*4 +
    r'\.' +
    SPACE_DIGIT*4 + r'(?:{})?'.format(SPACE_DIGIT)
)

# the version part v1 V2 v 1, etc
RE_VERSION = r'(?:\s*[vV]\s*\d+\s*)?'

# the word arxiv, as printed by the autotex, arXiv
RE_ARXIV = r'\s*a\s*r\s*X\s*i\s*v\s*:\s*'

# any words within square brackets [cs.A I]
RE_CATEGORIES = r'\[{}\]'.format(SPACE_WORD)

# two digit date, month, year "29 Jan 2012"
RE_DATE = SPACE_NUMBER + SPACE_WORD + r'(?:{}){}'.format(SPACE_DIGIT, '{2,4}')

# the full identifier for the banner
RE_ARXIV_ID = (
    RE_ARXIV +
    r'(?:' +
    r'(?:{})|(?:{})'.format(RE_NUM_NEW, RE_NUM_OLD) +
    r')' +
    RE_VERSION +
    RE_CATEGORIES +
    RE_DATE
)

REGEX_ARXIV_ID = re.compile(RE_ARXIV_ID)


def _extract_arxiv_stamp(txt):
    """
    Find location of stamp within the text and remove that section
    """
    match = REGEX_ARXIV_ID.search(txt)

    if not match:
        return txt, ''

    s, e = match.span()
    return '{} {}'.format(txt[:s].strip(), txt[e:].strip()), txt[s:e].strip()


def remove_stamp(txt, split=1000):
    """
    Given full text, remove the stamp placed in the pdf by arxiv itself. This
    deserves a bit of consideration since the stamp often becomes mangled by
    the text extraction tool (i.e. hard to find and replace) and can be
    reversed.

    Parameters
    ----------
    txt : string
        The full text of a document

    Returns
    -------
    out : string
        Full text without stamp
    """
    t0, t1 = txt[:split], txt[split:]
    txt0, stamp0 = _extract_arxiv_stamp(t0)
    txt1, stamp1 = _extract_arxiv_stamp(t0[::-1])

    if stamp0:
        return txt0 + t1
    elif stamp1:
        return txt1[::-1] + t1
    else:
        return txt
