# https://github.com/arXiv/arxiv-base@32e6ad0
"""
Copyright 2017 Cornell University

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""Convert between TeX escapes and UTF8."""
import re
from typing import Pattern, Dict, Match

accents = {
    # first accents with non-letter prefix, e.g. \'A
    "'A": 0x00c1, "'C": 0x0106, "'E": 0x00c9, "'I": 0x00cd,
    "'L": 0x0139, "'N": 0x0143, "'O": 0x00d3, "'R": 0x0154,
    "'S": 0x015a, "'U": 0x00da, "'Y": 0x00dd, "'Z": 0x0179,
    "'a": 0x00e1, "'c": 0x0107, "'e": 0x00e9, "'i": 0x00ed,
    "'l": 0x013a, "'n": 0x0144, "'o": 0x00f3, "'r": 0x0155,
    "'s": 0x015b, "'u": 0x00fa, "'y": 0x00fd, "'z": 0x017a,
    '"A': 0x00c4, '"E': 0x00cb, '"I': 0x00cf, '"O': 0x00d6,
    '"U': 0x00dc, '"Y': 0x0178, '"a': 0x00e4, '"e': 0x00eb,
    '"i': 0x00ef, '"o': 0x00f6, '"u': 0x00fc, '"y': 0x00ff,
    '.A': 0x0226, '.C': 0x010a, '.E': 0x0116, '.G': 0x0120,
    '.I': 0x0130, '.O': 0x022e, '.Z': 0x017b, '.a': 0x0227,
    '.c': 0x010b, '.e': 0x0117, '.g': 0x0121, '.o': 0x022f,
    '.z': 0x017c, '=A': 0x0100, '=E': 0x0112, '=I': 0x012a,
    '=O': 0x014c, '=U': 0x016a, '=Y': 0x0232, '=a': 0x0101,
    '=e': 0x0113, '=i': 0x012b, '=o': 0x014d, '=u': 0x016b,
    '=y': 0x0233, '^A': 0x00c2, '^C': 0x0108, '^E': 0x00ca,
    '^G': 0x011c, '^H': 0x0124, '^I': 0x00ce, '^J': 0x0134,
    '^O': 0x00d4, '^S': 0x015c, '^U': 0x00db, '^W': 0x0174,
    '^Y': 0x0176, '^a': 0x00e2, '^c': 0x0109, '^e': 0x00ea,
    '^g': 0x011d, '^h': 0x0125, '^i': 0x00ee, '^j': 0x0135,
    '^o': 0x00f4, '^s': 0x015d, '^u': 0x00fb, '^w': 0x0175,
    '^y': 0x0177, '`A': 0x00c0, '`E': 0x00c8, '`I': 0x00cc,
    '`O': 0x00d2, '`U': 0x00d9, '`a': 0x00e0, '`e': 0x00e8,
    '`i': 0x00ec, '`o': 0x00f2, '`u': 0x00f9, '~A': 0x00c3,
    '~I': 0x0128, '~N': 0x00d1, '~O': 0x00d5, '~U': 0x0168,
    '~a': 0x00e3, '~i': 0x0129, '~n': 0x00f1, '~o': 0x00f5,
    '~u': 0x0169,
    # and now ones with letter prefix \c{c} etc..
    'HO': 0x0150, 'HU': 0x0170, 'Ho': 0x0151, 'Hu': 0x0171,
    'cC': 0x00c7, 'cE': 0x0228,
    'cG': 0x0122, 'cK': 0x0136, 'cL': 0x013b, 'cN': 0x0145,
    'cR': 0x0156, 'cS': 0x015e, 'cT': 0x0162, 'cc': 0x00e7,
    'ce': 0x0229, 'cg': 0x0123, 'ck': 0x0137, 'cl': 0x013c,
    # Commented out due ARXIVDEV-2322 (bug reported by PG)
    # 'ci' : 'i\x{0327}' = chr(0x69).ch(0x327) # i with combining cedilla
    'cn': 0x0146, 'cr': 0x0157, 'cs': 0x015f, 'ct': 0x0163,
    'kA': 0x0104, 'kE': 0x0118, 'kI': 0x012e, 'kO': 0x01ea,
    'kU': 0x0172, 'ka': 0x0105, 'ke': 0x0119, 'ki': 0x012f,
    'ko': 0x01eb, 'ku': 0x0173, 'rA': 0x00c5, 'rU': 0x016e,
    'ra': 0x00e5, 'ru': 0x016f, 'uA': 0x0102, 'uE': 0x0114,
    'uG': 0x011e, 'uI': 0x012c, 'uO': 0x014e, 'uU': 0x016c,
    'ua': 0x0103, 'ue': 0x0115, 'ug': 0x011f,
    'ui': 0x012d, 'uo': 0x014f, 'uu': 0x016d,
    'vA': 0x01cd, 'vC': 0x010c, 'vD': 0x010e,
    'vE': 0x011a, 'vG': 0x01e6, 'vH': 0x021e, 'vI': 0x01cf,
    'vK': 0x01e8, 'vL': 0x013d, 'vN': 0x0147, 'vO': 0x01d1,
    'vR': 0x0158, 'vS': 0x0160, 'vT': 0x0164, 'vU': 0x01d3,
    'vZ': 0x017d, 'va': 0x01ce, 'vc': 0x010d, 'vd': 0x010f,
    've': 0x011b, 'vg': 0x01e7, 'vh': 0x021f, 'vi': 0x01d0,
    'vk': 0x01e9, 'vl': 0x013e, 'vn': 0x0148, 'vo': 0x01d2,
    'vr': 0x0159, 'vs': 0x0161, 'vt': 0x0165, 'vu': 0x01d4,
    'vz': 0x017e
}
r"""
Hash to lookup tex markup and convert to Unicode.

macron: a line above character (overbar \={} in TeX)
caron: v-shape above character (\v{ } in TeX)
See: http://www.unicode.org/charts/

"""

textlet = {
    'AA': 0x00c5, 'AE': 0x00c6, 'DH': 0x00d0, 'DJ': 0x0110,
    'ETH': 0x00d0, 'L': 0x0141, 'NG': 0x014a, 'O': 0x00d8,
    'oe': 0x0153, 'OE': 0x0152, 'TH': 0x00de, 'aa': 0x00e5,
    'ae': 0x00e6,
    'dh': 0x00f0, 'dj': 0x0111, 'eth': 0x00f0, 'i': 0x0131,
    'l': 0x0142, 'ng': 0x014b, 'o': 0x00f8, 'ss': 0x00df,
    'th': 0x00fe,
    # Greek (upper)
    'Gamma': 0x0393, 'Delta': 0x0394, 'Theta': 0x0398,
    'Lambda': 0x039b, 'Xi': 0x039E, 'Pi': 0x03a0,
    'Sigma': 0x03a3, 'Upsilon': 0x03a5, 'Phi': 0x03a6,
    'Psi': 0x03a8, 'Omega': 0x03a9,
    # Greek (lower)
    'alpha': 0x03b1, 'beta': 0x03b2, 'gamma': 0x03b3,
    'delta': 0x03b4, 'epsilon': 0x03b5, 'zeta': 0x03b6,
    'eta': 0x03b7, 'theta': 0x03b8, 'iota': 0x03b9,
    'kappa': 0x03ba, 'lambda': 0x03bb, 'mu': 0x03bc,
    'nu': 0x03bd, 'xi': 0x03be, 'omicron': 0x03bf,
    'pi': 0x03c0, 'rho': 0x03c1, 'varsigma': 0x03c2,
    'sigma': 0x03c3, 'tau': 0x03c4, 'upsion': 0x03c5,
    'varphi': 0x03C6,  # φ
    'phi':  0x03D5,  # ϕ
    'chi': 0x03c7, 'psi': 0x03c8, 'omega': 0x03c9,
}


def _p_to_match(tex_to_chr: Dict[str, int]) -> Pattern:
    # textsym and textlet both use the same sort of regex pattern.
    keys = r'\\(' + '|'.join(tex_to_chr.keys()) + ')'
    pstr = r'({)?' + keys + r'(\b|(?=_))(?(1)}|(\\(?= )| |{}|)?)'
    return re.compile(pstr)


textlet_pattern = _p_to_match(textlet)

textsym = {
    'P': 0x00b6, 'S': 0x00a7, 'copyright': 0x00a9,
    'guillemotleft': 0x00ab, 'guillemotright': 0x00bb,
    'pounds': 0x00a3, 'dag': 0x2020, 'ddag': 0x2021,
    'div': 0x00f7, 'deg': 0x00b0}

textsym_pattern = _p_to_match(textsym)


def _textlet_sub(match: Match) -> str:
    return chr(textlet[match.group(2)])


def _textsym_sub(match: Match) -> str:
    return chr(textsym[match.group(2)])


def texch2UTF(acc: str) -> str:
    """Convert single character TeX accents to UTF-8.

    Strip non-whitepsace characters from any sequence not recognized (hence
    could return an empty string if there are no word characters in the input
    string).

    chr(num) will automatically create a UTF8 string for big num
    """
    if acc in accents:
        return chr(accents[acc])
    else:
        return re.sub(r'[^\w]+', '', acc, flags=re.IGNORECASE)


def tex2utf(tex: str, letters: bool = True) -> str:
    r"""Convert some TeX accents and greek symbols to UTF-8 characters.

    :param tex: Text to filter.

    :param letters: If False, do not convert greek letters or
    ligatures.  Greek symbols can cause problems. Ex. \phi is not
    suppose to look like φ. φ looks like \varphi.  See ARXIVNG-1612

    :returns: string, possibly with some TeX replaced with UTF8

    """
    # Do dotless i,j -> plain i,j where they are part of an accented i or j
    utf = re.sub(r"/(\\['`\^\"\~\=\.uvH])\{\\([ij])\}", r"\g<1>\{\g<2>\}", tex)

    # Now work on the Tex sequences, first those with letters only match
    if letters:
        utf = textlet_pattern.sub(_textlet_sub, utf)

    utf = textsym_pattern.sub(_textsym_sub, utf)

    utf = re.sub(r'\{\\j\}|\\j\s', 'j', utf)  # not in Unicode?

    # reduce {{x}}, {{{x}}}, ... down to {x}
    while re.search(r'\{\{([^\}]*)\}\}', utf):
        utf = re.sub(r'\{\{([^\}]*)\}\}', r'{\g<1>}', utf)

    # Accents which have a non-letter prefix in TeX, first \'e
    utf = re.sub(r'\\([\'`^"~=.][a-zA-Z])',
                 lambda m: texch2UTF(m.group(1)), utf)

    # then \'{e} form:
    utf = re.sub(r'\\([\'`^"~=.])\{([a-zA-Z])\}',
                 lambda m: texch2UTF(m.group(1) + m.group(2)), utf)

    # Accents which have a letter prefix in TeX
    #  \u{x} u above (breve), \v{x}   v above (caron), \H{x}   double accute...
    utf = re.sub(r'\\([Hckoruv])\{([a-zA-Z])\}',
                 lambda m: texch2UTF(m.group(1) + m.group(2)), utf)

    # Don't do \t{oo} yet,
    utf = re.sub(r'\\t{([^\}])\}', r'\g<1>', utf)

    # bdc34: commented out in original Perl
    # $utf =~ s/\{(.)\}/$1/g; #  remove { } from around {x}

    return utf
