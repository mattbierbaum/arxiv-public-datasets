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

"""Parse Authors lines to extract author and affiliation data."""
import re
import os
import gzip
import json
from itertools import dropwhile
from typing import Dict, Iterator, List, Tuple
from multiprocessing import Pool, cpu_count

from arxiv_public_data.tex2utf import tex2utf
from arxiv_public_data.config import LOGGER, DIR_OUTPUT

logger = LOGGER.getChild('authorsplit')

PREFIX_MATCH = 'van|der|de|la|von|del|della|da|mac|ter|dem|di|vaziri'

"""
Takes data from an Author: line in the current arXiv abstract
file and returns a structured set of data:

 author_list_ptr = [
  [ author1_keyname, author1_firstnames, author1_suffix, affil1, affil2 ] ,
  [ author2_keyname, author2_firstnames, author1_suffix, affil1 ] ,
  [ author3_keyname, author3_firstnames, author1_suffix ]
         ]

Abstracted from Dienst software for OAI1 and other uses. This
routine should just go away when a better metadata structure is
adopted that deals with names and affiliations properly.

Must remember that there is at least one person one the archive
who has only one name, this should clearly be considered the key name.

Code originally written by Christina Scovel, Simeon Warner Dec99/Jan00
 2000-10-16 - separated.
 2000-12-07 - added support for suffix
 2003-02-14 - get surname prefixes from arXiv::Filters::Index [Simeon]
 2007-10-01 - created test script, some tidying [Simeon]
 2018-05-25 - Translated from Perl to Python [Brian C.]
"""


def parse_author_affil(authors: str) -> List[List[str]]:
    """
    Parse author line and returns an list of author and affiliation data.

    The list for each author will have at least three elements for
    keyname, firstname(s) and suffix. The keyname will always have content
    but the other strings might be empty strings if there is no firstname
    or suffix. Any additional elements after the first three are affiliations,
    there may be zero or more.

    Handling of prefix "XX collaboration" etc. is duplicated here and in
    arXiv::HTML::AuthorLink -- it shouldn't be. Likely should just be here.

    This routine is just a wrapper around the two parts that first split
    the authors line into parts, and then back propagate the affiliations.
    The first part is to be used along for display where we do not want
    to back propagate affiliation information.

    :param authors: string of authors from abs file or similar
    :return:
    Returns a structured set of data:
    author_list_ptr = [
       [ author1_keyname, author1_firstnames, author1_suffix, affil1, affil2 ],
       [ author2_keyname, author2_firstnames, author1_suffix, affil1 ] ,
       [ author3_keyname, author3_firstnames, author1_suffix ]
    ]
    """
    return _parse_author_affil_back_propagate(
        **_parse_author_affil_split(authors))


def _parse_author_affil_split(author_line: str) -> Dict:
    """
    Split author line into author and affiliation data.

    Take author line, tidy spacing and punctuation, and then split up into
    individual author an affiliation data. Has special cases to avoid splitting
    an initial collaboration name and records in $back_propagate_affiliation_to
    the fact that affiliations should not be back propagated to collaboration
    names.

    Does not handle multiple collaboration names.
    """
    if not author_line:
        return {'author_list': [], 'back_prop': 0}

    names: List[str] = split_authors(author_line)
    if not names:
        return {'author_list': [], 'back_prop': 0}

    names = _remove_double_commas(names)
    # get rid of commas at back
    namesIter: Iterator[str] = reversed(
        list(dropwhile(lambda x: x == ',', reversed(names))))
    # get rid of commas at front
    names = list(dropwhile(lambda x: x == ',', namesIter))

    # Extract all names (all parts not starting with comma or paren)
    names = list(map(_tidy_name, filter(
        lambda x: re.match('^[^](,]', x), names)))
    names = list(filter(lambda n: not re.match(
        r'^\s*et\.?\s+al\.?\s*', n, flags=re.IGNORECASE), names))

    (names, author_list,
     back_propagate_affiliations_to) = _collaboration_at_start(names)

    (enumaffils) = _enum_collaboration_at_end(author_line)

    # Split name into keyname and firstnames/initials.
    # Deal with different patterns in turn: prefixes, suffixes, plain
    # and single name.
    patterns = [('double-prefix',
                 r'^(.*)\s+(' + PREFIX_MATCH + r')\s(' +
                 PREFIX_MATCH + r')\s(\S+)$'),
                ('name-prefix-name',
                 r'^(.*)\s+(' + PREFIX_MATCH + r')\s(\S+)$'),
                ('name-name-prefix',
                 r'^(.*)\s+(\S+)\s(I|II|III|IV|V|Sr|Jr|Sr\.|Jr\.)$'),
                ('name-name',
                 r'^(.*)\s+(\S+)$'), ]

    # Now go through names in turn and try to get affiliations
    # to go with them
    for name in names:
        pattern_matches = ((mtype, re.match(m, name, flags=re.IGNORECASE))
                           for (mtype, m) in patterns)

        (mtype, match) = next(((mtype, m)
                               for (mtype, m) in pattern_matches
                               if m is not None), ('default', None))
        if match is None:
            author_entry = [name, '', '']
        elif mtype == 'double-prefix':
            s = '{} {} {}'.format(match.group(
                2), match.group(3), match.group(4))
            author_entry = [s, match.group(1), '']
        elif mtype == 'name-prefix-name':
            s = '{} {}'.format(match.group(2), match.group(3))
            author_entry = [s, match.group(1), '']
        elif mtype == 'name-name-prefix':
            author_entry = [match.group(2), match.group(1), match.group(3)]
        elif mtype == 'name-name':
            author_entry = [match.group(2), match.group(1), '']
        else:
            author_entry = [name, '', '']

        # search back in author_line for affiliation
        author_entry = _add_affiliation(
            author_line, enumaffils, author_entry, name)
        author_list.append(author_entry)

    return {'author_list': author_list,
            'back_prop': back_propagate_affiliations_to}


def parse_author_affil_utf(authors: str) -> List:
    """
    Call parse_author_affil() and do TeX to UTF conversion.

    Output structure is the same but should be in UTF and not TeX.
    """
    if not authors:
        return []
    return list(map(lambda author: list(map(tex2utf, author)),
                    parse_author_affil(authors)))


def _remove_double_commas(items: List[str]) -> List[str]:

    parts: List[str] = []
    last = ''
    for pt in items:
        if pt == ',' and last == ',':
            continue
        else:
            parts.append(pt)
            last = pt
    return parts


def _tidy_name(name: str) -> str:
    name = re.sub(r'\s\s+', ' ', name)  # also gets rid of CR
    # add space after dot (except in TeX)
    name = re.sub(r'(?<!\\)\.(\S)', r'. \g<1>', name)
    return name


def _collaboration_at_start(names: List[str]) \
        -> Tuple[List[str], List[List[str]], int]:
    """Perform special handling of collaboration at start."""
    author_list = []

    back_propagate_affiliations_to = 0
    while len(names) > 0:
        m = re.search(r'([a-z0-9\s]+\s+(collaboration|group|team))',
                      names[0], flags=re.IGNORECASE)
        if not m:
            break

        # Add to author list
        author_list.append([m.group(1), '', ''])
        back_propagate_affiliations_to += 1
        # Remove from names
        names.pop(0)
        # Also swallow and following comma or colon
        if names and (names[0] == ',' or names[0] == ':'):
            names.pop(0)

    return names, author_list, back_propagate_affiliations_to


def _enum_collaboration_at_end(author_line: str)->Dict:
    """Get separate set of enumerated affiliations from end of author_line."""
    # Now see if we have a separate set of enumerated affiliations
    # This is indicated by finding '(\s*('
    line_m = re.search(r'\(\s*\((.*)$', author_line)
    if not line_m:
        return {}

    enumaffils = {}
    affils = re.sub(r'\s*\)\s*$', '', line_m.group(1))

    # Now expect to have '1) affil1 (2) affil2 (3) affil3'
    for affil in affils.split('('):
        # Now expect `1) affil1 ', discard if no match
        m = re.match(r'^(\d+)\)\s*(\S.*\S)\s*$', affil)
        if m:
            enumaffils[m.group(1)] = re.sub(r'[\.,\s]*$', '', m.group(2))

    return enumaffils


def _add_affiliation(author_line: str,
                     enumaffils: Dict,
                     author_entry: List[str],
                     name: str) -> List:
    """
    Add author affiliation to author_entry if one is found in author_line.

    This should deal with these cases
    Smith B(labX) Smith B(1) Smith B(1, 2) Smith B(1 & 2) Smith B(1 and 2)
    """
    en = re.escape(name)
    namerex = r'{}\s*\(([^\(\)]+)'.format(en.replace(' ', 's*'))
    m = re.search(namerex, author_line, flags=re.IGNORECASE)
    if not m:
        return author_entry

    # Now see if we have enumerated references (just commas, digits, &, and)
    affils = m.group(1).rstrip().lstrip()
    affils = re.sub(r'(&|and)/,', ',', affils, flags=re.IGNORECASE)

    if re.match(r'^[\d,\s]+$', affils):
        for affil in affils.split(','):
            if affil in enumaffils:
                author_entry.append(enumaffils[affil])
    else:
        author_entry.append(affils)

    return author_entry


def _parse_author_affil_back_propagate(author_list: List[List[str]],
                                       back_prop: int) -> List[List[str]]:
    """Back propagate author affiliation.

    Take the author list structure generated by parse_author_affil_split(..)
    and propagate affiliation information backwards to preceeding author
    entries where none was give. Stop before entry $back_prop to avoid
    adding affiliation information to collaboration names.

    given, eg:
      a.b.first, c.d.second (affil)
    implies
      a.b.first (affil), c.d.second (affil)
    and in more complex cases:
      a.b.first, c.d.second (1), e.f.third, g.h.forth (2,3)
    implies
      a.b.first (1), c.d.second (1), e.f.third (2,3), g.h.forth (2,3)
    """
    last_affil: List[str] = []
    for x in range(len(author_list) - 1, max(back_prop - 1, -1), -1):
        author_entry = author_list[x]
        if len(author_entry) > 3:  # author has affiliation,store
            last_affil = author_entry
        elif last_affil:
            # author doesn't have affil but later one did => copy
            author_entry.extend(last_affil[3:])

    return author_list


def split_authors(authors: str) -> List:
    """
    Split author string into authors entity lists.

    Take an author line as a string and return a reference to a list of the
    different name and affiliation blocks. While this does normalize spacing
    and 'and', it is a key feature that the set of strings returned can be
    concatenated to reproduce the original authors line. This code thus
    provides a very graceful degredation for badly formatted authors lines, as
    the text at least shows up.
    """
    # split authors field into blocks with boundaries of ( and )
    if not authors:
        return []
    aus = re.split(r'(\(|\))', authors)
    aus = list(filter(lambda x: x != '', aus))

    blocks = []
    if len(aus) == 1:
        blocks.append(authors)
    else:
        c = ''
        depth = 0
        for bit in aus:
            if bit == '':
                continue
            if bit == '(':  # track open parentheses
                depth += 1
                if depth == 1:
                    blocks.append(c)
                    c = '('
                else:
                    c = c + bit
            elif bit == ')':  # track close parentheses
                depth -= 1
                c = c + bit
                if depth == 0:
                    blocks.append(c)
                    c = ''
                else:  # haven't closed, so keep accumulating
                    continue
            else:
                c = c + bit
        if c:
            blocks.append(c)

    listx = []

    for block in blocks:
        block = re.sub(r'\s+', ' ', block)
        if re.match(r'^\(', block):  # it is a comment
            listx.append(block)
        else:  # it is a name
            block = re.sub(r',?\s+(and|\&)\s', ',', block)
            names = re.split(r'(,|:)\s*', block)
            for name in names:
                if not name:
                    continue
                name = name.rstrip().lstrip()
                if name:
                    listx.append(name)

    # Recombine suffixes that were separated with a comma
    parts: List[str] = []
    for p in listx:
        if re.match(r'^(Jr\.?|Sr\.?\[IV]{2,})$', p) \
                and len(parts) >= 2 \
                and parts[-1] == ',' \
                and not re.match(r'\)$', parts[-2]):
            separator = parts.pop()
            last = parts.pop()
            recomb = "{}{} {}".format(last, separator, p)
            parts.append(recomb)
        else:
            parts.append(p)

    return parts

def parse_authorline(authors: str) -> str:
    """
    The external facing function from this module. Converts a complex authorline
    into a simple one with only UTF-8.

    Parameters
    ----------
    authors : string
        The raw author line from the metadata

    Returns
    -------
    clean_authors : string
        String represeting cleaned author line

    Examples
    --------
    >>> parse_authorline('A. Losev, S. Shadrin, I. Shneiberg')
    'Losev, A.; Shadrin, S.; Shneiberg, I.'

    >>> parse_authorline("C. Bal\\'azs, E. L. Berger, P. M. Nadolsky, C.-P. Yuan")
    'Balázs, C.; Berger, E. L.; Nadolsky, P. M.; Yuan, C. -P.'

    >>> parse_authorline('Stephen C. Power (Lancaster University), Baruch Solel (Technion)')
    'Power, Stephen C.; Solel, Baruch'

    >>> parse_authorline("L. Scheck (1), H.-Th. Janka (1), T. Foglizzo (2), and K. Kifonidis (1)\n  ((1) MPI for Astrophysics, Garching; (2) Service d'Astrophysique, CEA-Saclay)")
    'Scheck, L.; Janka, H. -Th.; Foglizzo, T.; Kifonidis, K.'
    """
    names = parse_author_affil_utf(authors)
    return '; '.join([', '.join([q for q in n[:2] if q]) for n in names])

def _parse_article_authors(article_author):
    try:
        return [article_author[0], parse_author_affil_utf(article_author[1])]
    except Exception as e:
        msg = "Author split failed for article {}".format(article_author[0])
        logger.error(msg)
        logger.exception(e)
        return [article_author[0], '']

def parse_authorline_parallel(article_authors, n_processes=None):
    """
    Parallelize `parse_authorline`
    Parameters
    ----------
        article_authors : list
            list of tuples (arXiv id, author strings from metadata)
        (optional)
        n_processes : int
            number of processes
    Returns
    -------
        authorsplit : list
            list of author strings in standardized format
            [
             [ author1_keyname, author1_firstnames, author1_suffix, affil1, 
                affil2 ] ,
             [ author2_keyname, author2_firstnames, author1_suffix, affil1 ] ,
             [ author3_keyname, author3_firstnames, author1_suffix ]
            ]
    """
    logger.info(
        'Parsing author lines for {} articles...'.format(len(article_authors))
    )

    pool = Pool(n_processes)
    parsed = pool.map(_parse_article_authors, article_authors)
    outdict = {aid: auth for aid, auth in parsed}

    filename = os.path.join(DIR_OUTPUT, 'authors-parsed.json.gz')
    logger.info('Saving to {}'.format(filename))
    with gzip.open(filename, 'wb') as fout:
        fout.write(json.dumps(outdict).encode('utf-8'))
