import multiprocessing
from arxiv_public_data import internal_citations
from arxiv_public_data.config import DIR_FULLTEXT
from argparse import ArgumentParser

if __name__ == '__main__':
    import sys

    parser = ArgumentParser(description="""Generate cocitation network from fulltext and save as a json file""")
    parser.add_argument("-N", type=int, default=multiprocessing.cpu_count(), help="OPTIONAL number of CPUs, default all available")
    parser.add_argument("--dir", type=str, default=DIR_FULLTEXT, help="OPTIONAL directory containing articles in text format, default "+ DIR_FULLTEXT)

    args = parser.parse_args()

    cites = internal_citations.citation_list_parallel(N=args.N, directory=args.dir)
    internal_citations.save_to_default_location(cites)
