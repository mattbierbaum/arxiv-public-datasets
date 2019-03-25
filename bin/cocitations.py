import multiprocessing
from arxiv_public_data import internal_citations

if __name__ == '__main__':
    import sys
    processes = multiprocessing.cpu_count() if len(sys.argv) <= 1 else int(sys.argv[1])

    cites = internal_citations.citation_list_parallel(N=processes)
    internal_citations.save_to_default_location(cites)
