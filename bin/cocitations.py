import multiprocessing
from arxiv_public_data import internal_citations
from arxiv_public_data.config import DIR_FULLTEXT

if __name__ == '__main__':
    import sys

    opts = [opt for opt in sys.argv[1:] if opt.startswith("-")]
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    opts_args = dict(zip(opts, args))
    if '-dir' in opts_args:  # option for directory of full text files
        txt_dir = opts_args['-dir']
    else:
        txt_dir = DIR_FULLTEXT  # default directory

    if '-N' in opts_args:  # option for number of cpus
        processes = opts_args['-N']
    else:
        processes = multiprocessing.cpu_count()  # default: all cpu available

    cites = internal_citations.citation_list_parallel(N=processes, directory=txt_dir)
    internal_citations.save_to_default_location(cites)
