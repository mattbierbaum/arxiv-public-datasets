import multiprocessing
from arxiv_public_data.fulltext import convert_directory_parallel,TIMELIMIT
from arxiv_public_data.config import DIR_PDFTARS
from argparse import ArgumentParser

if __name__ == '__main__':
    
    parser = ArgumentParser(description="""Convert all pdfs contained in a directory and its sub-directories into txt files""")
    parser.add_argument("--dir", type=str, default = DIR_PDFTARS, help="OPTIONAL directory containing pdfs, default "+ DIR_PDFTARS )
    parser.add_argument("-N", type=int, default=multiprocessing.cpu_count(), help="OPTIONAL number of CPUs, default all available")
    parser.add_argument("--TIMELIMIT", type=int, default=TIMELIMIT, help="OPTIONAL maximum time allowed per pdf in seconds, default "+ str(TIMELIMIT))
    args = parser.parse_args()
   
    convert_directory_parallel(args.dir,processes=args.N,timelimit=args.TIMELIMIT)
