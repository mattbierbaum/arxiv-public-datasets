import os
import multiprocessing
from argparse import ArgumentParser
from arxiv_public_data.fulltext import convert_directory_parallel
from arxiv_public_data.config import DIR_BASE, DIR_FULLTEXT, DIR_PDFTARS, DIR_OUTPUT
from arxiv_public_data import s3_bulk_download
from arxiv_public_data.s3_bulk_download import call

if __name__ == '__main__':

    parser = ArgumentParser(description="""Convert all pdfs contained in a directory and its sub-directories into txt 
    files""")
    parser.add_argument("-N", type=int, default=multiprocessing.cpu_count(), help="OPTIONAL number of CPUs, default "
                                                                                  "all available")
    parser.add_argument("--PLAIN_PDFS", type=bool, default=False,
                        help="OPTIONAL if plain pdfs are available in " + DIR_BASE + " (e.g. download from Kaggle), "
                                                                                     "default False")
    args = parser.parse_args()
    if not args.PLAIN_PDFS:
        manifest = s3_bulk_download.get_manifest()
        s3_bulk_download.process_manifest_files(manifest, processes=args.N)
    else:
        convert_directory_parallel(DIR_BASE, processes=args.N)
        call('rsync -rv --remove-source-files --include="*.txt" --exclude="*.pdf" '
             '--exclude="{}" --exclude="{}" --exclude="{}" {} {} '.format(os.path.basename(DIR_FULLTEXT),
                                                                          os.path.basename(DIR_PDFTARS),
                                                                          os.path.basename(DIR_OUTPUT),
                                                                          DIR_BASE + os.sep, DIR_FULLTEXT), 0)
