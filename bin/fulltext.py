import os
import multiprocessing
from argparse import ArgumentParser
from arxiv_public_data.fulltext import convert_directory_parallel
from arxiv_public_data.config import DIR_BASE, DIR_FULLTEXT, DIR_OUTPUT, DIR_PDFTARS
from arxiv_public_data import s3_bulk_download
from arxiv_public_data.s3_bulk_download import call

if __name__ == '__main__':

    parser = ArgumentParser(description="""Convert all pdfs contained in a directory and its sub-directories into txt 
    files""")
    parser.add_argument("-N", type=int, default=multiprocessing.cpu_count(), help="OPTIONAL number of CPUs, default "
                                                                                  "all available")
    parser.add_argument("--PLAIN_PDFS", action="store_true",
                        help="OPTIONAL, add this if plain pdfs are available in " + DIR_PDFTARS)
    args = parser.parse_args()

    if not args.PLAIN_PDFS:  # Convert Amazon S3 download files
        manifest = s3_bulk_download.get_manifest()  # Download if not already
        s3_bulk_download.process_manifest_files(manifest, processes=args.N)  # Convert to txt and move to DIR_FULLTEXT
    else:
        convert_directory_parallel(DIR_BASE, processes=args.N)  # Convert directory of plain PDFs file
        #  Subprocesss to move the converted text files inside DIR_FULLTEXT, recursively
        call('rsync -rv --remove-source-files --prune-empty-dirs --include="*.txt" --exclude="*.pdf" '
             '--exclude="{}" --exclude="{}" {} {} '.format(os.path.basename(DIR_FULLTEXT),
                                                           os.path.basename(DIR_OUTPUT),
                                                           DIR_BASE + os.sep, DIR_FULLTEXT), 0)
