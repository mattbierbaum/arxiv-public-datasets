from arxiv_public_data.oai_metadata import load_metadata
from arxiv_public_data.authors import parse_authorline_parallel

if __name__ == "__main__":
    import sys

    processes = int(sys.argv[1]) if len(sys.argv) > 1 else None
    metadata = load_metadata()
    article_authors = [[md.get('id'), md.get('authors')] for md in metadata]
    parse_authorline_parallel(article_authors, processes)
