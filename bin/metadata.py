from arxiv_public_data.oai_metadata import all_of_arxiv

if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else None
    all_of_arxiv(out)
