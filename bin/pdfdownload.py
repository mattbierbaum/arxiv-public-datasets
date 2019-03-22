from arxiv_public_data import s3_bulk_download

if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else None
    manifest = s3_bulk_download.get_manifest(out)
    s3_bulk_download.download_check_tarfiles(manifest)
