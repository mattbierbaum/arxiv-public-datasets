from arxiv_public_data import s3_bulk_download

if __name__ == '__main__':
    import sys
    processes = multiprocessing.cpu_count() if len(sys.argv) <= 1 else int(sys.argv[1])

    manifest = s3_bulk_download.get_manifest()
    s3_bulk_download.download_and_process_manifest_files(manifest, processes=processes)
