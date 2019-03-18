# arXiv public datasets

This is a repository that generates various public datasets from publically available
data and some mild post-processing and organization. Currently, it grabs or generates:

* **Article metadata** -- title, authors, category, doi, abstract, submitter
* **PDFs** -- all PDFs available through arXiv bulk download
* **LaTeX** -- all LaTeX source files available for bulk download
* **Plain text** -- PDFs converted to UTF-8 encoded plain text
* **Citation graph** -- intra-arXiv citation graph between arXiv IDs only (generated from plain text)

The requirements to generate the datasets from this repository vary from
dataset to dataset so they will be discussed section by section.


## Article metadata

Requirements:

* python3 with pipenv

**Run OAI metadata harvester**

    pipenv install
    pipenv shell
    python arxiv_public_data/oai_metadata.py [OPTIONAL filepath.json.gz]

This will download the entire ArXiv metadata set, saving it as a series of 
gzip-compressed JSON entries. The default save location is
`./data/arxiv-metadata-oai-<date>.json.gz`. This process will take at least 6
hours, as the OAI server only sends 1000 entries every 15 seconds. A resumption
token is saved, so the process can be restarted by running again with the same
filename input.

## PDFs

Requirements:

* Amazon AWS account with access keys

**Bulk download of ArXiv PDFs**

This download costs about $100 (and is 1.1TB) at the time of writing, as the 
[ArXiv bulk download](https://arxiv.org/help/bulk_data) only allows
requester-pays AWS S3 downloads. Set the `savedir` keyword arguments of
`get_manifest` and `download_manifest_files` to set where the files will be
saved, and be sure you have over 1TB of space in that location. The following
will download all of the ArXiv PDFs. If you want to also convert the PDFs to
text, see the section below for a combined download and conversion function.

    pipenv install
    pipenv shell
    python
    >>> import arxiv_public_data.s3_bulk_download as s3
    >>> savedir = <location to save S3 download>
    >>> manifest = s3.get_manifest(savedir=savedir)
    >>> s3.download_manifest_files(manifest, savedir=savedir)

## Plain text

Requirements:

* PDFs
* Docker

### Bulk converting PDF

**Install Docker**

There are many different guides, I shouldn't lock this readme into a particularly
release / time point so I would follow the official docker documention:
https://docs.docker.com/install/linux/docker-ce/ubuntu/

**Grab the code and build**

Next, we need the Python package and Dockerfile definition to create the image
which we will be using to run the PDF conversion. It can be had by cloning:

    git clone https://github.com/mattbierbaum/arxiv-fulltext.git
    cd arxiv-fulltext
    git checkout stamp-removal

Then, we can build the docker image by:

    cd extractor
    docker build -t fulltext -f Dockerfile .

**Running the image** 

This is done internally in our libraries, but for reference, we can run the
image on the command line by creating a host directory which we will be calling
PDFS and execute the docker image pointing to this directory by mounting it as
a volume:

    export PDFS=/some/fullpath/to/pdfs
    docker run --rm -v $PDFS:/pdfs fulltext

Then in PDFS, there should be a corresponding .txt for every .pdf

**Bulk PDF conversion**

To use our tool for text conversion of all the PDFs from the ArXiv bulk download
described above, execute the following. NOTE: if you have not already downloaded
the PDFs, this tool will do so. If you have downloaded them, be sure to specify
the correct `savedir` so that the tool will look for an existing file first. To
be safe, try using the keyword `dryrun=True` to print out the actions of the
tool without downloading.

    pipenv install
    pipenv shell
    python
    >>> import arxiv_public_data.s3_bulk_download as s3
    >>> savedir = <location of saved PDFs>
    >>> outdir = <location of converted text file hierarchy>
    >>> manifest = s3.get_manifest(savedir=savedir)
    >>> s3.download_and_process_manifest_file(manifest, processes=<number>,
        savedir=savedir, outdir=outdir)

This will take many core-hours. At the time of writing, converting 1.39
million articles required over 4000 core-hours.
