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

## PDFs

Requirements:

* AWS account with access keys

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
