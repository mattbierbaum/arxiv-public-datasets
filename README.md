# arXiv public datasets

This project is part of a submission to an ICLR 2019 workshop, RLGM
Representation Learning on Graphs and Manifolds. The manuscript can be found on
[arXiv:1905.00075](https://arxiv.org/abs/1905.00075). Our primary purpose is to
develop a set of tools to standardize and facilitate use of the arXiv as a
dataset. Due to licensing and distribution issues, our work is primarily a set
of scripts which builds the dataset from various public data sources. There are
additional cleaning, organization, and  aggregation functions that it performs
as well.

This project is under development as we try to best fit the needs of the
community. We have adopted [semver](https://semver.org/) and as such will denote
major releases with the first numeral in the tagged version.

Currently, the project grabs or generates:

* **Article metadata** -- title, authors string, category, doi, abstract, submitter
* **PDFs** -- all PDFs available through arXiv bulk download
* **Plain text** -- PDFs converted to UTF-8 encoded plain text
* **Citation graph** -- intra-arXiv citation graph between arXiv IDs only (generated from plain text)
* **Author string parsing** -- convert metadata author strings into standardized list of name, affiliations

We are able to host certain generated portions of this dataset as released
snapshots.  The iterations can be found under the releases tab:
[Releases](https://github.com/mattbierbaum/arxiv-public-datasets/releases).
However, the rest of it must be generated locally.

The requirements to generate the datasets from this repository vary from
dataset to dataset, but the requirements for all is:

## Setup on Linux

Install the required system packages (or use an alternative Python distribution
of your choice). For Debian / Ubuntu / similar:

    sudo apt install python3 python3-pip python3-virtualenv poppler-utils

Download the code and prepare the python environment:

    git clone https://github.com/mattbierbaum/arxiv-public-datasets
    cd arxiv-public-datasets

    virtualenv venv
    . venv/bin/activate

    pip3 install -e .
    pip3 install -r requirements.txt

Decide where the data should live and modify the config.json file. This
directory needs to have adequate space to hold ~ 1TB of pdfs and ~ 70GB of text
if you so choose to retrieve them:

    cp config.json.example config.json
    [set ARXIV_DATA in config.json to your own directory]

The scripts in `bin` will then create any of the three subdirectories:

    $ARXIV_DATA/tarpdfs   # raw pdf files from Amazon AWS bucket
    $ARXIV_DATA/fulltext  # .txt from raw .pdf
    $ARXIV_DATA/output    # co-citation network, parsed author strings, etc

## Article metadata

**Run OAI metadata harvester**

    python bin/metadata.py [OPTIONAL filepath.json.gz]

This will download the entire ArXiv metadata set, saving it as a series of
gzip-compressed JSON entries. The default save location is
`$ARXIV_DATA/arxiv-metadata-oai-<date>.json.gz`. This process will take at
least 6 hours, as the OAI server only sends 1000 entries every 15 seconds. A
resumption token is saved, so the process can be restarted by running again.

## PDFs

**Prepare credentials**

In addition to the setup above, you need to prepare your AWS credentials for
use with boto3, the Python AWS library. A long explanation is available
[here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html)
while the quick method is to:

    apt install awscli
    aws configure


**Bulk download of ArXiv PDFs**

This download costs about 100 USD (and is 1.1TB) at the time of writing, as the 
[ArXiv bulk download](https://arxiv.org/help/bulk_data) only allows
requester-pays AWS S3 downloads. Ensure that you have at least 1TB of free space
in the directory specified in `config.json`: 

    python bin/pdfdownload.py [OPTIONAL manifest_file.json.gz]

## Plain text

**Bulk PDF conversion**

To use our tool for text conversion of all the PDFs from the ArXiv bulk download
described above, execute the following. NOTE: if you have not already downloaded
the PDFs, this tool will do so. If you have downloaded them, be sure to not change
the `$ARXIV_DATA` so that it will not re-download the tars.

    python bin/fulltext.py [OPTIONAL number_of_processes, default cpu_count]

At the time of writing, converting 1.39 million articles requires over 400 core-hours 
using two Intel Xeon E5-2600 CPUs.

## Cocitation network

To generate the cocitation network, you first must have the full text. Then,
with the directories still set up, run:

    python bin/cocitations.py [OPTIONAL number_of_processes, default cpu_count]

The cocitation network will by default be saved in
`$ARXIV_DATA/output/internal-citations.json.gz`.

## Author string split

The OAI metadata from the ArXiv features author strings as submitted by article
authors. In order to use them in a principled way, theses strings must be parsed
and split. To generate and save these author splittings, run:

    python bin/authorsplit.py [OPTIONAL number_of_processes, default cpu_count]

The split author strings will by default be saved in
`$ARXIV_DATA/output/authors-parsed.json.gz`.
