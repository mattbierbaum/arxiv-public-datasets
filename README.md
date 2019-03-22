# arXiv public datasets

This is a repository that generates various public datasets from publically available
data and some mild post-processing and organization. Currently, it grabs or generates:

* **Article metadata** -- title, authors, category, doi, abstract, submitter
* **PDFs** -- all PDFs available through arXiv bulk download
* **Plain text** -- PDFs converted to UTF-8 encoded plain text
* **Citation graph** -- intra-arXiv citation graph between arXiv IDs only (generated from plain text)

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
    [edit config.json]

## Article metadata

**Run OAI metadata harvester**

    python bin/metadata.py [OPTIONAL filepath.json.gz]

This will download the entire ArXiv metadata set, saving it as a series of
gzip-compressed JSON entries. The default save location is
`$ARXIV_DATA/arxiv-metadata-oai-<date>.json.gz`. This process will take at
least 6 hours, as the OAI server only sends 1000 entries every 15 seconds. A
resumption token is saved, so the process can be restarted by running again.

## PDFs

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

### Bulk converting PDF

**Bulk PDF conversion**

To use our tool for text conversion of all the PDFs from the ArXiv bulk download
described above, execute the following. NOTE: if you have not already downloaded
the PDFs, this tool will do so. If you have downloaded them, be sure to not change
the `$ARXIV_DATA` so that it will not re-download the tars.

    python bin/fulltext.py [OPTIONAL number_of_processes, default cpu_count]

This will take many core-hours. At the time of writing, converting 1.39
million articles required over 4000 core-hours.

## Cocitation network

### Generating the network

    python bin/cocitations.py [OPTIONAL number_of_processes, default cpu_count]

