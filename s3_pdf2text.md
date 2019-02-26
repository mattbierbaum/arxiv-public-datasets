Bulk converting PDF
===================

Install Docker
--------------

There are many different guides, I shouldn't lock this readme into a particularly
release / time point so I would follow the official docker documention:
https://docs.docker.com/install/linux/docker-ce/ubuntu/

Grab the code and build
-----------------------

Next, we need the Python package and Dockerfile definition to create the image
which we will be using to run the PDF conversion. It can be had by cloning:

    git clone https://github.com/arXiv/arxiv-fulltext.git
    git checkout develop

Then, we can build the docker image by:

    cd extractor
    sed "s|CMD .*|CMD python3.6 /scripts/launch.py|g" Dockerfile > Dockerfile.all
    docker build -t fulltext -f Dockerfile.all .

Running the image
-----------------

Finally, we can run the image on the command line by creating a host directory
which we will be calling PDFS and execute the docker image pointing to this
directory by mounting it as a volume:

    export PDFS=/some/fullpath/to/pdfs
    docker run --rm -v $PDFS:/pdfs fulltext

Then in PDFS, there should be a corresponding .txt for every .pdf
