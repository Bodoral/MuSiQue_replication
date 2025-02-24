ARG CUDA_VERSION=12.2.2
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG UID

# Create a user (b)
RUN groupadd -g $UID customgroup && \
    useradd -m -u $UID -g customgroup bodor


# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        software-properties-common \
        curl \
        libglib2.0-0 \
        tmux \
        wget \
        vim 

RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.11-venv \
    python3.11-dev \
    python3.11-distutils \
    python3.11-gdbm \
    python3.11-tk \
    python3.11-lib2to3 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Set the working directory
WORKDIR /home/bodor

# NER: Install Spacy
RUN python3.11 -m pip install spacy && spacy download en

# NED1: Install ReFinED 
RUN python3.11 -m pip install https://github.com/amazon-science/ReFinED/archive/refs/tags/V1.zip

# NED2: Install BLINK
RUN mkdir models \
    && cd models \
    && git clone  https://github.com/facebookresearch/BLINK.git \
    && python3.11 -m pip install -e git+https://github.com/facebookresearch/BLINK.git#egg=blink \
    && cd BLINK \
    && chmod +x download_blink_models.sh \
    && ./download_blink_models.sh 

# Install any python packages
RUN python3.11 -m pip install pandas

# For mounting scripts and data
USER bodor 
RUN mkdir src data


