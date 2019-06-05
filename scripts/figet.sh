#!/bin/bash

set -o errexit

# Could be an absolute path to anywhere else
DATA=data

# Dataset
dataset_dir=${DATA}/release

# Embeddings
embeddings=${DATA}/word-embeds/glove.840B.300d.txt
type_embeddings=${DATA}/type-embeds/freq-cooc-sym-10dim.bin

# Checkpoints and prep
prep=${DATA}/prep

do_what=$1
prep_run=$2
this_prep=${prep}/${prep_run}

# to export statistics and the model's weights
mkdir -p tensorboard
mkdir -p models


if [ "${do_what}" == "get_data" ];
then
    printf "\nDownloading corpus...`date`\n"
    if [ -d "${dataset_dir}" ]; then
        echo "Seems that you already have the dataset!"
    else
        wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz -O ${DATA}/ultrafined.tar.gz
        (cd ${DATA} && tar -zxvf ultrafined.tar.gz && rm ultrafined.tar.gz)
    fi

    printf "\nDownloading word embeddings...`date`\n"
    if [ -d "${DATA}/word-embeds" ]; then
        echo "Seems that you already have the embeddings!"
    else
        mkdir -p ${DATA}/word-embeds
        wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O ${DATA}/word-embeds/embeddings.zip
        (cd ${DATA}/word-embeds && unzip embeddings.zip && rm embeddings.zip)
    fi

elif [ "${do_what}" == "preprocess" ];
then
    mkdir -p ${this_prep}
    python -u ./preprocess.py \
        --dataset=${dataset_dir} \
        --word2vec=${embeddings} \
        --type2vec=${type_embeddings} \
        --save_data=${this_prep}

elif [ "${do_what}" == "train" ];
then
    python -u ./train.py \
        --data=${this_prep} \
        --epochs=50 \
        --export_path=${prep_run} \
        --gpus=0 

elif [ "${do_what}" == "inference" ];
then
    python -u ./infer.py \
        --data=${this_prep} \
        --export_path=${prep_run} \
        --gpus=0
fi

