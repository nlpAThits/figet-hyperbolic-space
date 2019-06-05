#!/bin/bash

set -o errexit

# Could be an absolute path to anywhere else
DATA=data

# Dataset
dataset_dir=${DATA}/release

# Embeddings
embeddings=${DATA}/word-embeds/glove.840B.300d.txt
# embeddings=${DATA}/word-embeds/miniglove.txt
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
        --gpus=0 \
        --export_path=freq-sym
# TODO
elif [ "${do_what}" == "inference" ];
then
    ckpt=${ckpt}/${current_run}
    python -u ./infer.py \
        --data=${dataset_dir}/foo_dev.jsonl \
        --save_model=${ckpt}/${corpus_name}.model.pt \
        --save_idx2threshold=${ckpt}/${corpus_name}.thres \
        --pred=${ckpt}/${corpus_name}.pred.txt \
        --single_context=0 \
        --context_num_layers=2 --bias=0 --context_length=10
fi

