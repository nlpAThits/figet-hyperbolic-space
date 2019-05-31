#!/bin/bash

set -o errexit

# Could be an absolute path to anywhere else
DATA=data

# Dataset
dataset_dir=${DATA}/ultrafined

# Embeddings
# embeddings=${embeddings_dir}/glove.840B.300d.txt
embeddings=${DATA}/word-embeds/miniglove.txt
type_embeddings=${DATA}/type-embeds/uft.wn.minfreq100.dim10.bs50.1499.pt

# Checkpoints and prep
prep=${DATA}/prep

do_what=$1
prep_run=$2
this_prep=${prep}/${prep_run}

# to export statistics and the model's weights
mkdir -p tensorboard
mkdir -p models

if [ "${do_what}" == "preprocess" ];
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
        --epochs=5
        # --gpus=0
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

