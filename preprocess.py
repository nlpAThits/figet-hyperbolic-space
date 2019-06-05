#!/usr/bin/env python
# encoding: utf-8

import argparse
from tqdm import tqdm
import torch
from os import path

import figet
from figet.Constants import *
from figet.utils import process_line

DS = "distant_supervision"
CR = "crowd"
CR_TRAIN = f"{CR}/train_m.json"
CR_DEV = f"{CR}/dev.json"
CR_TEST = f"{CR}/test.json"
EL_TRAIN = f"{DS}/el_train.json"
EL_DEV = f"{DS}/el_dev.json"
HW_TRAIN = f"{DS}/headword_train.json"
HW_DEV = f"{DS}/headword_dev.json"

log = figet.utils.get_logging()


def make_vocabs(args):
    """
    It creates a Dict for the words on the whole dataset, and the types
    """
    token_vocab = figet.TokenDict(lower=False)
    type_vocab = figet.TypeDict()

    all_files = [path.join(args.dataset, fpath) for fpath in [EL_TRAIN, HW_TRAIN, CR_TRAIN, CR_DEV, CR_TEST]]
    bar = tqdm(desc="make_vocabs", total=figet.utils.wc(all_files))
    for data_file in all_files:
        for line in open(data_file, buffering=BUFFER_SIZE):
            bar.update()

            fields, tokens = process_line(line)

            for token in tokens:
                token_vocab.add(token)

            for mention_type in fields[TYPE]:
                type_vocab.add(mention_type)

    bar.close()

    char_vocab = figet.Dict()
    char_vocab.add(UNK_WORD)
    for char in CHARS:
        char_vocab.add(char)

    log.info(f"Created vocabs:\n\t#token: {token_vocab.size()}\n\t#type: {type_vocab.size()}\n"
             f"\t#chars: {char_vocab.size()}")

    return {TOKEN_VOCAB: token_vocab, TYPE_VOCAB: type_vocab, CHAR_VOCAB: char_vocab}


def make_word2vec(filepath, tokenDict):
    word2vec = figet.Word2Vec()
    log.info("Start loading pretrained word vecs")
    for line in tqdm(open(filepath), total=figet.utils.wc(filepath)):
        fields = line.strip().split()
        token = fields[0]
        try:
            vec = list(map(float, fields[1:]))
        except ValueError:
            continue
        word2vec.add(token, torch.Tensor(vec))

    ret = []
    oov = 0

    # PAD word (index 0) is a vector full of zeros
    ret.append(torch.zeros(word2vec.get_unk_vector().size()))
    tokenDict.label2wordvec_idx[figet.Constants.PAD_WORD] = 0

    for idx in range(1, tokenDict.size()):
        token = tokenDict.idx2label[idx]

        if token in word2vec:
            vec = word2vec.get_vec(token)
            tokenDict.label2wordvec_idx[token] = len(ret)
            ret.append(vec)
        else:
            oov += 1

    ret = torch.stack(ret)                  # creates a "matrix" of token.size() x embed_dim
    log.info("* OOV count: %d" %oov)
    log.info("* Embedding size (%s)" % (", ".join(map(str, list(ret.size())))))
    return ret


def make_type2vec(filepath, typeDict):
    log.info("Start loading pretrained type vecs")
    type_model = torch.load(filepath, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    types = type_model["objects"]
    vecs = type_model["model"]["lt.weight"]

    type2vec = {types[i]: vecs[i] for i in range(len(types))}

    ret = []
    target_vec = vecs[0]

    for idx in range(typeDict.size()):
        label = typeDict.idx2label[idx]
        if label in type2vec:               # It adds the right vector in case that it has it, or the previous vector
            target_vec = type2vec[label]    # It is a way to assign some "pseudo" random vector for the few types that
        ret.append(target_vec)              # we don't have a poincare embedding

    ret = torch.stack(ret)                  # creates a "matrix" of typeDict.size() x type_embed_dim
    log.info("* Embedding size (%s)" % (", ".join(map(str, list(ret.size())))))
    return ret


def make_data(data_files, vocabs, type_quantity, args):
    data = []
    for fname in data_files:
        file_path = path.join(args.dataset, fname)
        for line in tqdm(open(file_path, buffering=BUFFER_SIZE), total=figet.utils.wc(file_path)):
            fields, _ = process_line(line)

            mention = figet.Mention(fields)
            data.append(mention)

    log.info("Prepared {} mentions.".format(len(data)))
    dataset = figet.Dataset(data, args, type_quantity)

    log.info(f"Transforming to matrix {len(data)} mentions from {data_files}")
    dataset.to_matrix(vocabs, args)

    return dataset


def main(args):

    log.info("Preparing vocabulary...")
    vocabs = make_vocabs(args)

    log.info("Preparing pretrained word vectors...")
    word2vec = make_word2vec(args.word2vec, vocabs[TOKEN_VOCAB])

    log.info("Preparing pretrained type vectors...")
    type2vec = make_type2vec(args.type2vec, vocabs[TYPE_VOCAB])

    log.info("Preparing training...")
    train = make_data([CR_TRAIN, EL_TRAIN, HW_TRAIN], vocabs, len(type2vec), args)
    log.info("Preparing dev...")
    dev = make_data([CR_DEV], vocabs, len(type2vec), args)
    log.info("Preparing test...")
    test = make_data([CR_TEST], vocabs, len(type2vec), args)

    log.info("Saving pretrained word vectors to '%s'..." % (args.save_data + "/word2vec.pt"))
    torch.save(word2vec, args.save_data + "/word2vec.pt")

    log.info("Saving pretrained type vectors to '%s'..." % (args.save_data + "/type2vec.pt"))
    torch.save(type2vec, args.save_data + "/type2vec.pt")

    log.info("Saving data to '%s'..." % (args.save_data + "/data.pt"))
    save_data = {"vocabs": vocabs, "train": train, "dev": dev, "test": test}
    torch.save(save_data, args.save_data + "/data.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess.py")

    # Input data
    parser.add_argument("--dataset", required=True, help="Path to the dataset")
    parser.add_argument("--word2vec", required=True, type=str, help="Path to pretrained word vectors.")
    parser.add_argument("--type2vec", required=True, type=str, help="Path to pretrained type vectors.")
    parser.add_argument("--emb_size", default=300, type=int, help="Embedding size.")

    # Mention
    parser.add_argument("--mention_length", default=5, type=int,
                        help="Max amount of words taken for mention representation")
    parser.add_argument("--mention_char_length", default=25, type=int,
                        help="Max amount of chars taken for mention representation")

    # Context
    parser.add_argument("--full_context_length", default=25, type=int,
                        help="Max amount of words of the left + mention + right context.")
    parser.add_argument("--side_context_length", default=10, type=int,
                        help="Max length of the context on each side (left or right)")

    # Ops
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data.")

    # Output data
    parser.add_argument("--save_data", required=True, help="Path to the output data.")

    args = parser.parse_args()

    figet.utils.set_seed(42)

    main(args)
