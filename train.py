#!/usr/bin/env python
# encoding: utf-8
from __future__ import division

import argparse
import random
import torch
from torch.optim import Adam

import figet


def config_parser(parser):
    # Data options
    parser.add_argument("--data", required=True, type=str, help="Data path.")
    parser.add_argument("--save_model", default="./save/model.pt", type=str, help="Save the model.")

    # Mention and context encoder parameters
    parser.add_argument("--emb_size", default=300, type=int, help="Embedding size.")
    parser.add_argument("--char_emb_size", default=50, type=int, help="Char embedding size.")
    parser.add_argument("--positional_emb_size", default=25, type=int, help="Positional embedding size.")
    parser.add_argument("--context_rnn_size", default=200, type=int, help="RNN size of ContextEncoder.")
    parser.add_argument("--attn_size", default=100, type=int, help="Attention vector size.")
    parser.add_argument("--mention_dropout", default=0.5, type=float, help="Dropout rate for mention")
    parser.add_argument("--context_dropout", default=0.2, type=float, help="Dropout rate for context")

    # projection parameters
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--hidden_layers', type=int, default=1)
    parser.add_argument("--bias", default=1, type=int, help="Whether to use bias in the linear transformation.")
    parser.add_argument("--projection_dropout", default=0.3, type=float, help="Dropout rate for projection")
    parser.add_argument("--metric_dist_factor", default=1.0, type=float, help="Factor for the metric distance")
    parser.add_argument("--cosine_dist_factor", default=50.0, type=float, help="Factor for the cosine distance")

    # Other parameters
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Starting learning rate.")
    parser.add_argument("--l2", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size.")

    parser.add_argument("--epochs", default=50, type=int, help="Number of training epochs.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="""If the norm of the gradient vector exceeds this, renormalize it to max_grad_norm""")
    parser.add_argument("--gpus", default=[], nargs="+", type=int, help="Use CUDA on the listed devices.")
    parser.add_argument("--metric", default="hyperbolic", type=str, help="Metric of the space to use")
    parser.add_argument("--export_path", default="", type=str, help="Name of model to export")


parser = argparse.ArgumentParser("train.py")
config_parser(parser)
args = parser.parse_args()

if args.gpus:
    torch.cuda.set_device(args.gpus[0])

seed = random.randint(1, 100000)
figet.utils.set_seed(seed)

log = figet.utils.get_logging()
log.debug(args)


def get_dataset(data, args, key):
    dataset = data[key]
    log.info("Setting batch size")
    dataset.set_batch_size(args.batch_size)
    return dataset


def main():
    # Load data.
    log.debug("Loading data from '%s'." % args.data)
    data = torch.load(args.data + "/data.pt")
    vocabs = data["vocabs"]

    # datasets
    train_data = get_dataset(data, args, "train")
    dev_data = get_dataset(data, args, "dev")
    test_data = get_dataset(data, args, "test")

    log.debug("Loading word2vecs from '%s'." % args.data)
    word2vec = torch.load(args.data + "/word2vec.pt")
    log.debug("Loading type2vecs from '%s'." % args.data)
    type2vec = torch.load(args.data + "/type2vec.pt").type(torch.float)

    args.type_dims = type2vec.size(1)

    log.debug("Building model...")
    model = figet.Models.Model(args, vocabs)

    if len(args.gpus) >= 1:
        model.cuda()

    log.debug("Copying embeddings to model...")
    model.init_params(word2vec, type2vec)
    optim = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2)

    n_params = sum([p.nelement() for p in model.parameters()])
    log.debug(f"Number of parameters: {n_params}")

    coach = figet.Coach(model, optim, vocabs, train_data, dev_data, test_data, type2vec, word2vec, args)

    # Train.
    log.info("Start training...")
    coach.train()

    if args.export_path:
        torch.save(model.state_dict(), f"models/{args.export_path}.pt")

    log.info("Done!")


if __name__ == "__main__":
    main()
