#!/usr/bin/env python
# encoding: utf-8

# Usage: python infer.py --file models/et/freq-et-02.pt --prep freq-cooc --gpus=0   OPT: --metric=euclid if not hyper

import torch
import argparse
import figet
import itertools

parser = argparse.ArgumentParser("infer.py")

# Sentence-level context parameters
parser.add_argument("--emb_size", default=300, type=int, help="Embedding size.")
parser.add_argument("--char_emb_size", default=50, type=int, help="Char embedding size.")
parser.add_argument("--positional_emb_size", default=25, type=int, help="Positional embedding size.")
parser.add_argument("--context_rnn_size", default=200, type=int, help="RNN size of ContextEncoder.")

parser.add_argument("--attn_size", default=100, type=int, help="Attention vector size.")
parser.add_argument("--negative_samples", default=10, type=int, help="Amount of negative samples.")
parser.add_argument("--neighbors", default=30, type=int, help="Amount of neighbors to analize.")

# Other parameters
parser.add_argument("--bias", default=0, type=int, help="Whether to use bias in the linear transformation.")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Starting learning rate.")
parser.add_argument("--l2", default=0.00, type=float, help="L2 Regularization.")
parser.add_argument("--param_init", default=0.01, type=float,
                    help=("Parameters are initialized over uniform distribution"
                          "with support (-param_init, param_init)"))
parser.add_argument("--batch_size", default=512, type=int, help="Batch size.")
parser.add_argument("--mention_dropout", default=0.5, type=float, help="Dropout rate for mention")
parser.add_argument("--context_dropout", default=0.2, type=float, help="Dropout rate for context")
parser.add_argument("--niter", default=150, type=int, help="Number of iterations per epoch.")
parser.add_argument("--epochs", default=15, type=int, help="Number of training epochs.")
parser.add_argument("--max_grad_norm", default=5, type=float,
                    help="""If the norm of the gradient vector exceeds this, 
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument("--extra_shuffle", default=1, type=int,
                    help="""By default only shuffle mini-batch order; when true, shuffle and re-assign mini-batches""")
parser.add_argument('--seed', type=int, default=3435, help="Random seed")
parser.add_argument("--word2vec", default=None, type=str, help="Pretrained word vectors.")
parser.add_argument("--type2vec", default=None, type=str, help="Pretrained type vectors.")
parser.add_argument("--gpus", default=[], nargs="+", type=int, help="Use CUDA on the listed devices.")
parser.add_argument('--log_interval', type=int, default=1000, help="Print stats at this interval.")
parser.add_argument('--hidden_size', type=int, default=500)
parser.add_argument("--metric", default="hyper", type=str, help="Hyper prep or not.")

parser.add_argument('--file', help="model file with weights to process.")
parser.add_argument('--prep', help="Which prep to use.")

args = parser.parse_args()

DATA = f"/hits/fast/nlp/lopezfo/views/benultra/ckpt/prep/{args.prep}/benultra"  # local
if args.gpus:
    torch.cuda.set_device(args.gpus[0])
    DATA = f"/hits/basement/nlp/lopezfo/views/benultra/ckpt/prep/{args.prep}/benultra"  # haswell

log = figet.utils.get_logging()


def get_dataset(data, batch_size, key):
    dataset = data[key]
    dataset.set_batch_size(batch_size)
    return dataset


def main():
    log.debug("Loading data from '%s'." % DATA)
    data = torch.load(DATA + ".data.pt")
    vocabs = data["vocabs"]

    dev_data = get_dataset(data, 1024, "dev")
    test_data = get_dataset(data, 1024, "test")

    state_dict = torch.load(args.file)
    args.type_dims = state_dict["type_lut.weight"].size(1)

    proj_learning_rate = [args.learning_rate]   # not used
    proj_weight_decay = [0.0]                   # not used
    proj_bias = [1]                 # best param
    proj_hidden_layers = [1]        # best param
    proj_hidden_size = [args.hidden_size]
    proj_non_linearity = [None]         # not used
    proj_dropout = [0.3]                # not used

    k_neighbors = [4]                   # not used
    args.exp_name = f"infer"

    cosine_factors = [50]               # not used
    hyperdist_factors = [1]             # not used

    configs = itertools.product(proj_learning_rate, proj_weight_decay, proj_bias, proj_non_linearity, proj_dropout,
                                proj_hidden_layers, proj_hidden_size, cosine_factors, hyperdist_factors, k_neighbors)

    for config in configs:

        extra_args = {"activation_function": config[3]}

        args.proj_learning_rate = config[0]
        args.proj_weight_decay = config[1]
        args.proj_bias = config[2]
        args.proj_dropout = config[4]
        args.proj_hidden_layers = config[5]
        args.proj_hidden_size = config[6]

        args.cosine_factor = config[7]
        args.hyperdist_factor = config[8]

        args.neighbors = config[9]

        log.debug("Building model...")
        model = figet.Models.Model(args, vocabs, None, extra_args)
        model.load_state_dict(state_dict)

        if len(args.gpus) >= 1:
            model.cuda()

        type2vec = model.type_lut.weight.data

        coach = figet.Coach(model, None, vocabs, None, dev_data, test_data, None, type2vec, None, None, args, extra_args, config)

        # coach.validate_all_neighbors(test_data, "TEST", args.file.split("/")[-1], plot=True)
        #coach.print_full_validation(dev_data, "DEV")

        log.info("INFERENCE ON DEV DATA:")
        coach.instance_printer.show(dev_data)

        log.info("\n\nINFERENCE ON TEST DATA:")
        coach.instance_printer.show(test_data)

        coach.print_results(test_data, "TEST")


if __name__ == "__main__":
    main()
