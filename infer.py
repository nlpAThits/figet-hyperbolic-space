#!/usr/bin/env python
# encoding: utf-8

import torch
import argparse
import figet
from train import config_parser, get_dataset


parser = argparse.ArgumentParser("infer.py")
config_parser(parser)
args = parser.parse_args()

if args.gpus:
    torch.cuda.set_device(args.gpus[0])

log = figet.utils.get_logging()


def main():
    log.debug("Loading data from '%s'." % args.data)
    data = torch.load(args.data + "/data.pt")
    vocabs = data["vocabs"]

    dev_data = get_dataset(data, args, "dev")
    test_data = get_dataset(data, args, "test")

    state_dict = torch.load("models/" + args.export_path + ".pt")
    args.type_dims = state_dict["type_lut.weight"].size(1)

    log.debug("Building model...")
    model = figet.Models.Model(args, vocabs)
    model.load_state_dict(state_dict)

    if len(args.gpus) >= 1:
        model.cuda()

    type2vec = model.type_lut.weight.data

    coach = figet.Coach(model, None, vocabs, None, dev_data, test_data, type2vec, None, args)

    log.info("INFERENCE ON DEV DATA:")
    coach.instance_printer.show(dev_data)
    coach.print_results(dev_data, "DEV")

    log.info("\n\nINFERENCE ON TEST DATA:")
    coach.instance_printer.show(test_data)

    coach.print_results(test_data, "TEST")


if __name__ == "__main__":
    main()
