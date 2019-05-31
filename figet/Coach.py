#!/usr/bin/env python
# encoding: utf-8

import copy
import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
from statistics import mean
from tensorboardX import SummaryWriter

from figet.utils import get_logging
from figet.Predictor import kNN, assign_total_types
from figet.evaluate import evaluate, stratified_evaluate
from figet.Constants import TYPE_VOCAB
from figet.instance_printer import InstancePrinter

log = get_logging()


class Coach(object):

    def __init__(self, model, optim, vocabs, train_data, dev_data, test_data, type2vec, word2vec, args):
        self.model = model
        self.optim = optim
        self.vocabs = vocabs
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.args = args
        self.word2vec = word2vec
        self.type2vec = type2vec
        self.knn = kNN(type2vec, vocabs[TYPE_VOCAB], args.metric)
        self.instance_printer = InstancePrinter(vocabs, model, self.knn)
        self.writer = SummaryWriter(f"tensorboard/{args.export_path}")

    def train(self):
        log.debug(self.model)

        max_coarse_macro_f1, best_model_state, best_epoch = -1, None, 0

        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)

            results, dev_loss = self.infer_types(self.dev_data, "dev", epoch)
            coarse_results, _, _ = stratified_evaluate(results, self.vocabs[TYPE_VOCAB])
            coarse_split = coarse_results.split()
            coarse_macro_f1 = float(coarse_split[5])

            log.info(f"Results epoch {epoch}: Train loss: {train_loss:.2f}, dev loss: {dev_loss:.2f}, "
                     f"coarse macro F1 {coarse_macro_f1:.2f}")

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("dev/loss", dev_loss, epoch)
            self.writer.add_scalar("dev/strict_f1", float(coarse_split[2]), epoch)
            self.writer.add_scalar("dev/macro_f1", coarse_macro_f1, epoch)
            self.writer.add_scalar("dev/micro_f1", float(coarse_split[8]), epoch)

            if coarse_macro_f1 > max_coarse_macro_f1:
                max_coarse_macro_f1 = coarse_macro_f1
                best_model_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                log.info(f"* Best coarse macro F1 {coarse_macro_f1:0.2f} at epoch {epoch} *")

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        log.info(f"Final evaluation on best coarse macro F1 ({max_coarse_macro_f1}) from epoch {best_epoch}")
        self.model.load_state_dict(best_model_state)

        self.print_results(self.dev_data, "dev")
        self.print_results(self.test_data, "test")
        self.writer.close()

    def train_epoch(self, epoch):
        """:param epoch: int >= 1"""
        self.train_data.shuffle()
        total_loss = []
        # angles, dist_to_pos, euclid_dist, norms
        stats = [[[], [], [], []],
                 [[], [], [], []],
                 [[], [], [], []]]

        self.set_learning_rate(epoch)
        self.model.train()
        for i in tqdm(range(len(self.train_data)), desc="train_epoch_{}".format(epoch)):
            batch = self.train_data[i]

            self.optim.zero_grad()
            loss, predicted_embeds, _, angles, dist_to_pos, euclid_dist = self.model(batch)
            loss.backward()
            self.write_projector_norm(epoch, i)
            if self.args.max_grad_norm >= 0:
                clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optim.step()

            # Stats.
            for idx, item in enumerate(stats):
                item[0].append(angles[idx].mean().item())
                item[1].append(dist_to_pos[idx].mean().item())
                item[2].append(euclid_dist[idx].mean().item())
                item[3].append(torch.norm(predicted_embeds[idx].detach(), p=2, dim=1).mean().item())

            total_loss.append(loss.item())

        self.write_stats(stats, "train", epoch)
        return np.mean(total_loss)

    def print_results(self, dataset, name):
        log.info(f"\n\n\nResults on {name.upper()}")
        total_true_and_pred, _ = self.infer_types(dataset, name, -1)

        combined_eval = evaluate(total_true_and_pred)
        strat_eval = stratified_evaluate(total_true_and_pred, self.vocabs[TYPE_VOCAB])
        titles = ["Coarse", "Fine", "UltraFine"]
        string = "\n".join([f"{titles[i]}\n{strat_eval[i]}" for i in range(len(strat_eval))])

        log.info("Strict (p,r,f1), Macro (p,r,f1), Micro (p,r,f1)\n" + combined_eval)
        log.info(f"Stratified evaluation on {name.upper()}:\n{string}")

    def infer_types(self, data, name, epoch):
        total_loss, total_results = [], []
        # angles, dist_to_pos, euclid_dist, norms
        stats = [[[], [], [], []],
                 [[], [], [], []],
                 [[], [], [], []]]
        self.model.eval()
        with torch.no_grad():
            for i in tqdm(range(len(data)), desc=f"validate_typing_{name}_{epoch}"):
                batch = data[i]
                types = batch[5]

                loss, predicted_embeds, _, angles, dist_to_pos, euclid_dist = self.model(batch)

                partial_total_result = assign_total_types(predicted_embeds, types, self.knn)
                total_results += partial_total_result

                # collect projection stats
                total_loss.append(loss.item())
                for idx, item in enumerate(stats):
                    item[0].append(angles[idx].mean().item())
                    item[1].append(dist_to_pos[idx].mean().item())
                    item[2].append(euclid_dist[idx].mean().item())
                    item[3].append(torch.norm(predicted_embeds[idx].detach(), p=2, dim=1).mean().item())

            self.write_stats(stats, name, epoch)

            return total_results, np.mean(total_loss)

    def set_learning_rate(self, epoch):
        """
        Sets a reduced learning rate for the first and last few epochs
        :param epoch: 1-numerated
        """
        if epoch <= 5 or epoch > int(self.args.epochs * 0.9):
            learning_rate = self.args.learning_rate / 10
        else:
            learning_rate = self.args.learning_rate
        for g in self.optim.param_groups:
            g['lr'] = learning_rate

    def write_stats(self, stats, name, epoch):
        labels = ["coarse", "fine", "ultrafine"]
        for idx, item in enumerate(stats):
            if epoch != -1:
                prefix = f"Proj/{name}_{labels[idx]}"
                self.writer.add_scalar(f"{prefix}_angles", mean(item[0]), epoch)
                self.writer.add_scalar(f"{prefix}_d_to_pos", mean(item[1]), epoch)
                self.writer.add_scalar(f"{prefix}_euclid", mean(item[2]), epoch)
                self.writer.add_scalar(f"{prefix}_norm", mean(item[3]), epoch)

    def write_projector_norm(self, epoch, iter_i):
        labels = ["coarse", "fine", "ultrafine"]
        for granularity, layer in zip(labels, [self.model.coarse_projector.W_out, self.model.fine_projector.W_out,
                                               self.model.ultrafine_projector.W_out]):
            gradient = layer.weight.grad
            if gradient is not None:
                grad_norm = gradient.data.norm(2).item()
                self.writer.add_scalar(f"norm_{granularity}/epoch_{epoch}", grad_norm, iter_i)
