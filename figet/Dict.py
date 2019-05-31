#!/usr/bin/env python
# encoding: utf-8

import torch
from . import Constants
from .Constants import COARSE, FINE


class Dict(object):
    """
    Object that keeps a mapping between labels and ids. It also keeps the frequency of each term.
    """

    def __init__(self, data=None, lower=False):
        self.idx2label = {}
        self.label2idx = {}
        self.frequencies = {}
        self.lower = lower

        self.special = []

        if data is not None:
            if isinstance(data, str):
                self.load_file(data)
            else:
                self.add_specials(data)

    def size(self):
        return len(self.idx2label)

    def load_file(self, filepath):
        for line in open(filepath):
            fields = line.strip().split()
            label = fields[0]
            idx = int(fields[1])
            self.add(label, idx)

    def write_file(self, filepath):
        with open(filepath, "w") as f:
            for i in range(self.size()):
                label = self.idx2label[i]
                f.write("%s %d\n" % (label, i))

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.label2idx[key]
        except KeyError:
            return default

    def get_label(self, idx, default=None):
        try:
            return self.idx2label[idx]
        except KeyError:
            return default

    def add_special(self, label, idx=None):
        idx = self.add(label, idx)
        self.special.append(idx)

    def add_specials(self, labels):
        for label in labels:
            self.add_special(label)

    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idx2label[idx] = label
            self.label2idx[label] = idx
        else:
            if label in self.label2idx:
                idx = self.label2idx[label]
            else:
                idx = len(self.idx2label)
                self.idx2label[idx] = label
                self.label2idx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size=None):
        """
        [I think] Returns a copy of the Dict with only the most :size frequent words
        """
        if size and size >= self.size():
            return self

        if size is None:
            size = self.size()

        freq = torch.Tensor([self.frequencies[i] for i in range(len(self.frequencies))])
        _, idx = torch.sort(freq, 0, True)

        ret = Dict()
        ret.lower = self.lower

        for i in self.special:
            ret.add_special(self.idx2label[i])

        for i in idx[:size]:
            ret.add(self.idx2label[i])

        return ret

    def convert_to_idx(self, labels, unk=None, bos=None, eos=None, _type=torch.LongTensor):
        vec = []

        if bos is not None:
            vec.append(self.lookup(bos))

        unk_idx = self.lookup(unk)
        vec += [self.lookup(label, default=unk_idx) for label in labels]

        if eos is not None:
            vec.append(self.lookup(eos))

        return _type(vec)

    def convert_to_labels(self, idx, eos=None):
        labels = []
        if len(idx.size()) == 0:
            return labels
        for i in idx:
            labels += [self.get_label(i)]
            if i == eos:
                break
        return labels


class TokenDict(Dict):

    def __init__(self, lower=False):
        Dict.__init__(self, [Constants.PAD_WORD, Constants.UNK_WORD], lower)
        self.label2wordvec_idx = {
            Constants.PAD_WORD: Constants.PAD,
            Constants.UNK_WORD: Constants.UNK
        }
        self.word2vecIdx2label = None

    def lookup(self, key, default=Constants.PAD):
        """
        If key has a word2vec vector, should return the idx
        If key doesn't have a word2vec vector, should return the idx of the unk vector
        """
        key = key.lower() if self.lower else key

        if key in self.label2wordvec_idx:
            return self.label2wordvec_idx[key]
        return default

    def size_of_word2vecs(self):
        return len(self.label2wordvec_idx)

    def get_label_from_word2vec_id(self, idx, default=None):
        if not self.word2vecIdx2label:
            self.word2vecIdx2label = {v: k for k, v in self.label2wordvec_idx.items()}
        try:
            return self.word2vecIdx2label[idx]
        except KeyError:
            return default


class TypeDict(Dict):

    def __init__(self, data=None, lower=False):
        Dict.__init__(self, data, lower)
        self.coarse_ids = None
        self.fine_ids = None
        self.ultrafine_ids = None
        for cotype in COARSE:
            self.add(cotype)
        for fitype in FINE:
            self.add(fitype)

    def get_coarse_ids(self):
        if not self.coarse_ids:
            self.coarse_ids = {self.label2idx[label] for label in COARSE if label in self.label2idx}
        return self.coarse_ids

    def get_fine_ids(self):
        if not self.fine_ids:
            self.fine_ids = {self.label2idx[label] for label in FINE if label in self.label2idx}
        return self.fine_ids

    def get_ultrafine_ids(self):
        if not self.ultrafine_ids:
            self.ultrafine_ids = {self.label2idx[label] for label in self.label2idx
                                  if label in self.label2idx and label not in COARSE and label not in FINE}
        return self.ultrafine_ids

