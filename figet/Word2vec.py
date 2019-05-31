#!/usr/bin/env python
# encoding: utf-8

from .Constants import UNK_WORD


class Word2Vec(object):

    def __init__(self):
        self.token2vec = {}

    def add(self, token, vector):
        self.token2vec[token] = vector

    def get_vec(self, word):
        if word in self.token2vec:
            return self.token2vec[word]
        if word.lower() in self.token2vec:
            return self.token2vec[word.lower()]

        return self.get_unk_vector()

    def __contains__(self, word):
        return word in self.token2vec or word.lower() in self.token2vec

    def get_unk_vector(self):
        return self.token2vec[UNK_WORD]
