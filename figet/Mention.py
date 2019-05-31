#!/usr/bin/env python
# encoding: utf-8

import torch
import figet.Constants as const


class Mention(object):

    def __init__(self, fields):
        self.fields = fields

    def preprocess(self, vocabs, args):
        self.vocabs = vocabs
        self.context_len = args.side_context_length
        self.mention_len = args.mention_length
        self.mention_char_len = args.mention_char_length

        self.types = self.type_idx()        # type index in vocab
        self.mention = self.get_mention_idx()
        self.mention_chars = self.get_mention_chars()
        self.left_context = self.left_context_idx()
        self.right_context = self.right_context_idx()

    def get_mention_idx(self):
        head = self.fields[const.MENTION].split()[:self.mention_len]
        if not head:
            return torch.LongTensor([const.PAD])
        return self.vocabs[const.TOKEN_VOCAB].convert_to_idx(head, const.UNK_WORD)

    def left_context_idx(self):
        left_context_words = self.fields[const.LEFT_CTX][-self.context_len:]
        return self.vocabs[const.TOKEN_VOCAB].convert_to_idx(left_context_words, const.UNK_WORD)

    def right_context_idx(self):
        right_context_words = self.fields[const.RIGHT_CTX][:self.context_len]
        return self.vocabs[const.TOKEN_VOCAB].convert_to_idx(right_context_words, const.UNK_WORD)

    def type_idx(self):
        types = []
        for mention_type in self.fields[const.TYPE]:
            types.append(self.vocabs[const.TYPE_VOCAB].lookup(mention_type))
        return torch.LongTensor(types)

    def get_mention_chars(self):
        chars = self.fields[const.MENTION][:self.mention_char_len]
        if not chars:
            return torch.LongTensor([const.PAD])
        return self.vocabs[const.CHAR_VOCAB].convert_to_idx(chars, const.UNK_WORD)

    def type_len(self):
        return len(self.fields[const.TYPE])

    def clear(self):
        del self.fields
        del self.mention
        del self.mention_chars
        del self.left_context
        del self.right_context
        del self.types




