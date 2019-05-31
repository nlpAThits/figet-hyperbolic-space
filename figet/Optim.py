#!/usr/bin/env python
# encoding: utf-8

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


class Optim(object):
    """DEPRECATED"""
    def __init__(self, params, learning_rate, max_grad_norm, l2_reg=0.0):
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.params = list(params)
        self.optimizer = optim.SGD(self.params, lr=self.learning_rate, weight_decay=l2_reg)

    def step(self):
        if self.max_grad_norm != -1:    # -1 by default
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict())

    def zero_grad(self):
        self.optimizer.zero_grad()
