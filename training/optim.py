"""
Implementing the AdamW optimizer
"""

import torch
from torch import nn
# Implementation plan: 
# TODO1: Rewire each module in models so that the gradients all live in layer.grads["name"]
#   Done by codex, need check/test through, relatively straight forward check
# TODO: Implement optimizer class
#   The method is each model module will have a module.optim

class AdamW():
    # performs AdamW on a single thing
    def __init__(self, beta1, beta2, param):
        self.beta1 = beta1
        self.beta2 = beta2
        self.param = param # pass in by reference
        # step data
        self.momentum = 0
        self.velo = 0
        self.max_velo = 0

    def step(self, gradient):
        # gradient same shape as param
        assert (gradient.shape == self.param.shape), "bad"