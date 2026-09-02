"""
Training an LLM from scratch v1
This file is about defining loss, defining a single training step and propping loss back overall
batched training needs to be implemented
SDG for now, will implement AdamW later
"""

import torch
from loss import cross_entropy_loss
from model import LLM
from transformers import AutoTokenizer

class Trainer():
    def __init__(self, model, tokenizer, optimizer, scheduler, loss = cross_entropy_loss):
        self.model = model
        self.tok = tokenizer
        self.loss = loss
        self.optim = optimizer
        self.sche = scheduler
        self.pad = self.tok.padding_token
        self.losses = [] # Not sure if right data type to store a bajillion losses

    def train_single_step(self, x: torch.Tensor, target: torch.Tensor = None):
        x = self.tok(x)
        pred = self.model.forward(x)
        # x of shape (b, s)
        if target == None:
            target = x[:, 1:]
        loss, grad_o = self.loss(pred, target)
        self.losses.append(loss)
        # I might have to redefine loss in loss.py or smth but I want to mask out the loss on padding tokens
        # meaning in the loss if the thing is a padding token don't count it towards loss
        grad_o = self.optim(grad_o)
        self.model.backward(grad_o)
        self.model.step(self.sche.lr)
        self.sche.step()

    def train(self, dataset):
        for x in dataset.data:
            self.train_single_step(x)

    def loss(self, n: int):
        return self.losses[::n]