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

def train_single_step(model: LLM, x: torch.Tensor, target: torch.Tensor = None, loss = cross_entropy_loss):
    pred = model.forward(x)
    # x of shape (b, s)
    if target == None:
        target = x[:, 1:]
    loss, grad_o = loss(pred, target)