"""
Training an LLM from scratch v1
This file is about defining loss, defining a single training step and propping loss back overall
batched training needs to be implemented
SDG for now, will implement AdamW later
"""

import torch
from loss import cross_entropy_loss
from model import LLM

