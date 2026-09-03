"""
State dicts, checkpoints and device movement.

No nn.Module here, so none of this is free. Everything below rides on one
convention: a parameter lives in self.X, its gradient in self.grads["X"] on the
same object, and containers hold sublayers as attributes or lists. named_slots walks
that with vars(), so adding a layer to model.py cannot silently drop it from a
checkpoint. Names look like: layers.0.attn.W_Q, layers.0.MLP.layers.0.W, head.W
"""

import os
import torch                # pyright: ignore[reportMissingImports]
from torch import nn        # pyright: ignore[reportMissingImports]

from model import LLM


def _is_submodule(v):
    return hasattr(v, "params") and callable(getattr(v, "params"))


def named_slots(module, prefix = ""):
    """(name, owner, param_attr, grad_name) for every learnable tensor.

    Yields the owner and names, not the tensors: backward() may replace a gradient,
    so consumers should read owner.grads[grad_name] each step.
    """
    grads = getattr(module, "grads", {})
    for key, value in vars(module).items():
        name = prefix + key
        if isinstance(value, nn.Parameter):
            # A tied weight is listed once, under the layer that owns it. Without
            # this skip a tied head would yield the same Parameter a second time,
            # and an optimizer would step it twice per update.
            if key in getattr(module, "shared_params", ()):
                continue
            if key in grads:
                yield name, module, key, key
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if _is_submodule(item):
                    yield from named_slots(item, f"{name}.{i}.")
        elif _is_submodule(value):
            yield from named_slots(value, f"{name}.")


def named_parameters(model):
    return [(n, getattr(o, pa)) for n, o, pa, _ in named_slots(model)]


def named_gradients(model):
    return [(n, o.grads[gn]) for n, o, _, gn in named_slots(model)]


def clip_grad_norm(model, max_norm):
    """Scale gradients down in place if the global norm exceeds max_norm.
    Returns the norm before clipping -- it spikes well before the loss does.

    The norm is reduced on-device and read back ONCE. Calling float() per tensor
    instead would be ~100 GPU-to-CPU syncs per step, each one stalling the pipeline
    while everything queued behind it drains -- enough to dominate step time on a
    fast card, for a number that is only used for logging and one comparison.
    """
    grads = [g for _, g in named_gradients(model)]
    norm = float(torch.stack([g.pow(2).sum() for g in grads]).sum().sqrt())
    if max_norm and norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for g in grads:
            g.mul_(scale)
    return norm


def to_device(module, device):
    """Move every tensor under module onto device, in place.

    Deliberately broader than named_slots: it also catches buffers, above all
    Embed.pos_embed, which for "cosine" is a plain tensor and would otherwise stay
    on the CPU and blow up on the first add to a CUDA token embedding.
    """
    for key, value in vars(module).items():
        if isinstance(value, nn.Parameter):
            value.data = value.data.to(device)
        elif torch.is_tensor(value):
            setattr(module, key, value.to(device))
        elif isinstance(value, dict):
            for item_key, item in value.items():
                if torch.is_tensor(item):
                    value[item_key] = item.to(device)
        elif isinstance(value, (list, tuple)):
            for item in value:
                if _is_submodule(item):
                    to_device(item, device)
        elif _is_submodule(value):
            to_device(value, device)
    return module


def state_dict(model):
    return {n: p.data.detach().cpu().clone() for n, p in named_parameters(model)}


def load_state_dict(model, sd, strict = True):
    own = dict(named_parameters(model))
    missing = [k for k in own if k not in sd]
    unexpected = [k for k in sd if k not in own]
    if strict and (missing or unexpected):
        raise KeyError(f"state_dict mismatch. missing={missing} unexpected={unexpected}")
    for name, param in own.items():
        if name in sd:
            param.data.copy_(sd[name].to(param.data.device, dtype = param.data.dtype))
    return missing, unexpected


# LLM.meta_data uses readable keys; these are the constructor kwargs behind them
_META_TO_KWARG = {
    "layers": "layers", "hidden dimension": "hid_dim", "attention dimension": "attn_dim",
    "attention heads": "heads", "vocab size": "vocab_size", "position embedding": "pos_embed",
    "maxmium sequence length": "max_s",   # typo lives in model.py, keep them in sync
    "tied embedding": "tie",
}


def save_model(model, path, **extra):
    """Architecture + weights. **extra (step, losses, optimizer state) rides along."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok = True)
    config = {kw: model.meta_data[mk] for mk, kw in _META_TO_KWARG.items() if mk in model.meta_data}
    torch.save({"config": config, "state_dict": state_dict(model), "extra": extra}, path)
    return path


def load_model(path, device = "cpu", strict = True):
    """Returns (model, extra). weights_only=False since the checkpoint holds a
    config dict, so only load checkpoints you made yourself."""
    ckpt = torch.load(path, map_location = "cpu", weights_only = False)
    model = LLM(**ckpt["config"])
    load_state_dict(model, ckpt["state_dict"], strict = strict)
    return to_device(model, device), ckpt.get("extra", {})
