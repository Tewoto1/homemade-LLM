"""
Cross entropy over logits, with padding masked out.
"""

import torch    # pyright: ignore[reportMissingImports]

# Not a valid vocab id, so it cannot collide with a real token the way a
# tokenizer's own pad id would
IGNORE_INDEX = -100


def cross_entropy_loss(logits, targets, ignore_index = IGNORE_INDEX):
    """logits (b, s, V), targets (b, s) of ids. Returns (loss, dL/dlogits, n_valid).

    Takes LOGITS, not probabilities: fusing the softmax in is what gives the clean
    gradient softmax(logits) - onehot. Going via probabilities means dividing by p,
    which blows up exactly where the model is most confidently wrong.

    Normalization by the number of REAL tokens happens here and nowhere else. Every
    layer's backward is unnormalized, so this 1/n propagates through all of them by
    the chain rule.
    """
    logits.sub_(logits.max(dim = -1, keepdim = True).values) # For stability, making max at most 0

    valid = (targets != ignore_index)                               # (b, s) -- the mask
    n = valid.sum()
    if n == 0:
        return logits.new_zeros(()), torch.zeros_like(logits), 0

    safe = targets.clamp_min(0)                                     # gather cannot take -100
    target_logit = logits.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    nll = torch.logsumexp(logits, dim = -1) - target_logit          # (b, s) = -log p(target)
    loss = (nll * valid).sum() / n

    grad = torch.softmax(logits, dim = -1)
    grad.scatter_add_(-1, safe.unsqueeze(-1), -torch.ones_like(target_logit).unsqueeze(-1))
    grad = grad * valid.unsqueeze(-1) / n     # zero the padded rows, then normalize
    return loss, grad, int(n)