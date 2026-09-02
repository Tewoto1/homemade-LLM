import torch

def cross_entropy_loss(p: torch.Tensor, t: torch.Tensor, pad: int, vocab_size, eps = 1e-7):
    # t is of shape (b, s), we first make it onehot
    one_hot = torch.nn.functional.one_hot(t, vocab_size) # (b, s, v)
    one_hot[:, :, pad] = 0
    p_clp = torch.clamp_min(p, eps)
    loss = -torch.sum(one_hot * torch.log(p_clp), dim = -1)
    gradients = - one_hot/p_clp
    return loss, gradients # loss of shape (b, s) and gradients of shape (b, s, vocab_size)