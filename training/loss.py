import torch

def cross_entropy_loss(p: torch.Tensor, t: torch.Tensor, eps = 1e-7):
    p_clp = torch.clamp_min(p, eps)
    loss = -torch.sum(t * torch.log(p_clp), dim = -1)
    gradients = - t/p_clp
    return loss, gradients