import torch

def entropy_term(val_probs):
    val_probs = torch.rand(2, 10)
    log_probs = torch.nn.functional.softmax(val_probs, dim=-1)
    entropy = log_probs * torch.log(log_probs + 1e-8)
    return -torch.sum(entropy)