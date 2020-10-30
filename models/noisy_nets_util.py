import torch


def eps_fn(eps):
    return torch.sign(eps) * torch.sqrt(torch.abs(eps))


def sample_factored_noise(batch_size, in_size, out_size, device):
    eps_in = eps_fn(torch.randn(batch_size, in_size))
    eps_out = eps_fn(torch.randn(batch_size, out_size))
    eps_w = torch.einsum("bp, bq->bpq", eps_in, eps_out)
    eps_b = eps_out
    return eps_w.to(device), eps_b.to(device)
