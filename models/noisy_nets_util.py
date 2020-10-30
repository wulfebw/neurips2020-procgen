import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def eps_fn(eps):
    return torch.sign(eps) * torch.sqrt(torch.abs(eps))


class NoisyLinear(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 noise_mode="independent",
                 indp_sigma_init=0.017,
                 dtype=torch.float32):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.noise_mode = noise_mode
        self.indp_sigma_init = indp_sigma_init
        self.dtype = dtype

        self.w_mu = nn.parameter.Parameter(torch.Tensor(out_size, in_size).to(dtype))
        self.w_sigma = nn.parameter.Parameter(torch.Tensor(out_size, in_size).to(dtype))

        self.b_mu = nn.parameter.Parameter(torch.Tensor(out_size).to(dtype))
        self.b_sigma = nn.parameter.Parameter(torch.Tensor(out_size).to(dtype))

        self.reset_parameters()

    def reset_parameters(self):
        if self.noise_mode == "independent":
            mu_bound = math.sqrt(3 / self.in_size)
            nn.init.uniform_(self.w_mu, -mu_bound, mu_bound)
            nn.init.uniform_(self.b_mu, -mu_bound, mu_bound)
            nn.init.constant_(self.w_sigma, self.indp_sigma_init)
            nn.init.constant_(self.b_sigma, self.indp_sigma_init)
        else:
            raise NotImplementedError(self.noise_mode)

    @property
    def device(self):
        return self.w_mu.device

    def sample_noise(self, batch_size=1):
        if self.noise_mode == "independent":
            w_noise = torch.randn(batch_size, self.out_size, self.in_size)
            b_noise = torch.randn(batch_size, self.out_size)
        elif self.noise_mode == "factored":
            eps_in = eps_fn(torch.randn(batch_size, self.in_size))
            eps_out = eps_fn(torch.randn(batch_size, self.out_size))
            w_noise = torch.einsum("bp, bq->bpq", eps_in, eps_out)
            b_noise = eps_out
        else:
            raise NotImplementedError(self.noise_mode)

        return w_noise.to(self.device), b_noise.to(self.device)

    def forward(self, x, w_eps, b_eps):
        # If the noise tensors are None, take the mean output.
        if w_eps is None or b_eps is None:
            return F.linear(x, self.w_mu, self.b_mu)

        # This broadcasts the weights such that each weight along the batch axis is distinct.
        # This is very space inefficient, and probably also very computationally inefficient,
        # but I'm only sure about the first one.
        w = self.w_mu + self.w_sigma * w_eps
        b = self.b_mu + self.b_sigma * b_eps
        outputs = torch.bmm(w, x[:, :, None]).squeeze(2) + b
        return outputs
