import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def assert_all_the_same_length(*tensors):
    if len(tensors) == 0:
        return
    length = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == length


def sample_beta(size, concentration):
    concentration = torch.ones(size) * concentration
    return torch.distributions.beta.Beta(concentration, concentration).sample()


def apply_mixreg(*tensors, concentration=0.2):
    assert len(tensors) > 0
    assert_all_the_same_length(tensors)
    batch_size = len(tensors[0])

    lmbda = sample_beta(batch_size, concentration)

    permutation = torch.randperm(batch_size)
    mixed_tensors = []
    for tensor in tensors:
        broadcast_shape = [-1] + [1] * (len(tensor.shape) - 1)
        broadcast_lmbda = lmbda.reshape(broadcast_shape).to(tensor.device)
        other_tensor = tensor[permutation]
        mixed_tensor = broadcast_lmbda * tensor + (1 - broadcast_lmbda) * other_tensor
        mixed_tensors.append(mixed_tensor)
    return mixed_tensors, lmbda, permutation


def generate_batch():
    import copy

    num_imgs = 512
    height = 64
    width = 64

    import gym
    env = gym.make("procgen:procgen-caveflyer-v0")
    obs = []
    x = env.reset()
    obs.append(x)
    for i in range(1, num_imgs):
        x, _, done, _ = env.step(env.action_space.sample())
        obs.append(x)
    obs = torch.tensor(obs).to("cuda")

    value_targets = torch.randn(num_imgs, 1)
    action_dim = 5
    logits = torch.randn(num_imgs, action_dim)

    return dict(obs=obs, value_targets=value_targets, logits=logits)


def plot_mixreg_results(orig_obs, orig_values, orig_logits, obs, values, logits, lmbda,
                        permutation):
    batch_size = len(orig_obs)
    for i in range(batch_size):
        # Plot both the original images and the combination.
        fig, axs = plt.subplots(3, 1, figsize=(16, 16))
        a = orig_obs[i].detach().cpu().numpy()
        axs[0].set_title(f"V: {orig_values[i].cpu().numpy()}")
        axs[0].imshow(a)

        b = orig_obs[permutation[i]].detach().cpu().numpy()
        axs[1].set_title(f"V: {orig_values[permutation[i]].cpu().numpy()}")
        axs[1].imshow(b)

        mixture = obs[i].to(torch.int).detach().cpu().numpy()
        axs[2].set_title(f"V: {values[i].cpu().numpy()}")
        axs[2].imshow(mixture)

        fig.suptitle(f"Lambda: {lmbda[i]}")
        plt.show()
        plt.close()


def main():
    batch = generate_batch()
    mixed_batch, lmbda, permutation = apply_mixreg(batch["obs"], batch["value_targets"],
                                                   batch["logits"])
    plot_mixreg_results(batch["obs"], batch["value_targets"], batch["logits"], *mixed_batch, lmbda,
                        permutation)


if __name__ == "__main__":
    main()
