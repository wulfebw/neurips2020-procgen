import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision


def plot_translated(imgs, translated, w_translations, h_translations):
    import matplotlib.pyplot as plt
    for i in range(5):
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(
            f"width translation: {w_translations[i]}, height translation: {h_translations[i]}")
        axs[0][0].imshow(imgs[i, :, :, :3].detach().cpu().numpy())
        axs[0][1].imshow(imgs[i, :, :, 3:].detach().cpu().numpy())
        axs[1][0].imshow(translated[i, :, :, :3].detach().cpu().numpy())
        axs[1][1].imshow(translated[i, :, :, 3:].detach().cpu().numpy())
        plt.show()


def flip_signs_randomly(x):
    signs = np.random.randint(low=0, high=2, size=len(x))
    signs[signs < 1] = -1
    return x * signs


def sample_translations(min_translate, max_translate, size):
    w_translations = np.random.randint(low=min_translate, high=max_translate, size=size)
    w_translations = flip_signs_randomly(w_translations)
    h_translations = np.random.randint(low=min_translate, high=max_translate, size=size)
    h_translations = flip_signs_randomly(h_translations)
    return h_translations, w_translations


def random_translate_via_roll(imgs, min_translate=0, max_translate=6):
    n, h, w, c = imgs.shape

    # Randomly choose the amount to translate in each dimension.
    h_translations, w_translations = sample_translations(min_translate, max_translate, n)

    # Translate along width and height axes, filling with zeros.
    translated = torch.empty((n, h, w, c), dtype=imgs.dtype, device=imgs.device)
    for i, (img, h_t, w_t) in enumerate(zip(imgs, h_translations, w_translations)):
        translated[i] = torch.roll(img, (h_t, w_t), dims=(0, 1))
        if h_t > 0:
            translated[i, :h_t] = 0
        elif h_t < 0:
            translated[i, h_t:] = 0
        if w_t > 0:
            translated[i, :, :w_t] = 0
        elif w_t < 0:
            translated[i, :, w_t:] = 0

    return translated


def random_translate_via_index(imgs, min_translate=0, max_translate=6):
    n, h, w, c = imgs.shape

    # Randomly choose the amount to translate in each dimension.
    h_translations, w_translations = sample_translations(min_translate, max_translate, n)

    # Translate along width and height axes, filling with zeros.
    translated = torch.empty((n, h, w, c), dtype=imgs.dtype, device=imgs.device)
    for i, (img, h_t, w_t) in enumerate(zip(imgs, h_translations, w_translations)):
        # This could be shorter.
        if h_t > 0:
            if w_t == 0:
                translated[i, h_t:] = img[:-h_t]
                translated[i, :h_t] = 0
            elif w_t > 0:
                translated[i, h_t:, w_t:] = img[:-h_t, :-w_t]
                translated[i, :h_t, :w_t] = 0
            elif w_t < 0:
                translated[i, h_t:, :w_t] = img[:-h_t, -w_t:]
                translated[i, :h_t, w_t:] = 0
        elif h_t < 0:
            if w_t == 0:
                translated[i, :h_t] = img[-h_t:]
                translated[i, h_t:] = 0
            elif w_t > 0:
                translated[i, :h_t, w_t:] = img[-h_t:, :-w_t]
                translated[i, h_t:, :w_t] = 0
            elif w_t < 0:
                translated[i, :h_t, :w_t] = img[-h_t:, -w_t:]
                translated[i, h_t:, w_t:] = 0
        elif h_t == 0:
            if w_t == 0:
                translated[i] = img
            elif w_t > 0:
                translated[i, :, w_t:] = img[:, :-w_t]
                translated[i, :, :w_t] = 0
            elif w_t < 0:
                translated[i, :, :w_t] = img[:, -w_t:]
                translated[i, :, w_t:] = 0

    return translated


if __name__ == "__main__":

    def time_function(f, num_iters, *args):
        import time
        start = time.time()
        for itr in range(num_iters):
            sys.stdout.write(f"\r{itr + 1} / {num_iters}")
            f(*args)
        end = time.time()
        print(f"\nTook {end - start:0.4f} seconds")

    num_imgs = 2048
    height = 64
    width = 64
    channels = 6
    num_iters = 100

    obs = np.random.randint(low=0,
                            high=256,
                            size=num_imgs * height * width * channels,
                            dtype=np.uint8)
    obs = obs.reshape(num_imgs, height, width, channels)
    obs = torch.tensor(obs).to("cuda")

    time_function(random_translate_via_roll, num_iters, obs)
    time_function(random_translate_via_index, num_iters, obs)

    np.random.seed(1)
    result_a = random_translate_via_roll(obs)
    np.random.seed(1)
    result_b = random_translate_via_index(obs)

    np.testing.assert_array_almost_equal(result_a.detach().cpu().numpy(),
                                         result_b.detach().cpu().numpy())
