import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision


def flip_signs_randomly(x):
    signs = np.random.randint(low=0, high=2, size=len(x))
    signs[signs < 1] = -1
    return x * signs


def random_translate(imgs, min_translate=0, max_translate=6):
    n, h, w, c = imgs.shape

    # Randomly choose the amount to translate in each dimension.
    w_translations = np.random.randint(low=min_translate, high=max_translate, size=n)
    w_translations = flip_signs_randomly(w_translations)
    h_translations = np.random.randint(low=min_translate, high=max_translate, size=n)
    h_translations = flip_signs_randomly(h_translations)

    # Translate along width and height axes, filling with zeros.
    translated = torch.empty((n, h, w, c), dtype=imgs.dtype, device=imgs.device)
    for i, (img, h_t, w_t) in enumerate(zip(imgs, h_translations, w_translations)):
        translated[i] = torch.roll(img, (h_t, w_t), dims=(0, 1))
        if h_t > 0:
            translated[i][:h_t] = 0
        elif h_t < 0:
            translated[i][h_t:] = 0
        if w_t > 0:
            translated[i][:, :w_t] = 0
        elif w_t < 0:
            translated[i][:, w_t:] = 0


def random_translate_out(imgs, out, min_translate=0, max_translate=6):
    n, h, w, c = imgs.shape

    # Randomly choose the amount to translate in each dimension.
    w_translations = np.random.randint(low=min_translate, high=max_translate, size=n)
    w_translations = flip_signs_randomly(w_translations)
    h_translations = np.random.randint(low=min_translate, high=max_translate, size=n)
    h_translations = flip_signs_randomly(h_translations)

    # Translate along width and height axes, filling with zeros.
    for i, (img, h_t, w_t) in enumerate(zip(imgs, h_translations, w_translations)):
        out[i] = torch.roll(img, (h_t, w_t), dims=(0, 1))
        if h_t > 0:
            out[i][:h_t] = 0
        elif h_t < 0:
            out[i][h_t:] = 0
        if w_t > 0:
            out[i][:, :w_t] = 0
        elif w_t < 0:
            out[i][:, w_t:] = 0

    # import matplotlib.pyplot as plt
    # for i in range(10):
    #     fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    #     fig.suptitle(
    #         f"width translation: {w_translations[i]}, height translation: {h_translations[i]}")
    #     axs[0][0].imshow(imgs[i, :, :, :3].detach().cpu().numpy())
    #     axs[0][1].imshow(imgs[i, :, :, 3:].detach().cpu().numpy())
    #     axs[1][0].imshow(out[i, :, :, :3].detach().cpu().numpy())
    #     axs[1][1].imshow(out[i, :, :, 3:].detach().cpu().numpy())
    #     plt.show()

    # return out


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

    time_function(random_translate, num_iters, obs)

    out = torch.empty_like(obs, dtype=obs.dtype, device=obs.device)

    time_function(random_translate_out, num_iters, obs, out)
