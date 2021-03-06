import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision


def plot_translated(imgs, translated, w_translations, h_translations, num_display=10):
    n, h, w, c = imgs.shape
    import matplotlib.pyplot as plt
    for i in range(num_display):
        if c == 3:
            fig, axs = plt.subplots(2, 1, figsize=(16, 16))
            axs[0].imshow(imgs[i, :, :].detach().cpu().numpy())
            axs[1].imshow(translated[i, :, :, :3].detach().cpu().numpy())
        elif c == 6:
            fig, axs = plt.subplots(2, 2, figsize=(16, 16))
            axs[0][0].imshow(imgs[i, :, :, :3].detach().cpu().numpy())
            axs[0][1].imshow(imgs[i, :, :, 3:].detach().cpu().numpy())
            axs[1][0].imshow(translated[i, :, :, :3].detach().cpu().numpy())
            axs[1][1].imshow(translated[i, :, :, 3:].detach().cpu().numpy())

        fig.suptitle(
            f"width translation: {w_translations[i]}, height translation: {h_translations[i]}")
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
    translated = torch.zeros((n, h, w, c), dtype=imgs.dtype, device=imgs.device)
    for i, (img, h_t, w_t) in enumerate(zip(imgs, h_translations, w_translations)):
        if h_t > 0:
            if w_t == 0:
                translated[i, h_t:] = img[:-h_t]
            elif w_t > 0:
                translated[i, h_t:, w_t:] = img[:-h_t, :-w_t]
            elif w_t < 0:
                translated[i, h_t:, :w_t] = img[:-h_t, -w_t:]
        elif h_t < 0:
            if w_t == 0:
                translated[i, :h_t] = img[-h_t:]
            elif w_t > 0:
                translated[i, :h_t, w_t:] = img[-h_t:, :-w_t]
            elif w_t < 0:
                translated[i, :h_t, :w_t] = img[-h_t:, -w_t:]
        elif h_t == 0:
            if w_t == 0:
                translated[i] = img
            elif w_t > 0:
                translated[i, :, w_t:] = img[:, :-w_t]
            elif w_t < 0:
                translated[i, :, :w_t] = img[:, -w_t:]

    return translated


def random_cutout_color(imgs, min_cut=4, max_cut=20):
    n, h, w, c = imgs.shape
    w_cut = torch.randint(min_cut, max_cut + 1, (n, ))
    h_cut = torch.randint(min_cut, max_cut + 1, (n, ))
    fills = torch.randint(0, 255, (n, 1, 1, c))
    for img, wc, hc, fill in zip(imgs, w_cut, h_cut, fills):
        w1 = torch.randint(w - wc + 1, ())
        h1 = torch.randint(h - hc + 1, ())
        img[h1:h1 + hc, w1:w1 + wc] = fill
    return imgs


def random_cutout_color_fast(imgs, min_cut=4, max_cut=20):
    n, h, w, c = imgs.shape
    h_starts = np.random.randint(0, h - max_cut + 1, (n, ))
    h_size = np.random.randint(min_cut, max_cut + 1, (n, ))
    w_starts = np.random.randint(0, w - max_cut + 1, (n, ))
    w_size = np.random.randint(min_cut, max_cut + 1, (n, ))
    fills = torch.randint(0, 255, (n, 1, 1, c)).to(imgs.device)
    for i in range(n):
        imgs[i, h_starts[i]:h_starts[i] + h_size[i], w_starts[i]:w_starts[i] + w_size[i]] = fills[i]
    return imgs


def random_cutout(imgs, min_cut=4, max_cut=20):
    n, h, w, c = imgs.shape
    h_starts = np.random.randint(0, h - max_cut + 1, (n, ))
    h_size = np.random.randint(min_cut, max_cut + 1, (n, ))
    w_starts = np.random.randint(0, w - max_cut + 1, (n, ))
    w_size = np.random.randint(min_cut, max_cut + 1, (n, ))
    for i in range(n):
        imgs[i, h_starts[i]:h_starts[i] + h_size[i], w_starts[i]:w_starts[i] + w_size[i]] = 0
    return imgs


def random_channel_drop(imgs):
    n, h, w, c = imgs.shape
    drop_channels = np.random.randint(0, c, (n, ))
    # fills = torch.randint(0, 255, (n, 1, 1, 1)).to(imgs.device)
    for i in range(n):
        imgs[i, :, :, drop_channels[i]] = 0  # fills[i]

    # import matplotlib.pyplot as plt
    # for img in imgs:
    #     plt.imshow(img[:,:,3:].detach().cpu().numpy())
    #     plt.show()

    return imgs


def random_convolution(imgs):
    n, h, w, c = imgs.shape
    imgs = imgs.reshape(n, c, h, w)
    imgs = imgs.to(torch.float32)

    _device = imgs.device

    img_h, img_w = imgs.shape[2], imgs.shape[3]
    num_stack_channel = imgs.shape[1]
    num_batch = imgs.shape[0]
    num_trans = num_batch
    batch_size = int(num_batch / num_trans)

    # initialize random covolution
    rand_conv = nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)

    for trans_index in range(num_trans):
        torch.nn.init.xavier_normal_(rand_conv.weight.data)
        temp_imgs = imgs[trans_index * batch_size:(trans_index + 1) * batch_size]
        temp_imgs = temp_imgs.reshape(-1, 3, img_h, img_w)  # (batch x stack, channel, h, w)
        rand_out = rand_conv(temp_imgs)
        if trans_index == 0:
            total_out = rand_out
        else:
            total_out = torch.cat((total_out, rand_out), 0)
    total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)

    total_out = total_out.reshape(n, h, w, c)
    total_out = total_out.to(torch.uint8)
    return total_out


def random_flip(imgs, probability=1.0, flip_axis=1):
    """
    Args:
        flip_axis: If `1` then randomly performs an up/down flip, and `2` is a left/right flip.
    """
    assert flip_axis == 1 or flip_axis == 2
    device = imgs.device
    b, h, w, c = imgs.shape

    mask = np.random.uniform(size=(b, )) <= probability
    mask = torch.from_numpy(mask).to(torch.uint8).to(device)
    mask = mask.reshape(-1, 1).repeat((1, c)).reshape(-1, 1, 1, c)

    return mask * imgs.flip(dims=(flip_axis, )) + (1 - mask) * imgs


def random_rotation_naive(imgs):
    device = imgs.device
    b, h, w, c = imgs.shape
    ks = np.random.choice([-1, 1, 2], b)
    for i in range(b):
        imgs[i] = torch.rot90(imgs[i], k=ks[i])
    return imgs


def random_rotation_kornia(imgs):
    imgs = imgs.permute(0, 3, 1, 2)
    imgs = imgs.to(torch.float32)
    import kornia
    t = kornia.augmentation.RandomRotation(180)
    imgs = t(imgs)
    imgs = imgs.permute(0, 2, 3, 1)
    return imgs


def random_rotation(imgs):
    device = imgs.device
    b, h, w, c = imgs.shape

    ks = np.random.choice([-1, 1, 2], size=b)

    def get_masked_rotated_images(imgs, ks, k):
        imgs_rot = torch.rot90(imgs, k=k, dims=(1, 2))
        mask = ks == k
        mask = torch.from_numpy(mask).to(torch.uint8).to(imgs.device)
        mask = mask.reshape(-1, 1).repeat((1, c)).reshape(-1, 1, 1, c)
        return mask * imgs_rot

    return (get_masked_rotated_images(imgs, ks, 1) + get_masked_rotated_images(imgs, ks, -1) +
            get_masked_rotated_images(imgs, ks, 2))


def random_flip_and_rotation(imgs):
    imgs = random_rotation(imgs)
    imgs = random_flip(imgs, probability=0.5, flip_axis=1)
    imgs = random_flip(imgs, probability=0.5, flip_axis=2)
    return imgs


"""Functions that use data augmentations."""

PRIORITY_ORDERED_TRANSFORMS = [
    "random_rotation",
    "random_flip_left_right",
    "random_flip_up_down",
    "random_cutout",
    "random_cutout_color",
    "random_channel_drop",
    "random_conv",
    "random_translate",
]


def sort_transforms(ts):
    return sorted(ts, key=lambda x: PRIORITY_ORDERED_TRANSFORMS.index(x))


def apply_transform(imgs, transform, options):
    """Transforms the provided images with the requested transform.

    Returns:
        A tuple of the transformed images and a mask of same length as images
        indicating whether to apply the policy loss on the given transformed images.
    """
    policy_weight_mask = torch.ones(len(imgs), dtype=torch.float32, device=imgs.device)
    if transform == "random_translate":
        imgs = random_translate_via_index(imgs, **options.get("random_translate_options", {}))
    elif transform == "random_cutout_color":
        imgs = random_cutout_color_fast(imgs, **options.get("random_cutout_color_options", {}))
    elif transform == "random_cutout":
        imgs = random_cutout(imgs, **options.get("random_cutout_options", {}))
    elif transform == "random_channel_drop":
        imgs = random_channel_drop(imgs, **options.get("random_channel_drop_options", {}))
    elif transform == "random_flip_up_down":
        imgs = random_flip(imgs, flip_axis=1, **options.get("random_flip_options", {}))
        policy_weight_mask = torch.zeros(len(imgs), dtype=torch.float32, device=imgs.device)
    elif transform == "random_flip_left_right":
        imgs = random_flip(imgs, flip_axis=2, **options.get("random_flip_options", {}))
        policy_weight_mask = torch.zeros(len(imgs), dtype=torch.float32, device=imgs.device)
    elif transform == "random_rotation":
        imgs = random_rotation(imgs, **options.get("random_rotation_options", {}))
        policy_weight_mask = torch.zeros(len(imgs), dtype=torch.float32, device=imgs.device)
    else:
        raise NotImplementedError(f"Transform not implemented {transform}")

    return imgs, policy_weight_mask


def apply_data_augmentation_independently(imgs, options):
    num_transforms = len(options["transforms"])
    assert num_transforms > 0
    num_samples = len(imgs)
    assert num_samples > num_transforms
    num_samples_per_transform = num_samples // num_transforms
    transform_indices = np.random.permutation(num_transforms)
    policy_weight_mask = torch.ones(len(imgs), dtype=torch.float32, device=imgs.device)

    for i, transform_index in enumerate(transform_indices):
        transform = options["transforms"][transform_index]
        start = i * num_samples_per_transform
        end = start + num_samples_per_transform
        if i == num_transforms - 1:
            end = num_samples
        imgs[start:end], policy_weight_mask[start:end] = apply_transform(
            imgs[start:end], transform, options)
    return imgs, policy_weight_mask


def apply_data_augmentation_stacked(imgs, options):
    raise NotImplementedError("need to implement policy weighting for some transforms")
    assert len(options["transforms"]) > 0
    for transform in sort_transforms(options["transforms"]):
        imgs = apply_transform(imgs, transform, options)
    return imgs


def apply_data_augmentation(imgs, options):
    aug_mode = options.get("augmentation_mode", "independent")
    if aug_mode == "independent":
        return apply_data_augmentation_independently(imgs, options)
    elif aug_mode == "stacked":
        return apply_data_augmentation_stacked(imgs, options)
    else:
        raise ValueError(f"Invalid augmentation mode: {aug_mode}")


"""Some testing functions; terrible approach."""


def time_function(f, num_iters, *args):
    import time
    start = time.time()
    for itr in range(num_iters):
        sys.stdout.write(f"\r{itr + 1} / {num_iters}")
        f(*args)
    end = time.time()
    print(f"\nTook {end - start:0.4f} seconds")


def random_translate_main():
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


def random_cutout_color_main():
    import copy

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

    np.random.seed(1)
    torch.manual_seed(1)
    result_a = random_cutout_color(copy.deepcopy(obs))
    np.random.seed(1)
    torch.manual_seed(1)
    result_b = random_cutout_color_fast(copy.deepcopy(obs))
    # np.testing.assert_array_almost_equal(result_a.detach().cpu().numpy(),
    #                                      result_b.detach().cpu().numpy())

    # imgs = result_b
    # import matplotlib.pyplot as plt
    # for ob, img in zip(obs, imgs):
    #     fig, axs = plt.subplots(2, 1, figsize=(16, 16))
    #     axs[0].imshow(ob[:, :, :3].detach().cpu().numpy())
    #     axs[1].imshow(img[:, :, :3].detach().cpu().numpy())
    #     plt.show()

    time_function(random_cutout_color_fast, num_iters, obs)


def random_conv_main():
    import copy

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

    # result_a = random_convolution(copy.deepcopy(obs))

    # imgs = result_a
    # import matplotlib.pyplot as plt
    # for ob, img in zip(obs, imgs):
    #     fig, axs = plt.subplots(2, 1, figsize=(16, 16))
    #     axs[0].imshow(ob[:, :, :3].detach().cpu().to(torch.uint8).numpy())
    #     axs[1].imshow(img[:, :, :3].detach().cpu().to(torch.uint8).numpy())
    #     plt.show()

    time_function(random_convolution, num_iters, obs)


def random_channel_drop_main():
    import copy

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
    obs = torch.tensor(obs).to("cpu")

    result_a = random_channel_drop(copy.deepcopy(obs))

    imgs = result_a
    import matplotlib.pyplot as plt
    for ob, img in zip(obs, imgs):
        fig, axs = plt.subplots(2, 1, figsize=(16, 16))
        axs[0].imshow(ob[:, :, :3].detach().cpu().numpy())
        axs[1].imshow(img[:, :, :3].detach().cpu().numpy())
        plt.show()

    time_function(random_channel_drop, num_iters, obs)


def random_color_jitter(imgs):
    from torchvision.transforms import ColorJitter
    transform = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5)
    img_out = transform(imgs)
    return img_out


def random_color_jitter(imgs):
    b, c, h, w = imgs.shape
    imgs = imgs.to(torch.float32)
    imgs = imgs.view(-1, 3, h, w)
    transform_module = nn.Sequential(
        ColorJitterLayer(brightness=0.4,
                         contrast=0.4,
                         saturation=0.4,
                         hue=0.5,
                         p=1.0,
                         batch_size=b,
                         stack_size=2))
    imgs = transform_module(imgs).view(b, c, h, w)
    return imgs


def random_color_jitter_main():
    num_imgs = 2048
    height = 64
    width = 64
    channels = 6
    num_iters = 100

    obs = np.random.randint(low=0,
                            high=256,
                            size=num_imgs * height * width * channels,
                            dtype=np.uint8)
    obs = obs.reshape(num_imgs, channels, height, width)
    obs = torch.tensor(obs).to("cpu")

    # imgs = random_color_jitter(obs)

    # import matplotlib.pyplot as plt
    # for ob, img in zip(obs, imgs):
    #     fig, axs = plt.subplots(2, 1, figsize=(16, 16))
    #     axs[0].imshow(ob[:, :, :3].detach().cpu().numpy())
    #     axs[1].imshow(img[:, :, :3].detach().cpu().numpy())
    #     plt.show()

    time_function(random_color_jitter, num_iters, obs)


def random_flip_main():
    import copy

    num_imgs = 2048
    height = 64
    width = 64
    channels = 6
    num_iters = 100

    import gym
    env = gym.make("procgen:procgen-bigfish-v0")
    obs = []
    x = env.reset()
    obs.append(x)
    for i in range(1, num_imgs):
        x, _, done, _ = env.step(env.action_space.sample())
        obs.append(x)
    obs = torch.tensor(obs).to("cuda")

    # result_a = random_flip(copy.deepcopy(obs), flip_axis=2)
    # result_a = random_rotation(copy.deepcopy(obs))
    result_a = random_flip_and_rotation(copy.deepcopy(obs))

    imgs = result_a
    import matplotlib.pyplot as plt
    for ob, img in zip(obs, imgs):
        fig, axs = plt.subplots(2, 1, figsize=(16, 16))
        axs[0].imshow(ob[:, :, :3].detach().cpu().to(torch.uint8).numpy())
        axs[1].imshow(img[:, :, :3].detach().cpu().to(torch.uint8).numpy())
        plt.show()

    time_function(random_rotation, num_iters, copy.deepcopy(obs))
    time_function(random_rotation_naive, num_iters, copy.deepcopy(obs))
    time_function(random_flip, num_iters, copy.deepcopy(obs))
    time_function(random_flip_and_rotation, num_iters, copy.deepcopy(obs))


if __name__ == "__main__":
    # random_translate_main()
    # random_cutout_color_main()
    # random_color_jitter_main()
    # random_channel_drop_main()
    # random_conv_main()
    random_flip_main()
