from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=3,
                              padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape(
        ), f"x.shape[1:] {x.shape[1:]} self.get_output_shape() {self.get_output_shape()}"
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class EpisodeDiscriminator(nn.Module):
    def __init__(self, input_size, fe_hidden_sizes=[128], cls_hidden_sizes=[128, 64]):
        super().__init__()
        assert len(fe_hidden_sizes) > 0
        assert len(cls_hidden_sizes) > 0
        layers = []
        for size in fe_hidden_sizes:
            layers.append(
                SlimFC(in_size=input_size,
                       out_size=size,
                       initializer=normc_initializer(1.0),
                       activation_fn=nn.ReLU))
            input_size = size
        self.feature_extractor = nn.Sequential(*layers)

        input_size = fe_hidden_sizes[-1] * 2  # Concatenate the features from the two samples.
        layers = []
        for size in cls_hidden_sizes:
            layers.append(
                SlimFC(in_size=input_size,
                       out_size=size,
                       initializer=normc_initializer(1.0),
                       activation_fn=nn.ReLU))
            input_size = size
        layers.append(SlimFC(in_size=input_size, out_size=1, initializer=normc_initializer(1.0)))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x_1, x_2):
        x_1 = self.feature_extractor(x_1)
        x_2 = self.feature_extractor(x_2)
        x = torch.cat((x_1, x_2), dim=1)
        return self.classifier(x)


class ReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class EpisodeAdversarialModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name,
                 discriminator_weight, l2_weight, late_fusion):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        if late_fusion:
            self.obs_channels = 3
            self.frame_diff_channels = c - self.obs_channels
            self.initial_frame_diff = ConvSequence((self.frame_diff_channels, h, w), 8)
            self.initial_obs = ConvSequence((self.obs_channels, h, w), 16)
            shape = list(self.initial_obs.get_output_shape())
            shape[0] = 8 + 16
            shape = tuple(shape)
            out_channels_list = [32, 32]
        else:
            out_channels_list = [16, 32, 32]

        conv_seqs = []
        for out_channels in out_channels_list:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)
        self.hidden_fc_1 = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
        self.hidden_fc_2 = nn.Linear(in_features=256, out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

        self.discriminator = EpisodeDiscriminator(input_size=256)
        self.discriminator_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")
        self.discriminator_weight = discriminator_weight
        self.l2_weight = l2_weight

        self.late_fusion = late_fusion
        # self.t = 0

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW

        if self.late_fusion:
            x_frame_diff = self.initial_frame_diff(x[:, :self.frame_diff_channels, :, :])
            x_obs = self.initial_obs(x[:, self.frame_diff_channels:, :, :])
            x = torch.cat((x_frame_diff, x_obs), dim=1)

            # frame_diff = x[0, :3, :, :].detach().cpu().numpy()
            # obs_orig = x[0, 3:, :, :].detach().cpu().numpy()

            # self.t += 1
            # if (self.t + 1) % 1000 == 0:
            #     import matplotlib.pyplot as plt
            #     x_fd_d = x_frame_diff.detach().cpu().numpy()
            #     fig, axs = plt.subplots(2, 5, figsize=(16, 8))
            #     axs[0][0].imshow(obs_orig.transpose(1, 2, 0))
            #     frame_diff = frame_diff.transpose(1, 2, 0)
            #     frame_diff += 127 / 255
            #     axs[0][1].imshow(frame_diff)
            #     for i in range(x_fd_d.shape[1]):
            #         row = (i + 2) // 5
            #         col = (i + 2) % 5
            #         axs[row][col].imshow(x_fd_d[0, i, :, :])
            #     plt.show()

        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc_1(x)
        x = nn.functional.relu(x)
        self.features = x = self.hidden_fc_2(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    @override(TorchModelV2)
    def metrics(self):
        stats = dict()
        stats["l2_loss"] = self.l2_loss
        stats["discriminator"] = {
            "loss": float(self.discriminator_loss.detach().cpu()),
            "accuracy": self.accuracy,
            "avg_prc": self.avg_prc
        }
        return stats

    def l2_loss_fn(self):
        value = 0
        for name, param in self.named_parameters():
            if name.endswith(".weight"):
                value += torch.square(param).sum()
        return value

    def custom_loss(self, policy_loss, episode_ids):
        batch_size = self.features.shape[0]
        perm = torch.randperm(batch_size)
        alpha = 1.0

        # The logits are the input to a sigmoid which would indicate:
        # "1 = from the same episode" or "0 = different episode".
        reverse_grad_features = ReverseLayerF.apply(self.features, alpha)
        perm_features = self.features[perm]
        reverse_grad_perm_features = ReverseLayerF.apply(perm_features, alpha)
        logits = self.discriminator(reverse_grad_features, reverse_grad_perm_features)
        targets = (episode_ids == episode_ids[perm]).view(-1, 1).to(torch.float32)

        # Discriminator loss.
        self.discriminator_loss = self.discriminator_loss_fn(logits,
                                                             targets) * self.discriminator_weight

        # Discriminator metrics.
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()
        self.accuracy = ((probs > 0.5) == targets.astype(bool)).sum() / batch_size

        try:
            import sklearn.metrics
            self.avg_prc = sklearn.metrics.average_precision_score(targets, probs)
        except:
            self.avg_prc = 0

        self.l2_loss = self.l2_loss_fn() * self.l2_weight

        return policy_loss + self.discriminator_loss + self.l2_loss


ModelCatalog.register_custom_model("episode_adversarial", EpisodeAdversarialModel)
