import kornia
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

from models.noisy_nets_util import NoisyLinear


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
        x = x + inputs
        return x


class ConvSequence(nn.Module):
    def __init__(self, input_shape, out_channels, dropout_prob=0.0):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self._dropout_prob = dropout_prob

        self.conv = nn.Conv2d(in_channels=self._input_shape[0],
                              out_channels=self._out_channels,
                              kernel_size=3,
                              padding=1)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

        self.dropout_layer = nn.Dropout2d(self._dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        if self._dropout_prob > 0:
            x = self.dropout_layer(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

    def set_norm_layer_mode(self, mode):
        if mode == "train":
            if self._dropout_prob > 0:
                self.dropout_layer.train()
        elif mode == "eval":
            if self._dropout_prob > 0:
                self.dropout_layer.eval()


def initialize_parameters(params, mode):
    if mode == "default":
        pass
    elif mode == "orthogonal":
        for name, param in params:
            if len(param.shape) >= 2:
                torch.nn.init.orthogonal_(param)
            elif "bias" in name:
                param.data.fill_(0)
    else:
        raise ValueError(f"Unsupported initialization: {weight_init}")


class CustomImpalaCNN(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 num_filters=[16, 32, 32],
                 data_augmentation_options={},
                 dropout_prob=0.0,
                 optimizer_options={},
                 prev_action_mode="none",
                 fc_activation="relu",
                 fc_size=256,
                 weight_init="default",
                 intrinsic_reward_options={},
                 use_noisy_net=False,
                 noisy_rollout_length=16):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # This is a hack to make custom options accessible to the policy.
        # It must be stored on this class.
        self.data_augmentation_options = data_augmentation_options
        self.optimizer_options = optimizer_options
        self.intrinsic_reward_options = intrinsic_reward_options

        # These are actual model options.
        self.dropout_prob = dropout_prob
        self.prev_action_mode = prev_action_mode
        self.use_noisy_net = use_noisy_net
        self.noisy_rollout_length = noisy_rollout_length

        h, w, c = obs_space.shape
        shape = (c, h, w)

        conv_seqs = []
        for out_channels in num_filters:
            conv_seq = ConvSequence(shape, out_channels, dropout_prob=self.dropout_prob)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)

        hidden_fc_shape_in = shape[0] * shape[1] * shape[2]
        logits_fc_shape_in = fc_size
        value_fc_shape_in = fc_size
        if prev_action_mode == "concat":
            hidden_fc_shape_in = shape[0] * shape[1] * shape[2] + self.action_space.n
            logits_fc_shape_in = logits_fc_shape_in + self.action_space.n
            value_fc_shape_in = value_fc_shape_in + self.action_space.n

        # Store these for noisy nets usage later.
        self.logits_fc_shape_in = logits_fc_shape_in
        self.num_outputs = num_outputs

        self.hidden_fc = nn.Linear(in_features=hidden_fc_shape_in, out_features=fc_size)
        if self.use_noisy_net:
            self.logits_fc = NoisyLinear(in_size=logits_fc_shape_in, out_size=num_outputs)
        else:
            self.logits_fc = nn.Linear(in_features=logits_fc_shape_in, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=value_fc_shape_in, out_features=1)

        self.dropout_fc = nn.Dropout(self.dropout_prob)

        if fc_activation == "relu":
            self.fc_activation = nn.functional.relu
        elif fc_activation == "tanh":
            self.fc_activation = nn.functional.tanh
        else:
            raise ValueError(f"Unsupported fc activation: {fc_activation}")

        initialize_parameters(self.named_parameters(), weight_init)

        # Set externally during training.
        # There's got to be a better way to do this, but I'm not sure what it is.
        self.norm_layers_active = False
        self._detach_value_head = False

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        is_training = input_dict["is_training"]

        if not is_training and self.use_noisy_net:
            state = self._update_state(state)

        if is_training and self.norm_layers_active:
            self.set_norm_layer_mode("train")
        else:
            self.set_norm_layer_mode("eval")

        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_activation(x)

        if self.prev_action_mode == "concat":
            a = input_dict["prev_actions"]
            if isinstance(a, list):
                a = torch.tensor(np.asarray(a))
                if len(a.shape) > 1:
                    a = a.squeeze()
            a = torch.nn.functional.one_hot(a, self.action_space.n).to(x.device).to(torch.float32)
            assert len(a.shape) == len(x.shape), f"a.shape: {a.shape}, x.shape: {x.shape}"
            x = torch.cat((x, a), axis=-1)

        x = self.hidden_fc(x)
        x = self.fc_activation(x)
        x = self.dropout_fc(x)

        if self.prev_action_mode == "concat":
            x = torch.cat((x, a), axis=-1)

        if self.use_noisy_net:
            if len(state) > 0:
                # Rollout case.
                noise = state[1:]
            else:
                noise_keys = list(sorted(k for k in input_dict.keys() if "noise" in k))
                if len(noise_keys) > 0:
                    # Policy update.
                    noise = [input_dict[key] for key in noise_keys]
                else:
                    # Aux update.
                    noise = [None, None]
            logits = self.logits_fc(x, *noise)
        else:
            logits = self.logits_fc(x)

        if self._detach_value_head:
            value = self.value_fc(x.detach())
        else:
            value = self.value_fc(x)

        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def from_batch(self, train_batch, is_training=True):
        input_dict = {
            "obs": train_batch[SampleBatch.CUR_OBS],
            "is_training": is_training,
        }
        for key, value in train_batch.items():
            if "noise" in key:
                input_dict[key] = torch.tensor(value,
                                               dtype=self.logits_fc.dtype).to(self.logits_fc.device)

        if SampleBatch.PREV_ACTIONS in train_batch:
            input_dict["prev_actions"] = train_batch[SampleBatch.PREV_ACTIONS]
        if SampleBatch.PREV_REWARDS in train_batch:
            input_dict["prev_rewards"] = train_batch[SampleBatch.PREV_REWARDS]
        states = []
        i = 0
        while "state_in_{}".format(i) in train_batch:
            states.append(train_batch["state_in_{}".format(i)])
            i += 1

        return self.__call__(input_dict, states, train_batch.get("seq_lens"))

    def set_norm_layer_mode(self, mode):
        if mode == "train":
            self.dropout_fc.train()
        else:
            self.dropout_fc.eval()
        for conv_seq in self.conv_seqs:
            conv_seq.set_norm_layer_mode(mode)

    def _sample_noise(self, batch_size=1):
        device = self.logits_fc.weight.device
        in_size = self.logits_fc_shape_in
        out_size = self.num_outputs
        return sample_factored_noise(batch_size, in_size, out_size, device)

    def _update_state(self, prev_state):
        timesteps, noises = prev_state[0], prev_state[1:]
        timesteps += 1

        new_noise_indices = torch.where(timesteps % self.noisy_rollout_length == 0)[0]
        if len(new_noise_indices) > 0:
            new_noises = self.logits_fc.sample_noise(len(new_noise_indices))
            for i in range(len(noises)):
                noises[i][new_noise_indices] = new_noises[i]

        return [timesteps] + noises

    @override(TorchModelV2)
    def get_initial_state(self):
        if self.use_noisy_net:
            # This should actually return a noise vector for each fully connected layer.
            # But let's just start with one I guess.
            noises = [noise.squeeze(0) for noise in self.logits_fc.sample_noise()]
            return [torch.tensor(0)] + noises
        else:
            return []

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    def detach_value_head(self):
        self._detach_value_head = True

    def attach_value_head(self):
        self._detach_value_head = False


class RandomCrop(nn.Module):
    def __init__(self, pad, crop_size):
        super().__init__()
        self.transform = nn.Sequential(
            nn.ZeroPad2d(pad),
            kornia.augmentation.RandomCrop(crop_size),
        )

    def forward(self, x):
        x = self.transform(x)
        return x


class SplitImpalaCNN(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, common_filters,
                 policy_filters, value_filters, value_options):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.data_augmentation_options = {"mode": "none", "mode_options": {}}

        h, w, c = obs_space.shape
        shape = (c, h, w)

        common_conv_seqs = []
        for out_channels in common_filters:
            conv_seq = ConvSequence(shape, out_channels)
            shape = conv_seq.get_output_shape()
            common_conv_seqs.append(conv_seq)
        self.common_conv_seqs = nn.ModuleList(common_conv_seqs)

        policy_conv_seqs = []
        policy_conv_shape = shape
        for out_channels in policy_filters:
            conv_seq = ConvSequence(policy_conv_shape, out_channels)
            policy_conv_shape = conv_seq.get_output_shape()
            policy_conv_seqs.append(conv_seq)
        self.policy_conv_seqs = nn.ModuleList(policy_conv_seqs)

        if value_options.get("random_crop", False):
            self.random_crop = RandomCrop(pad=1, crop_size=shape[1:])
        else:
            self.random_crop = nn.Identity()

        value_conv_seqs = []
        value_conv_shape = shape
        for out_channels in value_filters:
            conv_seq = ConvSequence(value_conv_shape, out_channels)
            value_conv_shape = conv_seq.get_output_shape()
            value_conv_seqs.append(conv_seq)
        self.value_conv_seqs = nn.ModuleList(value_conv_seqs)

        self.policy_hidden_fc = nn.Linear(in_features=policy_conv_shape[0] * policy_conv_shape[1] *
                                          policy_conv_shape[2],
                                          out_features=256)
        self.value_hidden_fc = nn.Linear(in_features=value_conv_shape[0] * value_conv_shape[1] *
                                         value_conv_shape[2],
                                         out_features=256)
        self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=256, out_features=1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW

        in_rollout = self.in_rollout(x)
        if in_rollout:
            self.eval()
        # self.print_memory_info("before start: ", in_rollout)

        for conv_seq in self.common_conv_seqs:
            x = conv_seq(x)

        # Policy.
        # self.print_memory_info("before policy conv: ", in_rollout)
        px = x
        for conv_seq in self.policy_conv_seqs:
            px = conv_seq(px)
        # self.print_memory_info("before policy fc: ", in_rollout)
        px = torch.flatten(px, start_dim=1)
        px = nn.functional.relu(px)
        px = self.policy_hidden_fc(px)
        px = nn.functional.relu(px)
        logits = self.logits_fc(px)

        # Value.
        # self.print_memory_info("before value conv: ", in_rollout)
        vx = x
        if not in_rollout:
            vx = self.random_crop(vx)
        # self._plot(x, vx)
        for conv_seq in self.value_conv_seqs:
            vx = conv_seq(vx)
        # self.print_memory_info("before value fc: ", in_rollout)
        vx = torch.flatten(vx, start_dim=1)
        vx = nn.functional.relu(vx)
        vx = self.value_hidden_fc(vx)
        vx = nn.functional.relu(vx)
        value = self.value_fc(vx)
        self._value = value.squeeze(1)
        # self.print_memory_info("end: ", in_rollout)
        if in_rollout:
            self.train()
        return logits, state

    def print_memory_info(self, prefix="", in_rollout=False):
        if in_rollout:
            return

        if not hasattr(self, "prev_mem"):
            self.prev_mem = 0
        self.cur_mem = torch.cuda.memory_allocated()
        change = self.cur_mem - self.prev_mem
        self.prev_mem = self.cur_mem
        string = prefix + f" current: {self.cur_mem / 10e9:0.2f}GB, change: {change / 10e9:0.2f}GB"
        print(string)

    def in_rollout(self, x, threshold=300):
        # Another hack.
        # Need to know if the model is being used for rollouts or for training.
        # The training state isn't set by the rllib (or is set to training=True on purpose),
        # so assume that if the batch size is less than some threshold that we're rolling.
        return len(x) < threshold

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    def _plot(self, x, crop_x):
        import matplotlib.pyplot as plt
        for i in range(2):
            fig, axs = plt.subplots(2, 2)
            fig.suptitle(f"sample {i}")
            axs[0, 0].imshow(x[i, 0, :, :].detach().cpu().numpy())
            axs[0, 0].set_title("orig channel 0")
            axs[0, 1].imshow(x[i, 1, :, :].detach().cpu().numpy())
            axs[0, 1].set_title("orig channel 1")

            axs[1, 0].imshow(crop_x[i, 0, :, :].detach().cpu().numpy())
            axs[1, 0].set_title("crop channel 0")
            axs[1, 1].imshow(crop_x[i, 1, :, :].detach().cpu().numpy())
            axs[1, 1].set_title("crop channel 1")
            plt.tight_layout()
            plt.show()


ModelCatalog.register_custom_model("custom_impala_cnn_torch", CustomImpalaCNN)
ModelCatalog.register_custom_model("split_impala_cnn_torch", SplitImpalaCNN)
