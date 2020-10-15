import numpy as np
import torch
import torch.nn as nn

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.recurrent_torch_model import RecurrentTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.annotations import override

from models.custom_impala_cnn_torch import ConvSequence


class CustomImpalaCNNRNN(RecurrentTorchModel, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 num_filters,
                 dropout_prob=0.0,
                 policy_fc_size=256,
                 value_fc_size=256,
                 prev_action_mode="none",
                 weight_init="default",
                 data_augmentation_options={},
                 optimizer_options={},
                 intrinsic_reward_options={}):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # These options shouldn't be stored on the model but I'm not sure how else to access them.
        self.data_augmentation_options = data_augmentation_options
        self.optimizer_options = optimizer_options
        self.intrinsic_reward_options = intrinsic_reward_options
        # End unrelated options.

        self.dropout_prob = dropout_prob
        self.prev_action_mode = prev_action_mode
        self.rnn_hidden_dim = model_config["lstm_cell_size"]

        h, w, c = obs_space.shape
        shape = (c, h, w)

        # Conv feature extractor.
        conv_seqs = []
        for out_channels in num_filters:
            conv_seq = ConvSequence(shape,
                                    out_channels,
                                    dropout_prob=dropout_prob,
                                    batch_norm=False)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.conv_seqs = nn.ModuleList(conv_seqs)

        # Reduce size to rnn hidden dim.
        fc_in_shape_in = shape[0] * shape[1] * shape[2]
        self.fc_in = nn.Linear(in_features=fc_in_shape_in, out_features=self.rnn_hidden_dim)

        # Potentially concat the action taken previously.
        gru_cell_shape_in = self.rnn_hidden_dim
        if prev_action_mode == "concat":
            gru_cell_shape_in += self.action_space.n

        # Define the actual rnn cell.
        # self.rnn = nn.GRUCell(gru_cell_shape_in, self.rnn_hidden_dim)
        self.rnn = nn.GRU(gru_cell_shape_in, self.rnn_hidden_dim, batch_first=True)

        # Initialize the weights if applicable.
        if weight_init == "default":
            pass
        elif weight_init == "orthogonal":
            nn.init.orthogonal_(self.rnn.weight_hh_l0)
        else:
            raise ValueError(f"Unsupported weight initialization method: {weight_init}")

        # Potentially concat the action taken previously.
        logits_fc_shape_in = policy_fc_size
        value_fc_shape_in = value_fc_size
        if prev_action_mode == "concat":
            logits_fc_shape_in += self.action_space.n
            value_fc_shape_in += self.action_space.n

        # Define the output layers.
        self.logits_fc = nn.Linear(in_features=logits_fc_shape_in, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=value_fc_shape_in, out_features=1)

        # Dropout for the fully connected layers.
        self.dropout_fc = nn.Dropout(dropout_prob)

        # Variables that will be defined during execution.
        self._cur_value = None
        # This variable used to deactivate norm layers during training.
        self.norm_layers_active = False

    @override(TorchModelV2)
    def get_initial_state(self):
        h = [self.fc_in.weight.new(1, self.rnn_hidden_dim).zero_().squeeze(0)]
        return h

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value

    @override(RecurrentTorchModel)
    def forward(self, input_dict, state, seq_lens):
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        output, new_state = self.forward_rnn(
            add_time_dimension(input_dict["obs"].float(), seq_lens, framework="torch"),
            input_dict["prev_actions"], state, seq_lens)
        return torch.reshape(output, [-1, self.num_outputs]), new_state

    @override(RecurrentTorchModel)
    def forward_rnn(self, inputs, actions, state, seq_lens):
        # import matplotlib.pyplot as plt
        # img = inputs[0, 0].int().detach().cpu().numpy()
        # plt.imshow(img)
        # plt.show()

        # Permute to expect torch format.
        x = inputs
        b, t, h, w, c = x.shape
        x = x / 255.0
        x = x.permute(0, 1, 4, 2, 3)  # NTHWC => NTCHW

        # Turn of the norm layers during the rollout or when explicitly set to off.
        in_rollout = self._in_rollout(x)
        if self._in_rollout(x) or not self.norm_layers_active:
            self.set_norm_layer_mode("eval")
        else:
            self.set_norm_layer_mode("train")

        # Run the cnn.
        # Flatten time dimension into the batch in order to apply the CNN.
        x = torch.reshape(x, (-1, c, h, w))
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.dropout_fc(x)
        # Reduce dimension before rnn.
        x = self.fc_in(x)
        x = nn.functional.relu(x)
        x = self.dropout_fc(x)

        # Optionally add the action before the rnn.
        if self.prev_action_mode == "concat":
            a = actions
            if isinstance(a, list):
                a = torch.tensor(np.asarray(a))
                if len(a.shape) > 1:
                    a = a.squeeze()
            a = torch.nn.functional.one_hot(a.long(), self.action_space.n).to(x.device).to(x.dtype)
            assert len(a.shape) == len(x.shape), f"a.shape: {a.shape}, x.shape: {x.shape}"
            x = torch.cat((x, a), axis=-1)

        # Run the rnn.
        # First dim of hidden state is number of rnn layers * number of directions.
        h_in = state[0].view(1, b, self.rnn_hidden_dim)
        # Reshape to include the time dimension.
        x = x.view(b, t, -1)
        x, h = self.rnn(x, h_in)
        x = self.dropout_fc(x)

        # Remove the num rnn layers * num directions dimension from the hidden state.
        h = h.view(b, -1)
        # Merge the time dimension into the batch for returning the output.
        x = x.reshape(b * t, -1)

        # Optionally add the action after the rnn as well.
        if self.prev_action_mode == "concat":
            x = torch.cat((x, a), axis=-1)

        # Generate the model output.
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._cur_value = value.squeeze(-1)
        return logits, [h]

    def set_norm_layer_mode(self, mode):
        if mode == "train":
            self.dropout_fc.train()
        else:
            self.dropout_fc.eval()
        for conv_seq in self.conv_seqs:
            conv_seq.set_norm_layer_mode(mode)

    def _in_rollout(self, x):
        # Single timestep indicates rollout.
        return x.shape[1] == 1


ModelCatalog.register_custom_model("custom_impala_cnn_rnn_torch", CustomImpalaCNNRNN)
