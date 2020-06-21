from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv0 = nn.Conv2d(in_channels=channels,
#                                out_channels=channels,
#                                kernel_size=3,
#                                padding=1)
#         self.conv1 = nn.Conv2d(in_channels=channels,
#                                out_channels=channels,
#                                kernel_size=3,
#                                padding=1)

#     def forward(self, x):
#         inputs = x
#         x = nn.functional.relu(x)
#         x = self.conv0(x)
#         x = nn.functional.relu(x)
#         x = self.conv1(x)
#         return x + inputs

# class ConvSequence(nn.Module):
#     def __init__(self, input_shape, out_channels):
#         super().__init__()
#         self._input_shape = input_shape
#         self._out_channels = out_channels
#         self.conv = nn.Conv2d(in_channels=self._input_shape[0],
#                               out_channels=self._out_channels,
#                               kernel_size=3,
#                               padding=1)
#         self.res_block0 = ResidualBlock(self._out_channels)
#         self.res_block1 = ResidualBlock(self._out_channels)

#     def forward(self, x):
#         x = self.conv(x)
#         x = nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
#         x = self.res_block0(x)
#         x = self.res_block1(x)
#         assert x.shape[1:] == self.get_output_shape()
#         return x

#     def get_output_shape(self):
#         _c, h, w = self._input_shape
#         return (self._out_channels, (h + 1) // 2, (w + 1) // 2)

# class CSWM(TorchModelV2, nn.Module):
#     def __init__(self, obs_space, action_space, num_outputs, model_config, name):
#         TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
#         nn.Module.__init__(self)

#         h, w, c = obs_space.shape
#         shape = (c, h, w)

#         conv_seqs = []
#         for out_channels in [16, 32, 32]:
#             conv_seq = ConvSequence(shape, out_channels)
#             shape = conv_seq.get_output_shape()
#             conv_seqs.append(conv_seq)
#         self.conv_seqs = nn.ModuleList(conv_seqs)
#         self.hidden_fc = nn.Linear(in_features=shape[0] * shape[1] * shape[2], out_features=256)
#         self.logits_fc = nn.Linear(in_features=256, out_features=num_outputs)
#         self.value_fc = nn.Linear(in_features=256, out_features=1)

#     @override(TorchModelV2)
#     def forward(self, input_dict, state, seq_lens):
#         x = input_dict["obs"].float()
#         x = x / 255.0  # scale to 0-1
#         x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
#         for conv_seq in self.conv_seqs:
#             x = conv_seq(x)
#         x = torch.flatten(x, start_dim=1)
#         x = nn.functional.relu(x)
#         x = self.hidden_fc(x)
#         x = nn.functional.relu(x)
#         logits = self.logits_fc(x)
#         value = self.value_fc(x)
#         self._value = value.squeeze(1)
#         return logits, state

#     @override(TorchModelV2)
#     def value_function(self):
#         assert self._value is not None, "must call forward() first"
#         return self._value

# ModelCatalog.register_custom_model("cswm", CSWM)


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
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        return (self._out_channels, (h + 1) // 2, (w + 1) // 2)


class ObsEncoder(nn.Module):
    """Observation to object-mask encoder.

    Args:
        input_dim: Channels in the incoming image.
        hidden_dim: Number of channels in intermediate layers.
        num_objects: Number of objects for which to generate masks.
    """
    def __init__(self, input_dim, hidden_dim, num_objects):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, (3, 3), padding=1)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), padding=1)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.conv4 = nn.Conv2d(hidden_dim, num_objects, (3, 3), padding=1)

    def forward(self, obs):
        h = nn.functional.relu(self.ln1(self.conv1(obs)))
        h = nn.functional.relu(self.ln2(self.conv2(h)))
        h = nn.functional.relu(self.ln3(self.conv3(h)))
        return torch.sigmoid(self.conv4(h))


class MaskEncoder(nn.Module):
    """Object-mask to object-state encoder.

    Args:
        input_dim: The number of objects.
        hidden_dim: The conv number of filters.
        fc_hidden_dim: Hidden size of the fully connected layers.
        output_dim: Size of the object state / embedding.
    """
    def __init__(self, input_dim, hidden_dim, fc_hidden_dim, output_dim):
        super().__init__()

        self.cnn1 = nn.Conv2d(1, hidden_dim, (3, 3), stride=2, padding=0)
        self.ln1 = nn.BatchNorm2d(hidden_dim)

        self.cnn2 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), stride=2, padding=0)
        self.ln2 = nn.BatchNorm2d(hidden_dim)

        self.cnn3 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), stride=2, padding=0)
        self.ln3 = nn.BatchNorm2d(hidden_dim)

        self.cnn4 = nn.Conv2d(hidden_dim, hidden_dim, (3, 3), stride=2, padding=0)
        self.ln4 = nn.BatchNorm2d(hidden_dim)

        magic = 144
        self.fc1 = nn.Linear(magic, fc_hidden_dim)
        # I'm applying this layer norm differently than it was originally applied
        self.ln = nn.LayerNorm(fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)

    def forward(self, ins):
        batch_size, num_objects, height, width = ins.shape
        ins = ins.reshape(batch_size * num_objects, 1, height, width)
        h = nn.functional.relu(self.ln1(self.cnn1(ins)))
        h = nn.functional.relu(self.ln2(self.cnn2(h)))
        h = nn.functional.relu(self.ln3(self.cnn3(h)))
        h = nn.functional.relu(self.ln4(self.cnn4(h)))
        h = h.view(batch_size, num_objects, -1)
        h = nn.functional.relu(self.ln(self.fc1(h)))
        return self.fc2(h)


def unsorted_segment_sum(tensor, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, tensor.size(1))
    result = tensor.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, tensor.size(1))
    result.scatter_add_(0, segment_ids, tensor)
    return result


class GNN(torch.nn.Module):
    """???

    Args:
        input_dim: embedding_dim
        hidden_dim: hidden size of whatever
        num_objects: number of objects
    """
    def __init__(self, input_dim, hidden_dim, num_objects):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects

        self.edge_mlp = nn.Sequential(nn.Linear(input_dim * 2, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                      nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

        node_input_dim = hidden_dim + input_dim

        self.node_mlp = nn.Sequential(nn.Linear(node_input_dim, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                      nn.ReLU(), nn.Linear(hidden_dim, input_dim))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, edge_attr):
        del edge_attr  # Unused.
        out = torch.cat([source, target], dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = unsorted_segment_sum(edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, cuda):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)

            if cuda:
                self.edge_list = self.edge_list.cuda()

        return self.edge_list

    def forward(self, states):

        cuda = states.is_cuda
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.view(-1, self.input_dim)

        edge_attr = None
        edge_index = None

        if num_nodes > 1:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(batch_size, num_nodes, cuda)

            row, col = edge_index
            edge_attr = self._edge_model(node_attr[row], node_attr[col], edge_attr)

        node_attr = self._node_model(node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        return node_attr.view(batch_size, num_nodes, -1)


class CSWM(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, num_objects,
                 object_state_size):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)

        # Is this channel first or last format?
        # Seems like the original model assumes NCHW, why?
        # i.e., it flips it below; weird
        self.obs_encoder = ObsEncoder(input_dim=c, hidden_dim=16, num_objects=num_objects)
        self.mask_encoder = MaskEncoder(input_dim=num_objects,
                                        hidden_dim=16,
                                        fc_hidden_dim=128,
                                        output_dim=object_state_size)

        self.gnn = GNN(input_dim=object_state_size, hidden_dim=16, num_objects=num_objects)

        gnn_flat_output_dim = num_objects * object_state_size

        output_hidden_dim = 128
        self.hidden_fc = nn.Linear(in_features=gnn_flat_output_dim, out_features=output_hidden_dim)
        self.logits_fc = nn.Linear(in_features=output_hidden_dim, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=output_hidden_dim, out_features=1)

        self._obj_masks = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW
        self._obj_masks = self.obs_encoder(x)
        x = self.mask_encoder(self._obj_masks)
        x = self.gnn(x)
        x = torch.flatten(x, start_dim=1)
        x = nn.functional.relu(x)
        x = self.hidden_fc(x)
        x = nn.functional.relu(x)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    def object_masks(self):
        assert self._obj_masks is not None
        return self._obj_masks


ModelCatalog.register_custom_model("cswm", CSWM)
