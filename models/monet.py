from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils import try_import_torch

torch, nn = try_import_torch()


class Flatten(nn.Module):
    def forward(self, x):
        return x.flatten(start_dim=1)


class ComponentVAE(nn.Module):
    def __init__(self, input_nc, z_dim=16, full_res=False):
        super().__init__()
        self._input_nc = input_nc
        self._z_dim = z_dim
        # full_res = False # full res: 128x128, low res: 64x64
        h_dim = 4096 if full_res else 1024
        self.encoder = nn.Sequential(nn.Conv2d(input_nc + 1, 32, 3, stride=2, padding=1),
                                     nn.ReLU(True), nn.Conv2d(32, 32, 3, stride=2, padding=1),
                                     nn.ReLU(True), nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                     nn.ReLU(True), nn.Conv2d(64, 64, 3, stride=2, padding=1),
                                     nn.ReLU(True), Flatten(), nn.Linear(h_dim, 256), nn.ReLU(True),
                                     nn.Linear(256, 32))
        self.decoder = nn.Sequential(
            nn.Conv2d(z_dim + 2, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(True),
            nn.Conv2d(32, input_nc + 1, 1),
        )
        self._bg_logvar = 2 * torch.tensor(0.09).log()
        self._fg_logvar = 2 * torch.tensor(0.11).log()

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    @staticmethod
    def spatial_broadcast(z, h, w):
        # Batch size
        n = z.shape[0]
        # Expand spatially: (n, z_dim) -> (n, z_dim, h, w)
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)
        # Coordinate axes:
        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)
        # Expand from (h, w) -> (n, 1, h, w)
        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)
        # Concatenate along the channel dimension: final shape = (n, z_dim + 2, h, w)
        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, x, log_m_k, background=False, decode=True):
        """
        :param x: Input image
        :param log_m_k: Attention mask logits
        :return: x_k and reconstructed mask logits
        """
        params = self.encoder(torch.cat((x, log_m_k), dim=1))
        z_mu = params[:, :self._z_dim]
        z_logvar = params[:, self._z_dim:]
        z = self.reparameterize(z_mu, z_logvar)

        # "The height and width of the input to this CNN were both 8 larger than the target output (i.e. image) size
        #  to arrive at the target size (i.e. accommodating for the lack of padding)."
        h, w = x.shape[-2:]
        z_sb = self.spatial_broadcast(z, h + 8, w + 8)

        if decode:
            output = self.decoder(z_sb)
            x_mu = output[:, :self._input_nc]
            x_logvar = self._bg_logvar if background else self._fg_logvar
            m_logits = output[:, self._input_nc:]
        else:
            m_logits, x_mu, x_logvar = None, None, None

        return m_logits, x_mu, x_logvar, z_mu, z_logvar


class AttentionBlock(nn.Module):
    def __init__(self, input_nc, output_nc, resize=True):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(output_nc, affine=True)
        self._resize = resize

    def forward(self, *inputs):
        downsampling = len(inputs) == 1
        x = inputs[0] if downsampling else torch.cat(inputs, dim=1)
        x = self.conv(x)
        x = self.norm(x)
        x = skip = nn.functional.relu(x)
        if self._resize:
            x = nn.functional.interpolate(skip,
                                          scale_factor=0.5 if downsampling else 2.,
                                          mode='nearest')
        return (x, skip) if downsampling else x


class Attention(nn.Module):
    """Create a Unet-based generator"""
    def __init__(self, input_nc, output_nc, ngf=64):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Attention, self).__init__()
        self.downblock1 = AttentionBlock(input_nc + 1, ngf)
        self.downblock2 = AttentionBlock(ngf, ngf * 2)
        self.downblock3 = AttentionBlock(ngf * 2, ngf * 4)
        self.downblock4 = AttentionBlock(ngf * 4, ngf * 8)
        self.downblock5 = AttentionBlock(ngf * 8, ngf * 8, resize=False)
        # no resizing occurs in the last block of each path
        # self.downblock6 = AttentionBlock(ngf * 8, ngf * 8, resize=False)

        self.mlp = nn.Sequential(
            nn.Linear(4 * 4 * ngf * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4 * 4 * ngf * 8),
            nn.ReLU(),
        )

        # self.upblock1 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock2 = AttentionBlock(2 * ngf * 8, ngf * 8)
        self.upblock3 = AttentionBlock(2 * ngf * 8, ngf * 4)
        self.upblock4 = AttentionBlock(2 * ngf * 4, ngf * 2)
        self.upblock5 = AttentionBlock(2 * ngf * 2, ngf)
        # no resizing occurs in the last block of each path
        self.upblock6 = AttentionBlock(2 * ngf, ngf, resize=False)

        self.output = nn.Conv2d(ngf, output_nc, 1)

    def forward(self, x, log_s_k):
        # Downsampling blocks
        x, skip1 = self.downblock1(torch.cat((x, log_s_k), dim=1))
        x, skip2 = self.downblock2(x)
        x, skip3 = self.downblock3(x)
        x, skip4 = self.downblock4(x)
        x, skip5 = self.downblock5(x)
        skip6 = skip5
        # The input to the MLP is the last skip tensor collected from the downsampling path (after flattening)
        # _, skip6 = self.downblock6(x)
        # Flatten
        x = skip6.flatten(start_dim=1)
        x = self.mlp(x)
        # Reshape to match shape of last skip tensor
        x = x.view(skip6.shape)
        # Upsampling blocks
        # x = self.upblock1(x, skip6)
        x = self.upblock2(x, skip5)
        x = self.upblock3(x, skip4)
        x = self.upblock4(x, skip3)
        x = self.upblock5(x, skip2)
        x = self.upblock6(x, skip1)
        # Output layer
        x = self.output(x)
        x = nn.functional.logsigmoid(x)
        return x


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


class MONetModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, num_objects,
                 object_state_size):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        h, w, c = obs_space.shape
        shape = (c, h, w)
        z_dim = object_state_size
        self.num_objects = num_objects

        self.netAttn = Attention(c, 1)
        self.netCVAE = ComponentVAE(c, z_dim)
        self.eps = torch.finfo(torch.float).eps
        self.criterionKL = nn.KLDivLoss(reduction="batchmean")

        self.gnn = GNN(input_dim=object_state_size, hidden_dim=16, num_objects=num_objects)

        gnn_flat_output_dim = num_objects * object_state_size
        output_hidden_dim = 128
        self.hidden_fc = nn.Linear(in_features=gnn_flat_output_dim, out_features=output_hidden_dim)
        self.logits_fc = nn.Linear(in_features=output_hidden_dim, out_features=num_outputs)
        self.value_fc = nn.Linear(in_features=output_hidden_dim, out_features=1)

        self.loss_E = None
        self.x_tilde = None
        self.b = None
        self.m = None
        self.m_tilde_logits = None

        self.enc_weight = 0.5
        self.mask_weight = 0.5
        self.policy_loss_weight = 1.0

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        x = x / 255.0  # scale to 0-1
        x = x.permute(0, 3, 1, 2)  # NHWC => NCHW

        decode = True or input_dict["is_training"]

        self.loss_E = 0
        self.x_tilde = 0
        b = []
        m = []
        m_tilde_logits = []

        # Initial s_k = 1: shape = (N, 1, H, W)
        shape = list(x.shape)
        shape[1] = 1
        log_s_k = x.new_zeros(shape)
        z_mu_ks = []
        for k in range(self.num_objects):
            # Derive mask from current scope
            if k != self.num_objects - 1:
                log_alpha_k = self.netAttn(x, log_s_k)
                log_m_k = log_s_k + log_alpha_k
                # Compute next scope
                log_s_k += (1. - log_alpha_k.exp()).clamp(min=self.eps).log()
            else:
                log_m_k = log_s_k

            # Get component and mask reconstruction, as well as the z_k parameters
            m_tilde_k_logits, x_mu_k, x_logvar_k, z_mu_k, z_logvar_k = self.netCVAE(x,
                                                                                    log_m_k,
                                                                                    k == 0,
                                                                                    decode=decode)

            z_mu_ks.append(z_mu_k)

            if decode:
                # KLD is additive for independent distributions
                self.loss_E += -0.5 * (1 + z_logvar_k - z_mu_k.pow(2) - z_logvar_k.exp()).sum()

                m_k = log_m_k.exp()
                x_k_masked = m_k * x_mu_k

                # Exponents for the decoder loss
                b_k = log_m_k - 0.5 * x_logvar_k - (x - x_mu_k).pow(2) / (2 * x_logvar_k.exp())
                b.append(b_k.unsqueeze(1))

                # Get outputs for kth step
                setattr(self, 'm{}'.format(k), m_k * 2. - 1.)  # shift mask from [0, 1] to [-1, 1]
                setattr(self, 'x{}'.format(k), x_mu_k)
                setattr(self, 'xm{}'.format(k), x_k_masked)

                # Iteratively reconstruct the output image
                self.x_tilde += x_k_masked
                # Accumulate
                m.append(m_k)
                m_tilde_logits.append(m_tilde_k_logits)

        if decode:
            self.b = torch.cat(b, dim=1)
            self.m = torch.cat(m, dim=1)
            self.m_tilde_logits = torch.cat(m_tilde_logits, dim=1)

        z_mus = torch.cat(z_mu_ks, dim=1)
        x = self.hidden_fc(z_mus)
        logits = self.logits_fc(x)
        value = self.value_fc(x)
        self._value = value.squeeze(1)
        return logits, state

    @override(TorchModelV2)
    def value_function(self):
        assert self._value is not None, "must call forward() first"
        return self._value

    def object_masks(self):
        assert self.m is not None
        return self.m

    def custom_loss(self, policy_loss):
        n, num_objects, h, w = self.m_tilde_logits.shape
        mean_size = n * num_objects * h * w
        self.loss_E /= mean_size
        self.loss_D = -torch.logsumexp(self.b, dim=1).sum() / mean_size
        self.loss_mask = self.criterionKL(self.m_tilde_logits.log_softmax(dim=1),
                                          self.m) / (num_objects * h * w)
        self.self_supervised_loss = self.loss_D + self.enc_weight * self.loss_E + self.mask_weight * self.loss_mask
        return self.self_supervised_loss + policy_loss * self.policy_loss_weight


ModelCatalog.register_custom_model("monet", MONetModel)
