import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
from rlkit.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from rlkit.torch.networks import Mlp, CNN
from rlkit.torch.networks.basic import MultiInputSequential
from rlkit.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)
from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy,
    PolicyFromDistributionGenerator,
    MakeDeterministic,
)
from rlkit.pythonplusplus import identity

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# TODO: deprecate classes below in favor for PolicyFromDistributionModule


class TanhGaussianPolicyAdapter(TorchStochasticPolicy):
    """
    Usage:

    ```
    obs_processor = ...
    policy = TanhGaussianPolicyAdapter(obs_processor)
    ```
    """

    def __init__(
            self,
            obs_processor,
            obs_processor_output_dim,
            action_dim,
            hidden_sizes,
    ):
        super().__init__()
        self.obs_processor = obs_processor
        self.obs_processor_output_dim = obs_processor_output_dim
        self.mean_and_log_std_net = Mlp(
            hidden_sizes=hidden_sizes,
            output_size=action_dim * 2,
            input_size=obs_processor_output_dim,
        )
        self.action_dim = action_dim

    def forward(self, obs):
        h = self.obs_processor(obs)
        h = self.mean_and_log_std_net(h)
        mean, log_std = torch.split(h, self.action_dim, dim=1)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        tanh_normal = TanhNormal(mean, std)
        return tanh_normal


# noinspection PyMethodOverriding
class TanhGaussianPolicy(Mlp, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return TanhNormal(mean, std)

    def logprob(self, action, mean, std):
        tanh_normal = TanhNormal(mean, std)
        log_prob = tanh_normal.log_prob(
            action,
        )
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return log_prob


class GaussianPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)



class GaussianCNNPolicy(CNN, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            output_size=action_dim,
            init_w=init_w,
            output_activation=torch.tanh,
            **kwargs
        )
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            if self.std_architecture == "shared":
                last_hidden_size = obs_dim
                if len(hidden_sizes) > 0:
                    last_hidden_size = hidden_sizes[-1]
                self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = super().forward(obs, return_last_activations=True)
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(np.array([self.std, ])).float().to(
                ptu.device)

        return MultivariateDiagonalNormal(mean, std)


class GaussianMixturePolicy(Mlp, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            num_gaussians=1,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * num_gaussians,
            init_w=init_w,
            # output_activation=torch.tanh,
            **kwargs
        )
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]

            if self.std_architecture == "shared":
                self.last_fc_log_std = nn.Linear(last_hidden_size,
                                                 action_dim * num_gaussians)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim * num_gaussians, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        self.last_fc_weights = nn.Linear(last_hidden_size, num_gaussians)
        self.last_fc_weights.weight.data.uniform_(-init_w, init_w)
        self.last_fc_weights.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        preactivation = self.last_fc(h)
        mean = self.output_activation(preactivation)
        if self.std is None:
            # log_std = self.last_fc_log_std(h)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            # log_std = torch.sigmoid(self.last_fc_log_std(h))
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(self.std)
            log_std = self.log_std

        weights = F.softmax(self.last_fc_weights(h)).reshape(
            (-1, self.num_gaussians, 1))
        mixture_means = mean.reshape((-1, self.action_dim, self.num_gaussians,))
        mixture_stds = std.reshape((-1, self.action_dim, self.num_gaussians,))
        dist = GaussianMixture(mixture_means, mixture_stds, weights)
        return dist


class BinnedGMMPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            min_log_std=None,
            max_log_std=None,
            num_gaussians=1,
            std_architecture="shared",
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim * num_gaussians,
            init_w=init_w,
            # output_activation=torch.tanh,
            **kwargs
        )
        self.action_dim = action_dim
        self.num_gaussians = num_gaussians
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.log_std = None
        self.std = std
        self.std_architecture = std_architecture
        if std is None:
            last_hidden_size = obs_dim
            if len(hidden_sizes) > 0:
                last_hidden_size = hidden_sizes[-1]

            if self.std_architecture == "shared":
                self.last_fc_log_std = nn.Linear(last_hidden_size,
                                                 action_dim * num_gaussians)
                self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
                self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
            elif self.std_architecture == "values":
                self.log_std_logits = nn.Parameter(
                    ptu.zeros(action_dim * num_gaussians, requires_grad=True))
            else:
                raise ValueError(self.std_architecture)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX
        self.last_fc_weights = nn.Linear(last_hidden_size,
                                         action_dim * num_gaussians)
        self.last_fc_weights.weight.data.uniform_(-init_w, init_w)
        self.last_fc_weights.bias.data.uniform_(-init_w, init_w)

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        # preactivation = self.last_fc(h)
        # mean = self.output_activation(preactivation)
        if self.std is None:
            # log_std = self.last_fc_log_std(h)
            # log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            # log_std = torch.sigmoid(self.last_fc_log_std(h))
            if self.std_architecture == "shared":
                log_std = torch.sigmoid(self.last_fc_log_std(h))
            elif self.std_architecture == "values":
                log_std = torch.sigmoid(self.log_std_logits)
            else:
                raise ValueError(self.std_architecture)
            log_std = self.min_log_std + log_std * (
                        self.max_log_std - self.min_log_std)
            std = torch.exp(log_std)
        else:
            std = torch.from_numpy(self.std)
            log_std = self.log_std
        batch_size = len(obs)
        logits = self.last_fc_weights(h).reshape(
            (-1, self.action_dim, self.num_gaussians,))
        weights = F.softmax(logits, dim=2)
        linspace = np.tile(np.linspace(-1, 1, self.num_gaussians),
                           (batch_size, self.action_dim, 1))
        mixture_means = ptu.from_numpy(linspace)
        mixture_stds = std.reshape((-1, self.action_dim, self.num_gaussians,))
        dist = GaussianMixtureFull(mixture_means, mixture_stds, weights)
        return dist


class TanhGaussianObsProcessorPolicy(TanhGaussianPolicy):
    def __init__(self, obs_processor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pre_obs_dim = obs_processor.input_size
        self.pre_goal_dim = obs_processor.input_size
        self.obs_processor = obs_processor

    def forward(self, obs, *args, **kwargs):
        obs_and_goal = obs
        assert obs_and_goal.shape[1] == self.pre_obs_dim + self.pre_goal_dim
        obs = obs_and_goal[:, :self.pre_obs_dim]
        goal = obs_and_goal[:, self.pre_obs_dim:]

        h_obs = self.obs_processor(obs)
        h_goal = self.obs_processor(goal)

        flat_inputs = torch.cat((h_obs, h_goal), dim=1)
        return super().forward(flat_inputs, *args, **kwargs)


# noinspection PyMethodOverriding
class TanhCNNGaussianPolicy(CNN, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            std=None,
            init_w=1e-3,
            **kwargs
    ):
        super().__init__(
            init_w=init_w,
            **kwargs
        )
        obs_dim = self.input_width * self.input_height
        action_dim = self.output_size
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(self.hidden_sizes) > 0:
                last_hidden_size = self.hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        h = super().forward(obs, return_last_activations=True)

        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std

        tanh_normal = TanhNormal(mean, std)
        return tanh_normal


class MonsterTanhCNNGaussianPolicy(nn.Module):
    """
    Usage:

    ```
    policy = TanhGaussianPolicy(...)
    """

    def __init__(
            self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            std=None,
            init_w=1e-3,
            hidden_sizes=None,
            added_fc_input_size=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            output_conv_channels=False,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None
    ):
        self.deterministic = False
        if hidden_sizes is None:
            hidden_sizes = []
        assert len(kernel_sizes) == \
               len(n_channels) == \
               len(strides) == \
               len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)
        super().__init__()

        self.hidden_sizes = hidden_sizes
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.output_conv_channels = output_conv_channels
        self.pool_type = pool_type

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.fc_norm_layers = nn.ModuleList()

        for i, (out_channels, kernel_size, stride, padding) in enumerate(
                zip(n_channels, kernel_sizes, strides, paddings)
        ):
            conv = nn.Conv2d(input_channels,
                             out_channels,
                             kernel_size,
                             stride=stride,
                             padding=padding)
            hidden_init(conv.weight)
            conv.bias.data.fill_(0)

            conv_layer = conv
            self.conv_layers.append(conv_layer)
            input_channels = out_channels

            if pool_type == 'max2d':
                self.pool_layers.append(
                    nn.MaxPool2d(
                        kernel_size=pool_sizes[i],
                        stride=pool_strides[i],
                        padding=pool_paddings[i],
                    )
                )

        # use torch rather than ptu because initially the model is on CPU
        test_mat = torch.zeros(
            1,
            self.input_channels,
            self.input_width,
            self.input_height,
        )
        # find output dim of conv_layers by trial and add norm conv layers
        for i, conv_layer in enumerate(self.conv_layers):
            test_mat = conv_layer(test_mat)
            if self.conv_normalization_type == 'batch':
                self.conv_norm_layers.append(nn.BatchNorm2d(test_mat.shape[1]))
            if self.conv_normalization_type == 'layer':
                self.conv_norm_layers.append(nn.LayerNorm(test_mat.shape[1:]))
            if self.pool_type != 'none':
                test_mat = self.pool_layers[i](test_mat)

        self.conv_output_flat_size = int(np.prod(test_mat.shape))
        if self.output_conv_channels:
            self.last_fc = None
        else:
            fc_input_size = self.conv_output_flat_size
            # used only for injecting input directly into fc layers
            fc_input_size += added_fc_input_size
            for idx, hidden_size in enumerate(hidden_sizes):
                fc_layer = nn.Linear(fc_input_size, hidden_size)
                fc_input_size = hidden_size

                fc_layer.weight.data.uniform_(-init_w, init_w)
                fc_layer.bias.data.uniform_(-init_w, init_w)

                self.fc_layers.append(fc_layer)

                if self.fc_normalization_type == 'batch':
                    self.fc_norm_layers.append(nn.BatchNorm1d(hidden_size))
                if self.fc_normalization_type == 'layer':
                    self.fc_norm_layers.append(nn.LayerNorm(hidden_size))

            self.last_fc = nn.Linear(fc_input_size, output_size)
            self.last_fc.weight.data.uniform_(-init_w, init_w)
            self.last_fc.bias.data.uniform_(-init_w, init_w)
        obs_dim = self.input_width * self.input_height
        action_dim = self.output_size
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(self.hidden_sizes) > 0:
                last_hidden_size = self.hidden_sizes[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def forward(self, obs):
        conv_input = obs.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        #image = h.reshape((84, 84, 3)).numpy().copy()
        #cv2.imshow('goal', image)
        #cv2.waitKey(10)

        #h = self.apply_forward_conv(h)
        i = 0
        for layer in self.conv_layers:
            h = layer(h)
            if self.conv_normalization_type != 'none':
                pass
                #h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                pass
                #h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
            i += 1


        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = h.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)
        #h = self.apply_forward_fc(h)
        i = 0
        for layer in self.fc_layers:
            h = layer(h)
            if self.fc_normalization_type != 'none':
                pass
                #h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
            i += 1

        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, -2, 20)
            std = torch.exp(log_std)
        else:
            std = self.std

        tanh_normal = TanhNormal(mean, std)
        if self.deterministic:
            return tanh_normal.mean
        else:
            return tanh_normal


    def get_action(self, obs_np, ):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs_np, ):
        dist = self._get_dist_from_np(obs_np)
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist
    
    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h):
        for i, layer in enumerate(self.fc_layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                h = self.fc_norm_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def reset(self):
        pass

    def make_deterministic(self):
        self.deterministic = True
