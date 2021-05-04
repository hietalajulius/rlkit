import torch
from utils import get_variant, argsparser
from envs.cloth import ClothEnvPickled as ClothEnv
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.pythonplusplus import identity
import numpy as np
from torch import nn
from rlkit.torch.distributions import (
    TanhNormal
)
from typing import Tuple

from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy
)

class ScriptPolicy(torch.nn.Module):
    def __init__(self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes_all=[],
            hidden_sizes_aux=[],
            hidden_sizes_main=[],
            added_fc_input_size=0,
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            hidden_init=nn.init.xavier_uniform_,
            hidden_activation=nn.ReLU(),
            output_activation=identity,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            aux_output_size=1,
            std=None):
        super(ScriptPolicy, self).__init__()
        # CNN
        assert hidden_sizes_all[-1] == hidden_sizes_aux[0] + hidden_sizes_main[0]
        assert len(kernel_sizes) == \
            len(n_channels) == \
            len(strides) == \
            len(paddings)
        assert conv_normalization_type in {'none', 'batch', 'layer'}
        assert fc_normalization_type in {'none', 'batch', 'layer'}
        assert pool_type in {'none', 'max2d'}
        if pool_type == 'max2d':
            assert len(pool_sizes) == len(pool_strides) == len(pool_paddings)

        self.hidden_sizes_all = hidden_sizes_all
        self.hidden_sizes_aux = hidden_sizes_aux
        self.hidden_sizes_main = hidden_sizes_main
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.conv_normalization_type = conv_normalization_type
        self.fc_normalization_type = fc_normalization_type
        self.added_fc_input_size = added_fc_input_size
        self.conv_input_length = self.input_width * \
            self.input_height * self.input_channels
        self.pool_type = pool_type

        self.aux_output_size = aux_output_size
        self.aux_activation = nn.Sigmoid()

        self.init_w = init_w

        self.conv_layers = nn.ModuleList()
        self.conv_norm_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fc_all_layers = nn.ModuleList()
        self.fc_aux_layers = nn.ModuleList()
        self.fc_main_layers = nn.ModuleList()

        self.fc_all_norm_layers = nn.ModuleList()
        self.fc_aux_norm_layers = nn.ModuleList()
        self.fc_main_norm_layers = nn.ModuleList()

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

        fc_all_input_size = self.conv_output_flat_size
        fc_all_input_size += added_fc_input_size
        self.add_fc_layers(fc_all_input_size, self.fc_all_layers, self.fc_all_norm_layers, self.hidden_sizes_all)

        self.add_fc_layers(self.hidden_sizes_aux[0], self.fc_aux_layers, self.fc_aux_norm_layers, self.hidden_sizes_aux[1:])
        self.last_fc_aux = nn.Linear(self.hidden_sizes_aux[-1], aux_output_size)
        self.last_fc_aux.weight.data.uniform_(-init_w, init_w)
        self.last_fc_aux.bias.data.uniform_(-init_w, init_w)


        self.add_fc_layers(self.hidden_sizes_main[0], self.fc_main_layers, self.fc_main_norm_layers, self.hidden_sizes_main[1:])
        self.last_fc_main = nn.Linear(self.hidden_sizes_main[-1], output_size)
        self.last_fc_main.weight.data.uniform_(-init_w, init_w)
        self.last_fc_main.bias.data.uniform_(-init_w, init_w)

        # TanhCNNGaussianPolicy
        self.foo = "bar"
        self.vers = 0
        obs_dim = self.input_width * self.input_height
        action_dim = self.output_size
        self.log_std = None
        self.std = std
        if std is None:
            last_hidden_size = obs_dim
            if len(self.hidden_sizes_main) > 0:
                last_hidden_size = self.hidden_sizes_main[-1]
            self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
            self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
            self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        else:
            self.log_std = np.log(std)
            assert -20 <= self.log_std <= 2

    def forward(self, obs):
        out = self.cnn_forward(obs, return_last_main_activations=True)
        h = out[0]
        h_aux = out[1]
        mean = self.last_fc_main(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, -20, 2)
            std = torch.exp(log_std)
        else:
            std = self.std

        #tanh_normal = TanhNormal(mean, std)

        return mean, std, h_aux, torch.tanh(mean)

    def add_fc_layers(self, fc_input_size, fc_module_list, norm_module_list, hidden_sizes):
        for hidden_size in hidden_sizes:
            fc_layer = nn.Linear(fc_input_size, hidden_size)
            fc_input_size = hidden_size

            fc_layer.weight.data.uniform_(-self.init_w, self.init_w)
            fc_layer.bias.data.uniform_(-self.init_w, self.init_w)

            fc_module_list.append(fc_layer)

            if self.fc_normalization_type == 'batch':
                norm_module_list.append(nn.BatchNorm1d(hidden_size))
            if self.fc_normalization_type == 'layer':
                norm_module_list.append(nn.LayerNorm(hidden_size))

    def cnn_forward(self, input, return_last_main_activations: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        conv_input = input.narrow(start=0,
                                  length=self.conv_input_length,
                                  dim=1).contiguous()
        # reshape from batch of flattened images into (channels, w, h)
        h = conv_input.view(conv_input.shape[0],
                            self.input_channels,
                            self.input_height,
                            self.input_width)

        #debug_image = h[0].detach().cpu().numpy()
        #cv2.imshow('goal', debug_image.reshape((100, 100, 1)))
        #cv2.waitKey(10)
        
        #data = h.reshape((100, 100, 1)).cpu().numpy().copy()
        #cv2.imwrite(f'./train_images/{str(np.random.rand())}.png', data)

        #image = h.reshape((100, 100, 1)).cpu().numpy().copy()
        

        h = self.apply_forward_conv(h)

    

        # flatten channels for fc layers
        h = h.view(h.size(0), -1)
        if self.added_fc_input_size != 0:
            extra_fc_input = input.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )
            h = torch.cat((h, extra_fc_input), dim=1)

        h_all = h# self.apply_forward_fc(h, self.fc_all_layers, self.fc_all_norm_layers)

        for i, layer in enumerate(self.fc_all_layers):
            h_all = layer(h_all)
            if self.fc_normalization_type != 'none':
                pass
                #h = norm_layers[i](h)
            h_all = self.hidden_activation(h_all)

        h_aux = h_all[:, self.hidden_sizes_main[-1]:]
        h_main = h_all[:, :self.hidden_sizes_main[-1]]

        #h_aux = self.apply_forward_fc(h_aux, self.fc_aux_layers, self.fc_aux_norm_layers)

        for i, layer in enumerate(self.fc_aux_layers):
            h_aux = layer(h_aux)
            if self.fc_normalization_type != 'none':
                pass
                #h = norm_layers[i](h)
            h_aux = self.hidden_activation(h_aux)

        #h_main = self.apply_forward_fc(h_main, self.fc_main_layers, self.fc_main_norm_layers)
        for i, layer in enumerate(self.fc_main_layers):
            h_main = layer(h_main)
            if self.fc_normalization_type != 'none':
                pass
                #h = norm_layers[i](h)
            h_main = self.hidden_activation(h_main)

        h_aux = self.last_fc_aux(h_aux)
        h_aux = self.aux_activation(h_aux)

        if return_last_main_activations:
            return h_main, h_aux
            
        return self.output_activation(self.last_fc_main(h_main)), h_aux

    def apply_forward_conv(self, h):
        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if self.conv_normalization_type != 'none':
                pass
                #h = self.conv_norm_layers[i](h)
            if self.pool_type != 'none':
                pass
                #3h = self.pool_layers[i](h)
            h = self.hidden_activation(h)
        return h

    def apply_forward_fc(self, h, layers, norm_layers):
        for i, layer in enumerate(layers):
            h = layer(h)
            if self.fc_normalization_type != 'none':
                pass
                #h = norm_layers[i](h)
            h = self.hidden_activation(h)
        return h

class TanhScriptPolicy(ScriptPolicy, TorchStochasticPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        mean, std, h_aux, _ = super().forward(x)
        tanh_normal = TanhNormal(mean, std)
        return tanh_normal, h_aux



    
