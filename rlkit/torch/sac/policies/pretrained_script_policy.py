import torch
import torch.nn.functional as F
import torchvision.models as models
from rlkit.pythonplusplus import identity
from rlkit.torch.distributions import (
    TanhNormal
)
from typing import Tuple

from rlkit.torch.sac.policies.base import (
    TorchStochasticPolicy
)

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x



class CustomScriptPolicy(torch.nn.Module):
    def __init__(self,
            input_width,
            input_height,
            input_channels,
            output_size,
            kernel_sizes,
            n_channels,
            strides,
            paddings,
            hidden_sizes_aux=[],
            hidden_sizes_main=[],
            added_fc_input_size=0,
            net_type="resnet",
            conv_normalization_type='none',
            fc_normalization_type='none',
            init_w=1e-4,
            hidden_init=torch.nn.init.xavier_uniform_,
            hidden_activation=torch.nn.ReLU(),
            output_activation=identity,
            pool_type='none',
            pool_sizes=None,
            pool_strides=None,
            pool_paddings=None,
            aux_output_size=1,
            last_hidden_size=1000, #Tough to change in pretrained models,
            pretrained=True,
            std=None):
        super(CustomScriptPolicy, self).__init__()
        if net_type == "resnet":
            self.model = models.resnet18(pretrained=pretrained)
        elif net_type == "densenet":
            self.model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', pretrained=True)
            #self.model = models.densenet161(pretrained=pretrained)
        self.model.cuda()

        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.conv_input_length = self.input_width * self.input_height * self.input_channels
        self.aux_output_size = aux_output_size
        self.aux_activation = torch.nn.Sigmoid()
        self.added_fc_input_size = added_fc_input_size
        self.last_hidden_size = last_hidden_size
        self.mean_fc = torch.nn.Linear(self.last_hidden_size + added_fc_input_size, 3)
        self.std_fc = torch.nn.Linear(self.last_hidden_size + added_fc_input_size, 3)
        self.corners_fc = torch.nn.Linear(self.last_hidden_size, 8)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        print(x.shape, self.conv_input_length, self.added_fc_input_size)
        action_x = x.narrow(
                start=self.conv_input_length,
                length=self.added_fc_input_size,
                dim=1,
            )

        cnn_x = x.narrow(start=0, length=self.conv_input_length, dim=1).contiguous()
        cnn_x = cnn_x.view(cnn_x.shape[0],
                    self.input_channels,
                    self.input_height,
                    self.input_width)

        cnn_x = self.model.forward(cnn_x)

        action_x = torch.cat((cnn_x, action_x), dim=1)
        corners_x = cnn_x

        mean = self.mean_fc(action_x)
        tanh_mean = torch.tanh(mean)


        log_std = self.std_fc(action_x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        corners_x = self.corners_fc(corners_x)
        corners_x = self.aux_activation(corners_x)
        
        return mean, std, corners_x, tanh_mean

class CustomTanhScriptPolicy(CustomScriptPolicy, TorchStochasticPolicy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def forward(self, x):
        mean, std, h_aux, _ = super().forward(x)
        tanh_normal = TanhNormal(mean, std)
        return tanh_normal, h_aux
