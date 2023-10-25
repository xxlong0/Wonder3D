import torch
import torch.nn as nn

import models
from models.utils import get_activation
from models.network_utils import get_encoding, get_mlp
from systems.utils import update_module_step


@models.register('volume-radiance')
class VolumeRadiance(nn.Module):
    def __init__(self, config):
        super(VolumeRadiance, self).__init__()
        self.config = config
        self.with_viewdir = False #self.config.get('wo_viewdir', False)
        self.n_dir_dims = self.config.get('n_dir_dims', 3)
        self.n_output_dims = 3

        if self.with_viewdir:
            encoding = get_encoding(self.n_dir_dims, self.config.dir_encoding_config)
            self.n_input_dims = self.config.input_feature_dim + encoding.n_output_dims
            # self.network_base = get_mlp(self.config.input_feature_dim, self.n_output_dims, self.config.mlp_network_config)   
        else:
            encoding = None
            self.n_input_dims = self.config.input_feature_dim
            
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)    
        self.encoding = encoding
        self.network = network
    
    def forward(self, features, dirs, *args):

        # features = features.detach()
        if self.with_viewdir:
            dirs = (dirs + 1.) / 2. # (-1, 1) => (0, 1)
            dirs_embd = self.encoding(dirs.view(-1, self.n_dir_dims))
            network_inp = torch.cat([features.view(-1, features.shape[-1]), dirs_embd] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
            # network_inp_base = torch.cat([features.view(-1, features.shape[-1])] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
            color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
            # color_base = self.network_base(network_inp_base).view(*features.shape[:-1], self.n_output_dims).float()
            # color = color + color_base
        else:
            network_inp = torch.cat([features.view(-1, features.shape[-1])] + [arg.view(-1, arg.shape[-1]) for arg in args], dim=-1)
            color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()

        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)

    def regularizations(self, out):
        return {}


@models.register('volume-color')
class VolumeColor(nn.Module):
    def __init__(self, config):
        super(VolumeColor, self).__init__()
        self.config = config
        self.n_output_dims = 3
        self.n_input_dims = self.config.input_feature_dim
        network = get_mlp(self.n_input_dims, self.n_output_dims, self.config.mlp_network_config)
        self.network = network
    
    def forward(self, features, *args):
        network_inp = features.view(-1, features.shape[-1])
        color = self.network(network_inp).view(*features.shape[:-1], self.n_output_dims).float()
        if 'color_activation' in self.config:
            color = get_activation(self.config.color_activation)(color)
        return color

    def regularizations(self, out):
        return {}
