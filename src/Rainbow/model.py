from __future__ import division
import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


# Factorised NoisyLinear layer with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, args, action_space):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.convs = nn.Sequential(nn.Conv2d(12, 32, 3, stride=1, padding='same'), nn.ReLU(),
                                       nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),  # stride 1
                                       nn.Conv2d(64, 64, 3, stride=2),
                                       nn.ReLU())  # kernel size 3 , stride 2, stable version had kernel 4
        self.conv_output_size = self._get_conv_out([12, 59, 59])
        self.fc_1_v = NoisyLinear(self.conv_output_size + 19, args.hidden_size, std_init=args.noisy_std)
        self.fc_1_a = NoisyLinear(self.conv_output_size + 19, args.hidden_size, std_init=args.noisy_std)
        self.fc_2_v = NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_2_a = NoisyLinear(args.hidden_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_3_v = NoisyLinear(args.hidden_size, 1, std_init=args.noisy_std)
        self.fc_3_a = NoisyLinear(args.hidden_size, action_space, std_init=args.noisy_std)

    def _get_conv_out(self, shape):
        o = self.convs(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, b, a, o):
        b = b.view(-1, 1)
        a = a.view(a.size(0), -1)
        o = o.view(o.size(0), -1)
        x = x.reshape(x.size(0), -1, x.size(-2), x.size(-1))
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, b), 1)
        x = torch.cat((x, a), 1)
        x = torch.cat((x, o), 1)
        v = self.fc_3_v(F.relu(self.fc_2_v(F.relu(self.fc_1_v(x))))) # Value stream
        a = self.fc_3_a(F.relu(self.fc_2_a(F.relu(self.fc_1_a(x)))))  # Advantage stream
        q = v + a - a.mean(1, keepdim=True)  # Combine streams
        return q

    def reset_noise(self):
        for name, module in self.named_children():
            if 'fc' in name:
                module.reset_noise()
