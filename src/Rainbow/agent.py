from __future__ import division
import os
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from src.Rainbow.model import DQN


class Agent:
    def __init__(self, args, action_space):
        self.action_space = action_space
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.norm_clip = args.norm_clip
        self.tau = args.tau
        self.device = args.device

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                state_dict = torch.load(args.model,
                                        map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'), ('conv1.bias', 'convs.0.bias'),
                                             ('conv2.weight', 'convs.2.weight'), ('conv2.bias', 'convs.2.bias'),
                                             ('conv3.weight', 'convs.4.weight'), ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                        del state_dict[old_key]  # Delete old keys for strict load_state_dict
                self.online_net.load_state_dict(state_dict)
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)

        self.hard_update()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state, battery, last_action, orientation):
        with torch.no_grad():
            # state = torch.tensor(state[-1], dtype=torch.float32, device='cuda')
            state = torch.tensor(state, dtype=torch.float32, device=self.device)/255
            battery = torch.tensor(battery, dtype=torch.int32, device=self.device)
            last_action = F.one_hot(torch.tensor(last_action, dtype=torch.int64, device=self.device), 5)
            orientation = torch.tensor(orientation, dtype=torch.int32, device=self.device)
            return (self.online_net(state, battery, last_action, orientation)).argmax(1).tolist()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, battery, last_action, orientation,
                     epsilon=0.0000):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state, battery,
                                                                                                     last_action,
                                                                                                     orientation)

    def learn(self, mem):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights, battery, next_battery, last_action, \
        next_last_action, orientation, next_orientation = mem.sample(self.batch_size)
        # Calculate current state probabilities (online network noise already sampled)
        q_values = self.online_net(states, battery, F.one_hot(last_action, 5), orientation)
        q_curr = q_values[range(self.batch_size), actions]
        with torch.no_grad():
            # Calculate nth next state probabilities
            q_online_value = self.online_net(next_states, next_battery, F.one_hot(next_last_action, 5), next_orientation)
            argmax_indices_ns = q_online_value.argmax(1)  #
            self.target_net.reset_noise()  # Sample new target net noise
            q_target_values = self.target_net(next_states, next_battery, F.one_hot(next_last_action, 5), next_orientation)
            q_target = q_target_values[range(self.batch_size), argmax_indices_ns]
        q_target.detach()
        target = returns + nonterminals * (self.discount ** self.n) * q_target
        loss = F.smooth_l1_loss(q_curr, target, reduction="none")
        self.online_net.zero_grad()
        (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
        self.optimiser.step()
        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update_target_net(self):
        # self.target_net.load_state_dict(self.online_net.state_dict())
        self.eval()
        with torch.no_grad():
            for target_param, local_param in zip(self.target_net.parameters(), self.online_net.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        self.train()

    def hard_update(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='model.pth'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state, battery, last_action,orientation):
        with torch.no_grad():
            last_action = F.one_hot(last_action, 5)
            return (self.online_net(state.unsqueeze(0), battery.unsqueeze(0),
                                    last_action.unsqueeze(0),orientation.unsqueeze(0))).max(1)[0].item()

    def eval_q(self, state, battery, last_action,orientation):
        self.eval()
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device)/255
            battery = torch.tensor(battery, dtype=torch.int32, device=self.device)
            last_action = F.one_hot(torch.tensor(last_action, dtype=torch.int64, device=self.device), 5)
            orientation = torch.tensor(orientation, dtype=torch.int32, device=self.device)
            return (self.online_net(state, battery, last_action, orientation)).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
