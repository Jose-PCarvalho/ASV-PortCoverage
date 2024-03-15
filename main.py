import argparse
import bz2
import random
from datetime import datetime
import os
import pickle

import numpy as np
import torch
import yaml
from tqdm import trange
from src.Environment.Environment import *
from src.Rainbow.agent import *
from src.Rainbow.memory import ReplayMemory
from test import test

def log(s, log_dir=None):
    print('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s)
    if dir is not None:
        with open(log_dir, 'a') as file:
            file.write(('[' + str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S')) + '] ' + s + '\n'))


def save_memory(memory, memory_path, disable_bzip):
    if disable_bzip:
        with open(memory_path, 'wb') as pickle_file:
            pickle.dump(memory, pickle_file)
    else:
        with bz2.open(memory_path, 'wb') as zipped_pickle_file:
            pickle.dump(memory, zipped_pickle_file)


def use_pseudo(ovrl):
    p = -1.4 * (1 - ovrl) + 0.7
    eps = np.random.random()
    if eps < p:
        return True
    else:
        return False


# Note that hyperparameters may originally be reported in ATARI game frames instead of agent steps
parser = argparse.ArgumentParser(description='Rainbow')
parser.add_argument('--seed', type=int, default=123, help='Random seed')
parser.add_argument('--id', type=str, default='Test', help='Experiment ID')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--T-max', type=int, default=int(70e4), metavar='STEPS',
                    help='Number of training steps (4x number of frames)')
parser.add_argument('--history-length', type=int, default=3, metavar='T', help='Number of consecutive states processed')
parser.add_argument('--hidden-size', type=int, default=512, metavar='SIZE', help='Network hidden size')
parser.add_argument('--noisy-std', type=float, default=0.5, metavar='σ',
                    help='Initial standard deviation of noisy linear layers')
parser.add_argument('--model', type=str, metavar='PARAMS', help='Pretrained model (state dict)')
parser.add_argument('--memory-capacity', type=int, default=int(1e6), metavar='CAPACITY',
                    help='Experience replay memory capacity')
parser.add_argument('--replay-frequency', type=int, default=4, metavar='k', help='Frequency of sampling from memory')
parser.add_argument('--priority-exponent', type=float, default=0.5, metavar='ω',
                    help='Prioritised experience replay exponent (originally denoted α)')
parser.add_argument('--priority-weight', type=float, default=0.4, metavar='β',
                    help='Initial prioritised experience replay importance sampling weight')
parser.add_argument('--multi-step', type=int, default=1, metavar='n', help='Number of steps for multi-step return')
parser.add_argument('--discount', type=float, default=0.999, metavar='γ', help='Discount factor')
parser.add_argument('--reward-clip', type=int, default=1, metavar='VALUE', help='Reward clipping (0 to disable)')
parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='η', help='Learning rate')
parser.add_argument('--adam-eps', type=float, default=float(1.6e-4), metavar='ε', help='Adam epsilon')
parser.add_argument('--batch-size', type=int, default=32, metavar='SIZE', help='Batch size')
parser.add_argument('--norm-clip', type=float, default=2.5, metavar='NORM', help='Max L2 norm for gradient clipping')
parser.add_argument('--learn-start', type=int, default=int(2e3), metavar='STEPS',
                    help='Number of steps before starting training')
parser.add_argument('--evaluate', action='store_true', help='Evaluate only')
parser.add_argument('--evaluation-interval', type=int, default=5000, metavar='STEPS',
                    help='Number of training steps between evaluations')
parser.add_argument('--evaluation-episodes', type=int, default=5, metavar='N',
                    help='Number of evaluation episodes to average over')
parser.add_argument('--evaluation-size', type=int, default=10000, metavar='N',
                    help='Number of transitions to use for validating Q')
parser.add_argument('--render', action='store_true', help='Display screen (testing only)')
parser.add_argument('--enable-cudnn', action='store_true', help='Enable cuDNN (faster but nondeterministic)')
parser.add_argument('--checkpoint-interval', default=5000,
                    help='How often to checkpoint the model, defaults to 0 (never checkpoint)')
parser.add_argument('--memory', help='Path to save/load the memory from')
parser.add_argument('--disable-bzip-memory', action='store_true',
                    help='Don\'t zip the memory file. Not recommended (zipping is a bit slower and much, much smaller)')
parser.add_argument('--config-file', type=str, default='configs/Evaluation.yaml')
parser.add_argument('--log-file', type=str, default='results/log.txt')
parser.add_argument('--starting-environment', type=int, default=1)
parser.add_argument('--tau', type=float, default=0.001)

# Setup
args = parser.parse_args()
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
    print(' ' * 26 + k + ': ' + str(v))
results_dir = os.path.join('results', args.id)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf'), 'overlap': [], 'time_save': []}
np.random.seed(args.seed)
torch.manual_seed(np.random.randint(1, 10000))
if torch.cuda.is_available() and not args.disable_cuda:
    print("Initiating cuda")
    args.device = torch.device('cuda')
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = True
else:
    args.device = torch.device('cpu')
    print("Initiating CPU")

with open(args.config_file, 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file

if args.log_file:
    with open(args.log_file, 'w') as file:
        pass

number_envs = len(conf.keys())
env = Environment(EnvironmentParams(conf['env1']))
action_space = env.action_space()
starting_priority_weight = args.priority_weight
dqn = Agent(args, action_space)
mem = ReplayMemory(args, args.memory_capacity)
avg_overlap = 0
retries = 0
all_T = 0
T = 0
e = args.starting_environment
priority_weight_increase = (args.priority_weight) / (1e6)
while e < number_envs + 1:

    env_str = 'env' + str(e)
    args.T_max = conf[env_str]['base_steps']
    env = Environment(EnvironmentParams(conf[env_str]))
    print(conf[env_str])
    non_obstacles = conf[env_str]['number_obstacles'] == 0
    env_size = conf[env_str]['size']
    use_pseudo_agent, pseudo_episode = non_obstacles, non_obstacles
    val_mem = ReplayMemory(args, args.evaluation_size)
    all_T += T
    T, done, truncated = 0, True, True
    avg_overlap = 1
    while T < args.evaluation_size:

        if done or truncated:
            state, info = env.reset()

        action = env.get_heuristic_action()
        next_state, _, done, truncated, info = env.step(action)
        val_mem.append(state[0], state[1], state[2], state[3], [-1], [0.0], done, truncated)
        state = next_state
        T += 1
    else:
        # Training loop
        dqn.train()
        done = True
        truncated = False
        last_truncated = False

        for T in trange(1, args.T_max + 1):

            if done or truncated:
                last_truncated = truncated
                state, info = env.reset()

            if T % args.replay_frequency == 0:
                dqn.reset_noise()  # Draw a new set of noisy weights

            if (last_truncated or any(info)) and np.random.random() < 0.9:
                action = dqn.act(state[0], state[1], state[2], state[3])
                if last_truncated:
                    ac = env.get_heuristic_action()
                else:
                    ac = env.get_heuristic_action(info)
                for i, a in enumerate(ac):
                    if a is not None:
                        action[i] = a
            else:
                action = dqn.act(state[0], state[1],
                                 state[2], state[3])
                # ac = env.detect_collision(action)
                # if ac is not None:
                #     for i, a in enumerate(ac):
                #         if a is not None:
                #             action[i] = a

            next_state, reward, done, truncated, info = env.step(action)  # Step

            mem.append(state[0], state[1], state[2], state[3], action, reward, done,
                       truncated)  # Append transition to memory

            # Train and test
            if T >= args.learn_start:

                mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)

                if T % args.replay_frequency == 0:
                    dqn.learn(mem)  # Train with n-step distributional double-Q learning

                if T % args.evaluation_interval == 0:
                    dqn.eval()  # Set DQN (online network) to evaluation mode
                    avg_reward, avg_Q, avg_overlap, avg_time_save = test(args, T + all_T, dqn, val_mem, metrics,
                                                                         results_dir,
                                                                         env_args=conf[env_str])  # Test
                    log('T = ' + str(T) + ' / ' + str(args.T_max) + ' | env: ' + conf[env_str]['name'] +
                        ' | Avg. reward: ' + str(avg_reward) + ' | Avg. Q: ' + str(avg_Q) + ' | Avg. Overlap: ' + str(
                        avg_overlap) + ' | Avg. Time Save: ' + str(avg_time_save), args.log_file)
                    dqn.train()  # Set DQN (online network) back to training mode

                    # If memory path provided, save it
                    if args.memory is not None:
                        save_memory(mem, args.memory, args.disable_bzip_memory)

                # Update target network
                if T % args.replay_frequency == 0:
                    dqn.update_target_net()

                # Checkpoint the network
                if (args.checkpoint_interval != 0) and (T % args.checkpoint_interval == 0):
                    dqn.save(results_dir, 'checkpoint.pth')

            state = next_state

        e += 1
        dqn.save(results_dir, conf[env_str]['name'] + '.pth')

    # env.close()
