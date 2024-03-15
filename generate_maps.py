import bz2
import pickle
import random

from tqdm import trange
import os
from src.Environment.Environment import *
import yaml

with open('configs/Evaluation.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file


def save_memory(memory, memory_path):
    if os.path.exists(memory_path):
        os.remove(memory_path)
    with bz2.open(memory_path, 'wb') as zipped_pickle_file:
        pickle.dump(memory, zipped_pickle_file)


env = Environment(EnvironmentParams(conf['env1']))
maps = []
for T in trange(1, 10000):
    env.reset(False)
    env.render()
    maps.append(copy.deepcopy(env.state))
save_memory(maps, 'maps/datasets/map1.pth')
