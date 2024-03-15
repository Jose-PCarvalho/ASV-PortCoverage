import itertools
import sys

from src.Environment.Environment import *

from src.Environment.Actions import *
from src.Environment.Vizualization import *

import random
import yaml


def read_integer():
    while True:
        try:
            num = int(input("Enter an integer: "))
            if num == 8:
                return Actions.FORWARD
            elif num == 2:
                return Actions.BACKWARD
            elif num == 6:
                return Actions.ROTATE
            elif num == 4:
                return Actions.ROTATE
            else:
                return Actions.FORWARD
            return num
        except ValueError:
            print("Invalid input. Please enter an integer.")


with open('configs/Evaluation.yaml', 'rb') as f:
    conf = yaml.safe_load(f.read())  # load the config file

env = Environment(EnvironmentParams(conf['env2']))
Viz = Vizualization()
sys.setrecursionlimit(2000)
while True:
    observation, _ = env.reset()
    print('')
    env.render(center=False)
    for t in itertools.count():
        observation_, reward, done, truncated, info = env.step(env.get_heuristic_action())
        env.render(center=True)
        print(reward)
        observation = observation_
        if done or truncated:
            print("yei")
            break
