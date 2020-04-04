import argparse
from itertools import repeat

import torch
import torch.multiprocessing as mp
from numpy import std, mean
from scipy.stats import sem

from model import A3CModel
from simulator import _get_simulator


def tester(idx, model):
    n_gpu = torch.cuda.device_count()
    device = 'cpu' if n_gpu == 0 else torch.device(idx % n_gpu)

    model.to(device)

    simulator = _get_simulator()

    episode_reward = 0
    state = simulator.reset()
    done = False
    while not done:
        state = torch.tensor(state).float().to(device)
        action = model.choose_action(state)

        state, reward, done, _ = simulator.step(action)

        episode_reward += reward

    return episode_reward


if __name__ == '__main__':
    mp.set_start_method("spawn", True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_episodes', type=int, default=20, help='Number of training steps')
    parser.add_argument("--n_workers", type=int, default=5, help='Number of training workers')
    parser.add_argument('--load_model', default=None, help='File to load the model')
    args = parser.parse_args()

    model = A3CModel()
    model.load(args.load_model)
    model.share_memory()

    rewards = mp.Pool(args.n_workers).starmap_async(func=tester, iterable=zip(range(args.n_episodes), repeat(model, args.n_episodes))).get()

    print('Rewards in each episode:', rewards)
    print('mean: ', mean(rewards))
    print('stderr: ', sem(rewards))
    print('stddev: ', std(rewards))
