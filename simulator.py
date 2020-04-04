import gym
import torch


# Factory to create a simulator object
def _get_simulator():
    return gym.make('CartPole-v0')


# Convert state object to tensor (Edit this function depending upon the simulator)
def _state_to_tensor(state):
    return torch.tensor(state).float()
