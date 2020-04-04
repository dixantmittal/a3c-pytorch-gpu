import torch
from torch import nn as nn

from torch.distributions import Categorical


# A simple A3C model.
class A3CModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(4, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 2))

        self.value = nn.Sequential(nn.Linear(4, 128),
                                   nn.ReLU(),
                                   nn.Linear(128, 1))

    # Find the policy logits
    def actor(self, x):
        return self.policy(x)

    # Find the critic value
    def critic(self, x):
        return self.value(x)

    # Sample an action based on actor's policy. If train=True, return logits and critic values
    def choose_action(self, x, train=False):
        actor = self.actor(x)
        action = Categorical(logits=actor).sample().cpu().item()
        if train:
            critic = self.critic(x)
            return action, actor, critic
        else:
            return action

    # Save model parameters to file
    def save(self, file):
        if file is None or file == '':
            return

        torch.save(self.state_dict(), file)

    # Load model parameters from file
    def load(self, file):
        if file is None or file == '':
            return

        self.load_state_dict(torch.load(file))
