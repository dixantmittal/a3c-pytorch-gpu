import torch


# Class to store data from an episode compactly
class Episode:
    def __init__(self):
        self.logits = []
        self.critics = []
        self.actions = []
        self.rewards = []

    def add(self, actor, critic, action, reward):
        self.logits.append(actor)
        self.critics.append(critic)
        self.actions.append(action)
        self.rewards.append(reward)

    def get(self):
        device = self.logits[0].device

        return (torch.stack(self.logits),
                torch.stack(self.critics).squeeze(),
                torch.tensor(self.actions).long().to(device),
                torch.tensor(self.rewards).float().to(device))
