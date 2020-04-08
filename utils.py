import torch
import torch.nn.functional as f

from episode import Episode
from simulator import _get_simulator, _state_to_tensor


# Copy gradients from source to target's parameters
def _copy_gradients(target, source):
    for shared_param, param in zip(target.parameters(), source.parameters()):
        if param.grad is not None:
            shared_param._grad = param.grad.clone().cpu()


def _calculate_loss(logits, critics, actions, rewards, args):
    # Find Q values
    for i in reversed(range(0, len(rewards) - 1)):
        rewards[i] += args.gamma * rewards[i + 1]

    # Find the weighted likelihood
    negative_likelihood = f.cross_entropy(logits, actions.long(), reduction='none')
    weighted_negative_likelihood = negative_likelihood * (rewards - critics.detach().clone())

    # Calculate the combined loss for actor and critic
    loss = torch.sum(weighted_negative_likelihood) + args.alpha * f.smooth_l1_loss(critics, rewards)

    return loss


# Sync target's parameters with source's parameters
def _sync_model(target, source):
    target.load_state_dict(source.state_dict())


# Play an episode using the model and collect the data
def _collect_episode(model, args):
    episode = Episode()

    # Fetch a simulator object
    simulator = _get_simulator()

    # Start the episode
    state = simulator.reset()
    done = False
    while not done:
        # Convert state to tensor
        state = _state_to_tensor(state).to(args.device)

        # Sample an action from the model
        action, actor, critic = model.choose_action(state, train=True)

        # Simulate the action
        state, reward, done, _ = simulator.step(action)

        # Store the data in episode object
        episode.add(actor, critic, action, reward)

    # Return episode's data
    return episode.get()
