import logging
from copy import deepcopy
from datetime import datetime

import torch
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from utils import _sync_model, _collect_episode, _calculate_loss, _copy_gradients


def worker(idx, shared_model, args):
    writer = SummaryWriter('tensorboard/worker:{:02}-{}'.format(idx, datetime.now().strftime("%d:%m::%H:%M")))
    logging.basicConfig(filename='logs/worker:{:02}.log'.format(idx),
                        filemode='w',
                        format='%(message)s',
                        level=logging.DEBUG)

    # Find the number of GPUs available
    n_gpu = torch.cuda.device_count()
    # Assign the worker to a GPU
    args.device = 'cpu' if n_gpu == 0 else torch.device(idx % n_gpu)

    # Create an optimiser for shared model's parameters
    optimiser = torch.optim.SGD(shared_model.parameters(), lr=args.lr)

    # Create a local copy of the model
    model = deepcopy(shared_model)

    # Move the local model to the assigned device
    model.to(args.device)
    model.train()

    for i in tqdm(range(args.n_steps), position=idx, desc='worker:{:02}'.format(idx)):
        # Sync the local model with the shared model at the beginning
        _sync_model(model, shared_model)

        # Collect the data from an episode
        logits, critics, actions, rewards = _collect_episode(model, args)

        logging.info('Iteration: %s, Episode reward: %s', i, rewards.sum().cpu().item())
        writer.add_scalar('episode_reward', rewards.sum().cpu().item(), i)
        writer.close()

        # Calculate the loss value using local model
        loss = _calculate_loss(logits, critics, actions, rewards, args)

        # Calculate gradients and store them in local model
        loss.backward()

        # Clip gradients
        clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # Copy the gradients to shared model
        _copy_gradients(shared_model, model)

        # Take an optimisation step
        optimiser.step()

        # Remove the gradients from local model
        model.zero_grad()
