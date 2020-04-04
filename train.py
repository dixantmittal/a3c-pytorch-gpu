import argparse
import os

import torch.multiprocessing as mp

from model import A3CModel
from worker import worker

if not os.path.exists('logs'):
    os.makedirs('logs')

if __name__ == "__main__":
    # Set method to spawn to use CUDA
    mp.set_start_method("spawn", True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument("--gamma", type=float, default=0.99, help='Discount factor')
    parser.add_argument("--n_workers", type=int, default=5, help='Number of training workers')
    parser.add_argument('--alpha', type=float, default=0.5, help='Coefficient for value loss')
    parser.add_argument('--max_grad_norm', type=float, default=50, help='Max L2-norm for the gradients')
    parser.add_argument('--save_model', default=None, help='File to save the model')
    parser.add_argument('--load_model', default=None, help='File to load the model')
    args = parser.parse_args()

    # Create model object on CPU
    model = A3CModel()
    model.load(args.load_model)

    # Put the model's parameters in the shared memory
    model.share_memory()

    # Create processes
    processes = [mp.Process(target=worker, args=(i, model, args)) for i in range(args.n_workers)]

    # Start processes
    [p.start() for p in processes]

    # Wait for processes to finish
    [p.join() for p in processes]

    # Save the model
    model.save(args.save_model)
