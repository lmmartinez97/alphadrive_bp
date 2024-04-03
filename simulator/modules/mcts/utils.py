"""Utility functions for the AlphaZero algorithm."""


from __future__ import division

import os

from datetime import datetime
from rich import print
from typing import List, Tuple

from .helpers import AlphaZeroConfig, SharedStorage, ReplayBuffer
from .network import Network

import numpy as np
#####################
# MCTS functions   #
#####################

def softmax_sample(visit_counts: List[Tuple[int, int]]) -> Tuple[int, int]:
    """
    Selects an action probabilistically based on the visit counts using the softmax function.

    Args:
        visit_counts (List[Tuple[int, int]]): A list of tuples where each tuple contains the visit count and the corresponding action.

    Returns:
        Tuple[int, int]: The selected action and its visit count.
    """
    counts, actions = zip(*visit_counts)
    probs = np.exp(counts) / np.sum(np.exp(counts))  # Apply softmax to visit counts
    action = np.random.choice(actions, p=probs)  # Select action probabilistically
    #find action index
    action_index = actions.index(action)
    counts = counts[action_index]
    return action, counts

def launch_job(f, *args):
    f(*args)

def make_uniform_network():
    return Network()

#####################
# Utility functions #
#####################

def train_network(
    config: AlphaZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, training_iter: int, path: str
):
    """
    Trains the neural network using self-play game data from the replay buffer.

    Parameters:
      - config (AlphaZeroConfig): Configuration settings for AlphaZero.
      - storage (SharedStorage): Object responsible for storing and retrieving neural network checkpoints during training.
      - replay_buffer (ReplayBuffer): Buffer containing self-play games for training.
      - training_iter (int): The current training iteration.
      - path (str): The path to the directory where the model will be saved.

    Returns:
      None
    """
    network = storage.latest_network()
    batch = replay_buffer.sample_batch()
    network.train(batch=batch)
    network.save_model(path + "/" + f"network_{training_iter}")
    storage.save_network(training_iter, network)

def create_directory(base_path: str) -> str:
    """
    Creates a new directory with the current date and time as its name.

    Args:
        base_path (str): The base path where the new directory will be created.

    Returns:
        str: The full path of the created directory.
        
    Example:
        >>> create_directory("/home/user/Documents")
        '/home/user/Documents/2022-01-01_12-00-00'
    """
    now = datetime.now()
    dir_name = now.strftime("%Y-%m-%d_%H-%M-%S")
    full_path = os.path.join(base_path, dir_name)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created directory: {full_path}")

    return full_path
