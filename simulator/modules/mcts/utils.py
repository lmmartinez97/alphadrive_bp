"""Utility functions for the AlphaZero algorithm."""


from __future__ import division

import os
import datetime
from network import Network


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
    return 0, 0


def launch_job(f, *args):
    f(*args)


def make_uniform_network():
    return Network()

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

def create_directory(base_path):
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    dir_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the full path to the new directory
    full_path = os.path.join(base_path, dir_name)

    # Create the new directory
    os.makedirs(full_path, exist_ok=True)

    return full_path
