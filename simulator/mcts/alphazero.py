"""AlphaZero training loop."""


from __future__ import google_type_annotations
from __future__ import division

from helpers import SharedStorage, ReplayBuffer, AlphaZeroConfig
from network import Network
from self_play import run_selfplay
from training import train_network
from utils import launch_job

def alphazero(config: AlphaZeroConfig) -> 'Network':
  """
  The main function that coordinates the AlphaZero training process.

  Args:
    config (AlphaZeroConfig): Configuration settings for AlphaZero.

  Returns:
    'Network': The latest trained neural network.
  """
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for i in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()