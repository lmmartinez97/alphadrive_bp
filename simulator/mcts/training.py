"""Helper classes for the AlphaZero algorithm."""


from __future__ import google_type_annotations
from __future__ import division

import tensorflow as tf

from helpers import AlphaZeroConfig, SharedStorage, ReplayBuffer
from network import Network

def train_network(config: AlphaZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
  """
  Trains the neural network using self-play game data from the replay buffer.

  Parameters:
    - config (AlphaZeroConfig): Configuration settings for AlphaZero.
    - storage (SharedStorage): Object responsible for storing and retrieving neural network checkpoints during training.
    - replay_buffer (ReplayBuffer): Buffer containing self-play games for training.

  Returns:
    None
  """
  network = Network()
  optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule, config.momentum)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
  """
  Updates the weights of the neural network based on the training batch.

  Parameters:
    - `optimizer` (tf.train.Optimizer): TensorFlow optimizer for weight updates.
    - `network` (Network): Neural network instance to be updated.
    - `batch` (List[Tuple[List[numpy.array], Tuple[float, List[float]]]]): Training batch containing game states and target values.
    - `weight_decay` (float): Coefficient for L2 regularization.

  Returns:
    None
  """
  loss = 0
  for image, (target_value, target_policy) in batch:
    value, policy_logits = network.inference(image)
    loss += (
        tf.losses.mean_squared_error(value, target_value) +
        tf.nn.softmax_cross_entropy_with_logits(
            logits=policy_logits, labels=target_policy))

  for weights in network.get_weights():
    loss += weight_decay * tf.nn.l2_loss(weights)

  optimizer.minimize(loss)