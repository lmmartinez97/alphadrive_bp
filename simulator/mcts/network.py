"""Network class for the AlphaZero algorithm."""


from __future__ import google_type_annotations
from __future__ import division

import numpy
import pickle
import tensorflow as tf

from tensorflow import keras

from typing import List, Tuple
from helpers import AlphaZeroConfig, SharedStorage, ReplayBuffer

# https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras

class Network(object):
    def __init__(self, config: List[int] = None) -> None:
        pass

    def inference(self, image: List[numpy.array]) -> Tuple[float, List[float]]:
        pass

    def train(self, batch) -> None:
        pass
    
    def save_model(self, filepath: str) -> bool:
        """
        Saves the current model to a specified file.

        Args:
        filepath (str): The desired path for saving the model.

        Returns:
        bool: True if the model was successfully saved, False otherwise.
        """
        try:
            with open(filepath, "wb") as file:
                # Placeholder: Serialize the current model for saving.
                # pickle.dump(self.current_model, file)
                return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """
        Loads a pre-trained model from a specified file.

        Args:
        filepath (str): The path to the saved model file.

        Returns:
        bool: True if the model was successfully loaded, False otherwise.
        """
        try:
            with open(filepath, "rb") as file:
                loaded_model = pickle.load(file)
                # Placeholder: Assign loaded model to the current instance.
                # self.loaded_model = loaded_model
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


    def save_model(self, filepath: str) -> bool:
        """
        Saves the current model to a specified file.

        Args:
        filepath (str): The desired path for saving the model.

        Returns:
        bool: True if the model was successfully saved, False otherwise.
        """
        try:
            with open(filepath, "wb") as file:
                # Placeholder: Serialize the current model for saving.
                # pickle.dump(self.current_model, file)
                return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False


def train_network(
    config: AlphaZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
    """
    Trains the neural network using self-play game data from the replay buffer.

    Parameters:
      - config (AlphaZeroConfig): Configuration settings for AlphaZero.
      - storage (SharedStorage): Object responsible for storing and retrieving neural network checkpoints during training.
      - replay_buffer (ReplayBuffer): Buffer containing self-play games for training.

    Returns:
      None
    """
    network = Network(config = config.arch_config)
    optimizer = tf.train.MomentumOptimizer(
        config.learning_rate_schedule, config.momentum
    )
    for i in range(config.training_steps):
        if i % config.checkpoint_interval == 0:
            storage.save_network(i, network)
        batch = replay_buffer.sample_batch()
        update_weights(optimizer, network, batch, config.weight_decay)
    storage.save_network(config.training_steps, network)
