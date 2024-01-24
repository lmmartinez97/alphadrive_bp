"""Network class for the AlphaZero algorithm."""


from __future__ import google_type_annotations
from __future__ import division

import numpy
import pickle
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers, models

from typing import List, Tuple
from helpers import AlphaZeroConfig, SharedStorage, ReplayBuffer

# https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras

class Network(object):
    def __init__(self, config: List[int] = None) -> None:
        
        self.arch = config
        self.model = self.build_model()

    def build_model(self):
        x = layers.Input(shape=(self.arch[0],))
        for units in self.arch[1:]:
            x = layers.Dense(units, activation='relu')(x)

        value = layers.Dense(1, activation=None)(x)
        policy = layers.Dense(3, activation=None)(x)

        self.model = models.Model(inputs=x, outputs=[value, policy])
        
        return self.model

    def inference(self, image: List[numpy.array]) -> Tuple[float, List[float]]:
        return self.model.predict(image)

    def train(self, batch) -> None:
        states, targets = zip(*batch)
        value_targets, policy_targets = zip(*targets)
        
        states = numpy.array(states)
        value_targets = numpy.array(value_targets)
        policy_targets = numpy.array(policy_targets)
        
        self.model.compile(
            optimizer='adam',
            loss=['mean_squared_error', 'categorical_crossentropy']
        )
    
        self.model.fit(states, [value_targets, policy_targets], verbose=1)
    
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
            self.model.save(filepath)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False


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
