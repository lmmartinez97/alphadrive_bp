"""Network class for the AlphaZero algorithm."""


from __future__ import division

import numpy as np
import pickle
import tensorflow as tf


from tensorflow import keras
from tensorflow.keras import layers, models

from typing import List, Tuple

# https://stackoverflow.com/questions/44036971/multiple-outputs-in-keras

class Network(object):
    """
    A class used to represent the Neural Network for the AlphaZero algorithm.

    ...

    Attributes
    ----------
    arch : List[int]
        a list of integers representing the architecture of the neural network
    model : keras.Model
        a keras model representing the neural network

    Methods
    -------
    build_model():
        Builds the keras model based on the architecture provided during initialization.

    inference(image: List[numpy.array]) -> Tuple[float, List[float]]:
        Makes a prediction based on the provided image.

    train(batch) -> None:
        Trains the model on the provided batch of data.
    """
    def __init__(self, config: List[int] = None) -> None:
        """
        Constructs all the necessary attributes for the Network object.

        Parameters
        ----------
            config : List[int], optional
                The architecture of the neural network (default is None)
        """
        self.arch = config
        self.model = self.build_model()

    def build_model(self):
        """
        Builds the keras model based on the architecture provided during initialization.

        Returns
        -------
        keras.Model
            The built keras model.
        """
        x = layers.Input(shape=(self.arch[0],))
        input_layer = x
        for units in self.arch[1:]:
            x = layers.Dense(units, activation='relu')(x)

        value = layers.Dense(1, activation=None)(x)
        policy = layers.Dense(3, activation=None)(x)
        policy = layers.Softmax()(policy)

        self.model = models.Model(inputs=input_layer, outputs=[value, policy])
        
        return self.model

    def inference(self, image: List[np.array]) -> Tuple[float, List[float]]:
        """
        Makes a prediction based on the provided image.

        Parameters
        ----------
        image : List[numpy.array]
            The image to make a prediction on.

        Returns
        -------
        Tuple[float, List[float]]
            The predicted value and policy.
        """
        image = np.array(image).reshape(1, self.arch[0])
        value, policy = self.model.predict(image, verbose=0)
        value = np.squeeze(value)
        policy = np.squeeze(policy)
        
        return [value, policy]

    def train(self, batch) -> None:
        """
        Trains the model on the provided batch of data.

        Parameters
        ----------
        batch : List[Tuple]
            The batch of data to train on.
        """
        states, targets = zip(*batch)
        value_targets, policy_targets = zip(*targets)
        
        states = np.array(states)
        value_targets = np.array(value_targets)
        policy_targets = np.array(policy_targets)
        
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



