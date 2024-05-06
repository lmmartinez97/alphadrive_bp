
## Global Imports
from typing import Dict
from copy import deepcopy

## Local Imports
from .network import Network

class SharedStorage:
    """
    A shared storage for keeping track of neural network checkpoints during training.

    Attributes:
        _networks (Dict[int, 'Network']): A dictionary mapping training steps to network checkpoints.

    Methods:
        latest_network() -> 'Network':
            Retrieves the most recent network checkpoint. If no checkpoints are stored, returns a placeholder.

        save_network(step: int, network: 'Network') -> None:
            Stores a network checkpoint at a specified training step.
    """

    def __init__(self) -> None:
        """
        Initializes a new SharedStorage instance with an empty dictionary for storing network checkpoints.
        """
        self._networks = {}

    def latest_network(self) -> Network:
        """
        Retrieves the most recent network checkpoint.

        If no checkpoints are stored, returns a placeholder with a uniform policy and a value of 0.5.

        Returns:
            'Network': The most recent network checkpoint, or a placeholder if no checkpoints are stored.
        """
        if self._networks:
            # Return a copy of the most recent network so that it can be modified without affecting the original.
            return deepcopy(self._networks[max(self._networks.keys())])
        else:
            # Placeholder: Policy -> uniform, value -> 0.5
            return (0.5, [1/3 for _ in range(3)])

    def save_network(self, step: int, network: Network) -> None:
        """
        Stores a network checkpoint at a specified training step.

        Args:
            step (int): The training step at which the checkpoint is saved.
            network ('Network'): The network checkpoint to be saved.
        """
        self._networks[step] = network
