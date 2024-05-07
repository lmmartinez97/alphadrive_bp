
## Global imports
from typing import List, Tuple
import numpy as np
import copy

## Local imports
from .node import Node
from ..carla.simulation import Simulation

class Game:
    """
    Represents the state of a game.

    It provides methods for checking the terminal state, getting the terminal value, getting legal actions,
    cloning the game state, applying an action, storing search statistics, making image and target
    for training, and getting the current player.
    
    This method needs to be completed with game-specific logic. 
    
    TODO: Implement missing methods.
          Consider subclassing for different scenarios (e.g. dense lane circulation, merging, roundabout, intersection, etc.)
          Include the simulation attribute to control interaction with CARLA client.
    Attributes:
        node_history (List[Node]): Represents the history of nodes visited during the search.
        num_actions (int): Represents the size of the action space for the game.
    """

    def __init__(self, node_history: List[Node] = None, num_actions: int = 3) -> None:
        """
        Initializes a new Game instance.

        Args:
            node_history (List[Node], optional): List of nodes representing the game history.
            num_actions (int, optional): The size of the action space for the game.
        """
        self.node_history = [] if node_history is None else node_history
        self.num_actions = num_actions  

    def terminal(self, simulation: Simulation) -> bool:
        """
        Checks if the game is in a terminal state.

        Returns:
            bool: True if the game is in a terminal state, False otherwise.
        """
        return simulation.is_terminal()

    def terminal_value(self, simulation: Simulation) -> float:
        """
        Returns the reward associated with the terminal state of the current game.

        Args:
            simulation (Simulation): The simulation object representing the current game state.

        Returns:
            float: The terminal value indicating the outcome or score of the game.
        """
        self.reward_value = simulation.get_reward()
        return self.reward_value

    def legal_actions(self) -> List[int]:
        """
        Returns legal actions at the current state.

        Returns:
            List[int]: List of legal actions.
            
        Available actions:
            - 0: straight
            - 1: shift right
            - 2: shift left
        """
        return [0, 2]

    def clone(self) -> "Game":
        """
        Creates a copy of the game state.

        Returns:
            Game: A new instance representing a copy of the game state.
        """
        return copy.deepcopy(self)

    def apply(self, action: int, recording: bool, simulation: Simulation, node: Node) -> None:
        """
        Applies an action to the game state.

        Args:
            action (int): The action to be applied.
        """
        self.node_history.append(node)
        simulation.mcts_step(verbose=False, recording=recording, action=action)