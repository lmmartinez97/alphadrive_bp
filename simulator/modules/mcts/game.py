
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
        action_history (List[int]): List of actions representing the game history.
        state_history (List[np.array]): List of states representing the game history.
        child_visits (List[List[float]]): Stores the visit count distribution of child nodes for each state in the game.
        num_actions (int): Represents the size of the action space for the game.
    """

    def __init__(self, action_history: List[int] = None, state_history: List[np.array] = None, node_history: List[Node] = None) -> None:
        """
        Initializes a new Game instance.

        Args:
            action_history (List[int], optional): List of actions representing the game history.
            state_history (List[np.array], optional): List of states representing the game history.
            node_history (List[Node], optional): List of nodes representing the game history.
        """
        self.action_history = action_history or []
        self.state_history = state_history or []
        self.node_history = node_history or []
        self.child_visits = []
        self.num_actions = 3  # action space size for chess; 11259 for shogi, 362 for Go
        self.reward_value = 0

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

    def apply(self, action: int, recording: bool, simulation: 'Simulation') -> None:
        """
        Applies an action to the game state.

        Args:
            action (int): The action to be applied.
        """
        simulation.mcts_step(verbose=False, recording=recording, action=action)
        self.action_history.append(action)
        self.state_history.append(simulation.get_state(decision_index=len(self.action_history) - 1))

    def store_search_statistics(self, root: "Node"):
        """
        Stores visit statistics for child nodes.

        Args:
            root (Node): The root node of the search tree.
        """
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append(
            [
                (root.children[a].visit_count / sum_visits) if a in root.children else 0
                for a in range(self.num_actions)
            ]
        )


    def make_image(self, node_index: int) -> List[np.array]:
        """
        Constructs a game-specific feature representation.

        Args:
            node_index (int): The index of the current game state.

        Returns:
            List[float]: List of feature planes representing the game state. Comes from autoencoder
        """
        return self.state_history[node_index]

    def make_target(self, node_index: int) -> Tuple[float, List[float]]:
        """
        Constructs a target tuple for training.

        Args:
            node_index (int): The index of the current game state.

        Returns:
            Tuple[float, List[float]]: Target value and policy for training the neural network.
        """
        return (self.reward_value, self.child_visits[node_index])