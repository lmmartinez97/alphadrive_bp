#############################################
# Title: AlphaZero Helpers
# Author: Luis Miguel Martinez
# Date: Feb - 2024
# Notes: Improved from original pseudocode by DeepMind.
#############################################

"""Helper classes for the AlphaZero algorithm."""

from __future__ import division

import copy
import numpy as np

from copy import deepcopy
from rich import print
from typing import Dict, List, Union, Tuple

from .network import Network
from ..carla.simulation import Simulation

##########################
####### Helpers ##########

class AlphaZeroConfig:
    """
    A configuration class for the AlphaZero algorithm.

    This class takes a dictionary of configuration parameters and sets up the 
    necessary attributes for the AlphaZero algorithm. The configuration parameters 
    include parameters for self-play and training.

    Attributes:
        num_actors (int): The number of actors for the self-play process.
        num_sampling_moves: The number of moves up to which actions are decided probabilisticly 
                            -instead of greedily- in the self-play process.       
        max_moves (int): The maximum number of moves in a game.
        num_simulations (int): The number of Monte Carlo Tree Search simulations to run for each move.
        root_dirichlet_alpha (float): Parameter for exploration during MCTS.
        root_exploration_fraction (float): Parameter for exploration during MCTS.
        pb_c_base (int): Parameter for the PUCT algorithm.
        pb_c_init (float): Parameter for the PUCT algorithm.
        network_arch (List[int]): The architecture of the neural network.
        training_steps (int): The number of training steps to run.
        checkpoint_interval (int): The interval at which to save checkpoints.
        training_iterations (int): The number of training iterations to run.
        games_per_iteration (int): The number of games to play in each iteration.
        window_size (int): The size of the window for the replay buffer.
        batch_size (int): The size of the batch for training the network.
        weight_decay (float): Parameter for the optimizer.
        momentum (float): Parameter for the optimizer.
        learning_rate_schedule (Dict[int, float]): The learning rate schedule for training.
    """

    def __init__(self, config_dict: Dict[str, Union[int, float, List[int], Dict[int, float]]]) -> None:
        # Self-Play
        self.num_actors: int = config_dict.get("num_actors", 5000)
        self.num_sampling_moves: int = config_dict.get("num_sampling_moves", 30)
        self.max_moves: int = config_dict.get("max_moves", 512)
        self.num_simulations: int = config_dict.get("num_simulations", 800)
        self.root_dirichlet_alpha: float = config_dict.get("root_dirichlet_alpha", 0.3)
        self.root_exploration_fraction: float = config_dict.get("root_exploration_fraction", 0.25)
        self.pb_c_base: int = config_dict.get("pb_c_base", 19652)
        self.pb_c_init: float = config_dict.get("pb_c_init", 1.25)

        # Training
        self.network_arch: List[int] = config_dict.get("network_arch", [200, 128, 64, 16])  # input layer, hidden layers, NO output layer
        self.training_steps: int = config_dict.get("training_steps", int(700e3))
        self.checkpoint_interval: int = config_dict.get("checkpoint_interval", int(1e3))
        self.training_iterations: int = config_dict.get("training_iterations", 60)  # added attribute
        self.games_per_iteration: int = config_dict.get("games_per_iteration", 50)  # added attribute
        self.window_size: int = config_dict.get("window_size", int(1e6))
        self.batch_size: int = config_dict.get("batch_size", 4096)
        self.weight_decay: float = config_dict.get("weight_decay", 1e-4)
        self.momentum: float = config_dict.get("momentum", 0.9)
        self.learning_rate_schedule: Dict[int, float] = config_dict.get("learning_rate_schedule", {0: 2e-1, 100e3: 2e-2, 300e3: 2e-3, 500e3: 2e-4})

class Node(object):
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) algorithm.

    Attributes:
      - visit_count (int): Number of times the node has been visited during the search.
      - to_play (int): Player to play at the node.
      - prior (float): Prior probability of selecting the node.
      - value_sum (float): Sum of values encountered during the search.
      - children (Dict[int, Node]): Child nodes of the current node.

    Methods:
      - expanded -> bool: Checks if the node has been expanded (has children).
      - value -> float: Returns the average value of the node.
    """

    def __init__(self, prior: float) -> None:
        """
        Initializes a new Node with the given prior probability.

        Args:
            prior (float): Prior probability of selecting the node.
        """
        self.visit_count: int = 0
        self.state: List[float] = []
        self.prior: float = prior
        self.value_sum: float = 0
        self.children: Dict[int, 'Node'] = {}

    def expanded(self) -> bool:
        """
        Checks if the node has been expanded (has children).

        Returns:
            bool: True if the node has children, False otherwise.
        """
        return len(self.children) > 0
    
    def assign_state(self, state: List[float]) -> None:
        """
        Assigns the state to the node.

        Args:
            state (List[float]): The state to be assigned to the node.
        """
        self.state = state

    def value(self) -> float:
        """
        Returns the average value of the node.

        Returns:
            float: The average value of the node.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

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
        return [0, 1, 2]

    def clone(self) -> "Game":
        """
        Creates a copy of the game state.

        Returns:
            Game: A new instance representing a copy of the game state.
        """
        return copy.deepcopy(self)

    def apply(self, action: int, recording: bool, simulation: 'Simulation', node: Node) -> None:
        """
        Applies an action to the game state.

        Args:
            action (int): The action to be applied.
        """
        simulation.mcts_step(verbose=False, recording=recording, action=action)
        self.action_history.append(action)
        self.state_history.append(simulation.get_state(decision_index=len(self.action_history) - 1))
        self.node_history.append(node)

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
    
class ReplayBuffer:
    """
    A replay buffer for storing and sampling self-play game data.

    Attributes:
        window_size (int): The maximum size of the replay buffer.
            When the buffer exceeds this size, old games are discarded.
        batch_size (int): The size of batches to be sampled during training.
        buffer (List[Game]): A list to store self-play games.

    Methods:
        save_game(game: Game): Saves a self-play game to the replay buffer.
        sample_batch() -> List[Tuple[List[np.array], Tuple[float, List[float]]]]:
            Samples a batch of self-play game data for training.
    """

    def __init__(self, config: "AlphaZeroConfig") -> None:
        """
        Initializes a new ReplayBuffer instance.

        Args:
            config (AlphaZeroConfig): Configuration object containing parameters.
        """
        self.window_size: int = config.window_size
        self.batch_size: int = config.batch_size
        self.buffer: List[Game] = []

    def save_game(self, game: "Game") -> None:
        """
        Saves a self-play game to the replay buffer.

        Args:
            game (Game): The self-play game to be saved.

        Notes:
            If the buffer exceeds the maximum window size, old games are discarded.
        """
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self) -> List[Tuple[List[np.array], Tuple[float, List[float]]]]:
        """
        Samples a batch of self-play game data for training.

        Returns:
            List[Tuple[List[np.array], Tuple[float, List[float]]]]:
                A list of tuples containing game states (images) and their target values (value, policy).
        """
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.action_history) for g in self.buffer))
        games = np.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.action_history) / move_sum for g in self.buffer],
        )
        game_pos = [(g, np.random.randint(len(g.action_history))) for g in games]
        return [(g.make_image(node_index=i), g.make_target(node_index = i)) for (g, i) in game_pos]


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

    def latest_network(self) -> "Network":
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

    def save_network(self, step: int, network: "Network") -> None:
        """
        Stores a network checkpoint at a specified training step.

        Args:
            step (int): The training step at which the checkpoint is saved.
            network ('Network'): The network checkpoint to be saved.
        """
        self._networks[step] = network
