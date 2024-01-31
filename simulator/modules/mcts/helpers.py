"""Helper classes for the AlphaZero algorithm."""


from __future__ import division

import numpy
import tensorflow as tf

from copy import deepcopy
from network import Network
from typing import List, Tuple
from utils import make_uniform_network

##########################
####### Helpers ##########


class AlphaZeroConfig(object):
    def __init__(self, config_dict):
        # Self-Play
        self.num_actors = config_dict.get("num_actors", 5000)
        self.num_sampling_moves = config_dict.get("num_sampling_moves", 30)
        self.max_moves = config_dict.get("max_moves", 512)
        self.num_simulations = config_dict.get("num_simulations", 800)
        self.root_dirichlet_alpha = config_dict.get("root_dirichlet_alpha", 0.3)
        self.root_exploration_fraction = config_dict.get(
            "root_exploration_fraction", 0.25
        )
        self.pb_c_base = config_dict.get("pb_c_base", 19652)
        self.pb_c_init = config_dict.get("pb_c_init", 1.25)

        # Training
        self.network_arch = config_dict.get(
            "network_arch", [200, 128, 64, 16]
        )  # input layer, hidden layers, NO output layer
        self.training_steps = config_dict.get("training_steps", int(700e3))
        self.checkpoint_interval = config_dict.get("checkpoint_interval", int(1e3))
        self.training_iterations = config_dict.get(
            "training_iterations", 60
        )  # added attribute
        self.games_per_iteration = config_dict.get(
            "games_per_iteration", 50
        )  # added attribute
        self.window_size = config_dict.get("window_size", int(1e6))
        self.batch_size = config_dict.get("batch_size", 4096)
        self.weight_decay = config_dict.get("weight_decay", 1e-4)
        self.momentum = config_dict.get("momentum", 0.9)
        self.learning_rate_schedule = config_dict.get(
            "learning_rate_schedule", {0: 2e-1, 100e3: 2e-2, 300e3: 2e-3, 500e3: 2e-4}
        )


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

    def __init__(self, prior: float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self) -> bool:
        """
        Checks if the node has been expanded (has children).

        Returns:
            bool: True if the node has children, False otherwise.
        """
        return len(self.children) > 0

    def value(self) -> float:
        """
        Returns the average value of the node.

        Returns:
            float: The average value of the node.
        """
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


class Game(object):
    """
    Represents the state of the game.

    Attributes:
      history (List[int]): List of actions representing the game history.
          It records the sequence of actions taken during the game.
      child_visits (List[List[float]]): Stores the visit count distribution
          of child nodes for each state in the game.
      num_actions (int): Represents the size of the action space for the game.
          It is the total number of possible actions that can be taken by a player.
    """

    def __init__(self, history: List[int] = None):
        """
        Initializes a new Game instance.

        Args:
            history (List[int], optional): List of actions representing the game history.
                Defaults to an empty list.
        """
        self.history = history or []
        self.child_visits = []
        self.num_actions = (
            4672  # action space size for chess; 11259 for shogi, 362 for Go
        )

    def terminal(self) -> bool:
        """
        Checks if the game is in a terminal state.

        Returns:
            bool: True if the game is in a terminal state, False otherwise.
        """
        # Game specific termination rules.
        pass

    def terminal_value(self, to_play: int) -> float:
        # Game specific value.
        """
        Returns the reward associated with the terminal state of the current game.

        Args:
            to_play (int): The player to play at the terminal state.

        Returns:
            float: The terminal value indicating the outcome or score of the game.
        """
        pass

    def legal_actions(self) -> List[int]:
        # Game specific calculation of legal actions.
        """
        Returns legal actions at the current state.

        Returns:
            List[int]: List of legal actions.
        """
        return []

    def clone(self) -> "Game":
        """
        Creates a copy of the game state.

        Returns:
            Game: A new instance representing a copy of the game state.
        """
        return Game(list(self.history))

    def apply(self, action: int):
        """
        Applies an action to the game state.

        Args:
            action (int): The action to be applied.

        Notes:
            This method interacts with the Carla client to execute the action
            and updates the game state based on the client's response.
        """
        # self.history.append(action)
        pass

    def store_search_statistics(self, root: "Node"):
        """
        Stores visit statistics for child nodes.

        Args:
            root (Node): The root node of the search tree.
        """
        sum_visits = sum(child.visit_count for child in root.children.itervalues())
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in range(self.num_actions)
            ]
        )

    def make_image(self, state_index: int) -> List[numpy.array]:
        """
        Constructs a game-specific feature representation.

        Args:
            state_index (int): The index of the current game state.

        Returns:
            List[numpy.array]: List of feature planes representing the game state.
        """
        # Game specific feature planes.
        return []

    def make_target(self, state_index: int) -> Tuple[float, List[float]]:
        """
        Constructs a target tuple for training.

        Args:
            state_index (int): The index of the current game state.

        Returns:
            Tuple[float, List[float]]: Target value and policy for training the neural network.
        """
        return (self.terminal_value(state_index % 2), self.child_visits[state_index])

    def to_play(self) -> int:
        return len(self.history) % 2


class ReplayBuffer(object):
    """
    A replay buffer for storing and sampling self-play game data.

    Attributes:
      window_size (int): The maximum size of the replay buffer.
          When the buffer exceeds this size, old games are discarded.
      batch_size (int): The size of batches to be sampled during training.
      buffer (List[Game]): A list to store self-play games.
    """

    def __init__(self, config: "AlphaZeroConfig"):
        """
        Initializes a new ReplayBuffer instance.

        Args:
          config (AlphaZeroConfig): Configuration object containing parameters.
        """
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game: "Game"):
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

    def sample_batch(self) -> List[Tuple[List[numpy.array], Tuple[float, List[float]]]]:
        """
        Samples a batch of self-play game data for training.

        Returns:
          List[Tuple[List[numpy.array], Tuple[float, List[float]]]]:
              A list of tuples containing game states (images) and their target values (value, policy).
        """
        # Sample uniformly across positions.
        move_sum = float(sum(len(g.history) for g in self.buffer))
        games = numpy.random.choice(
            self.buffer,
            size=self.batch_size,
            p=[len(g.history) / move_sum for g in self.buffer],
        )
        game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
        return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]


class SharedStorage(object):
    """
    A shared storage for keeping track of neural network checkpoints.
    Attributes:
      _networks (Dict[int, 'Network']): A dictionary to store network checkpoints with training steps as keys.
    Methods:
      latest_network() -> 'Network':
          Returns the latest stored network checkpoint.

      save_network(step: int, network: 'Network') -> None:
          Saves a network checkpoint at a specified training step.
    """

    def __init__(self):
        self._networks = {}

    def latest_network(self) -> "Network":
        """
        Returns the latest stored network checkpoint.

        Returns:
          'Network': The latest stored network checkpoint.
        """
        if self._networks:
            return deepcopy(self._networks[max(self._networks.keys())]) #return copy of Network so that it can be modified without affecting the original
        else:
            return (
                (0.5, [1/3 for _ in range(3)]),
            )  # Placeholder: Policy -> uniform, value -> 0.5

    def save_network(self, step: int, network: "Network") -> None:
        """
        Saves a network checkpoint at a specified training step.

        Args:
          step (int): The training step at which the checkpoint is saved.
          network ('Network'): The network checkpoint to be saved.

        Returns:
          None
        """
        self._networks[step] = network
