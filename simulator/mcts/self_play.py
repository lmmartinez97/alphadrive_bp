"""Selfplay functions for the AlphaZero algorithm."""


from __future__ import google_type_annotations
from __future__ import division

import math
import numpy

from helpers import SharedStorage, ReplayBuffer, AlphaZeroConfig, Game, Node
from network import Network
from typing import List, Tuple
from utils import softmax_sample

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer):
  """
    Continuously runs self-play to generate game data for training.

    Parameters:
      - config: Instance of the AlphaZeroConfig class that contains parameters for execution and training.
      - storage: Object responsible for storing and retrieving neural network checkpoints during training.
      - replay_buffer: Buffer for storing self-play games to be used in training.

  """
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network) -> 'Game':
  """
    Plays a single game using Monte Carlo Tree Search (MCTS).

    Args:
      - config: Instance of the AlphaZeroConfig class containing parameters for execution and training.
      - network: Instance of the Network class representing the current neural network model.

    Returns:
      - game: The final state of the game after completing the self-play.

  """
  game = Game()
  while not game.terminal() and len(game.history) < config.max_moves:
    action, root = run_mcts(config, game, network)
    game.apply(action)
    game.store_search_statistics(root)
  return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network) -> Tuple[int, Node]:
  """
  Runs the Monte Carlo Tree Search (MCTS) algorithm to select the best action.

  Args:
    config (AlphaZeroConfig): Configuration settings for AlphaZero.
    game (Game): The current state of the game.
    network (Network): The neural network used for value and policy predictions.

  Returns:
    Tuple[int, Node]: The selected action and the root node of the search tree.
  """
  root = Node(0)
  evaluate(root, game, network)
  add_exploration_noise(config, root)

  for _ in range(config.num_simulations):
    node = root
    scratch_game = game.clone()
    search_path = [node]

    while node.expanded():
      action, node = select_child(config, node)
      scratch_game.apply(action)
      search_path.append(node)

    value = evaluate(node, scratch_game, network)
    backpropagate(search_path, value, scratch_game.to_play())
  return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node) -> int:
  """
  Selects the action to take based on the current game state and the MCTS search results.

  Args:
    config (AlphaZeroConfig): Configuration settings for AlphaZero.
    game (Game): The current state of the game.
    root (Node): The root node of the MCTS search tree.

  Returns:
    action: The selected action to take.
  """
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.items()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action


# Select the child with the highest UCB score.

def select_child(config: AlphaZeroConfig, node: Node) -> Tuple[int, Node]:
  """
  Selects the child node with the highest UCB (Upper Confidence Bound) score.

  Args:
      config (AlphaZeroConfig): Configuration settings for AlphaZero.
      node (Node): The parent node from which to select a child.

  Returns:
      Tuple[int, Node]: The selected action and the corresponding child node.
  """
  _, action, child = max(
      (ucb_score(config, node, child), action, child)
      for action, child in node.children.items()
  )
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node) -> float:
  """
  Calculates the Upper Confidence Bound (UCB) score for a child node in the Monte Carlo Tree Search (MCTS) algorithm.

  Args:
    - config: (AlphaZeroConfig): Configuration settings for AlphaZero.
    - parent: (Node): The parent node in the search tree.
    - child: (Node): The child node for which the UCB score is calculated.

  Returns:
    - float: The UCB score for the child node.
  """
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network) -> float:
  """
  Evaluate the given node in the Monte Carlo Tree Search using the neural network.

  Args:
      node (Node): The node to be evaluated.
      game (Game): The current state of the game.
      network (Network): The neural network used for value and policy predictions.

  Returns:
      float: The value predicted by the neural network for the given game state.
  """
  value, policy_logits = network.inference(game.make_image(-1))
  node.to_play = game.to_play()
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
  policy_sum = sum(policy.values())
  for action, p in policy.items():
    node.children[action] = Node(p / policy_sum)
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  """
  Backpropagates the evaluation value through the Monte Carlo Tree Search (MCTS) tree.

  Args:
      - search_path (List[Node]): List of nodes representing the search path from the leaf to the root.
      - value (float): The evaluation value to be backpropagated.
      - to_play: The player to play at the terminal state.

  Returns:
      None

  """
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  """
  Adds Dirichlet noise to the prior of the root to encourage exploration.

  Args:
    - config (AlphaZeroConfig): Configuration settings for AlphaZero.
    - node (Node): The root node of the MCTS search tree.

  Return:
    - None
  """
  actions = list(node.children.keys())
  noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac