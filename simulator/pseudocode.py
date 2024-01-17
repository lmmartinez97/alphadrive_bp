"""Pseudocode description of the AlphaZero algorithm."""


from __future__ import google_type_annotations
from __future__ import division

import math
import numpy
import pickle
import tensorflow as tf
from typing import List, Tuple

##########################
####### Helpers ##########


class AlphaZeroConfig(object):

  def __init__(self):
    ### Self-Play
    self.num_actors = 5000

    self.num_sampling_moves = 30
    self.max_moves = 512  # for chess and shogi, 722 for Go.
    self.num_simulations = 800

    # Root prior exploration noise.
    self.root_dirichlet_alpha = 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
    self.root_exploration_fraction = 0.25

    # UCB formula
    self.pb_c_base = 19652
    self.pb_c_init = 1.25

    ### Training
    self.training_steps = int(700e3)
    self.checkpoint_interval = int(1e3)
    self.window_size = int(1e6)
    self.batch_size = 4096

    self.weight_decay = 1e-4
    self.momentum = 0.9
    # Schedule for chess and shogi, Go starts at 2e-2 immediately.
    self.learning_rate_schedule = {
        0: 2e-1,
        100e3: 2e-2,
        300e3: 2e-3,
        500e3: 2e-4
    }


class Node(object):

  def __init__(self, prior: float):
    self.visit_count = 0
    self.to_play = -1
    self.prior = prior
    self.value_sum = 0
    self.children = {}

  def expanded(self) -> bool:
    return len(self.children) > 0

  def value(self) -> float:
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
    self.num_actions = 4672  # action space size for chess; 11259 for shogi, 362 for Go

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

  def clone(self) -> 'Game':
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
    #self.history.append(action)
    pass

  def store_search_statistics(self, root: 'Node'):
    """
    Stores visit statistics for child nodes.

    Args:
        root (Node): The root node of the search tree.
    """
    sum_visits = sum(child.visit_count for child in root.children.itervalues())
    self.child_visits.append([
        root.children[a].visit_count / sum_visits if a in root.children else 0
        for a in range(self.num_actions)
    ])

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
    return (self.terminal_value(state_index % 2),
            self.child_visits[state_index])

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

  def __init__(self, config: 'AlphaZeroConfig'):
    """
    Initializes a new ReplayBuffer instance.

    Args:
      config (AlphaZeroConfig): Configuration object containing parameters.
    """
    self.window_size = config.window_size
    self.batch_size = config.batch_size
    self.buffer = []

  def save_game(self, game: 'Game'):
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
        p=[len(g.history) / move_sum for g in self.buffer]
    )
    game_pos = [(g, numpy.random.randint(len(g.history))) for g in games]
    return [(g.make_image(i), g.make_target(i)) for (g, i) in game_pos]

class Network(object):
  """
  A placeholder for the neural network used in AlphaZero.

  Methods:
    inference(image: List[numpy.array]) -> Tuple[float, List[float]]:
        Performs inference on the input image and returns the value and policy.

    get_weights() -> List:
        Returns the weights of the neural network.

    load_model(filepath: str) -> bool:
        Loads a pre-trained model from a specified file.

    save_model(filepath: str) -> bool:
        Saves the current model to a specified file.
  """

  def inference(self, image: List[numpy.array]) -> Tuple[float, List[float]]:
    """
    Performs inference on the input image and returns the value and policy.

    Args:
      image (List[numpy.array]): The input image, a representation of the game state.

    Returns:
      Tuple[float, List[float]]:
          A tuple containing the predicted value (expected outcome) and policy (action probabilities).
    """
    return (-1, [])  # Placeholder for the actual implementation.

  def get_weights(self) -> List:
    """
    Returns the weights of the neural network.

    Returns:
      List: The weights of the neural network.
    """
    # Placeholder for the actual implementation.
    return []

  def load_model(self, filepath: str) -> bool:
    """
    Loads a pre-trained model from a specified file.

    Args:
      filepath (str): The path to the saved model file.

    Returns:
      bool: True if the model was successfully loaded, False otherwise.
    """
    try:
        with open(filepath, 'rb') as file:
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
        with open(filepath, 'wb') as file:
            # Placeholder: Serialize the current model for saving.
            # pickle.dump(self.current_model, file)
            return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False


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

  def latest_network(self) -> 'Network':
    """
    Returns the latest stored network checkpoint.

    Returns:
      'Network': The latest stored network checkpoint.
    """
    if self._networks:
        return self._networks[max(self._networks.keys())]
    else:
        return make_uniform_network()  # Placeholder: Policy -> uniform, value -> 0.5

  def save_network(self, step: int, network: 'Network') -> None:
    """
    Saves a network checkpoint at a specified training step.

    Args:
      step (int): The training step at which the checkpoint is saved.
      network ('Network'): The network checkpoint to be saved.

    Returns:
      None
    """
    self._networks[step] = network


##### End Helpers ########
##########################


# AlphaZero training is split into two independent parts: Network training and
# self-play data generation.
# These two parts only communicate by transferring the latest network checkpoint
# from the training to the self-play, and the finished games from the self-play
# to the training.
def alphazero(config: AlphaZeroConfig):
  storage = SharedStorage()
  replay_buffer = ReplayBuffer(config)

  for i in range(config.num_actors):
    launch_job(run_selfplay, config, storage, replay_buffer)

  train_network(config, storage, replay_buffer)

  return storage.latest_network()


##################################
####### Part 1: Self-Play ########


# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(config: AlphaZeroConfig, storage: SharedStorage,
                 replay_buffer: ReplayBuffer):
  while True:
    network = storage.latest_network()
    game = play_game(config, network)
    replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network):
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
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network):
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


def select_action(config: AlphaZeroConfig, game: Game, root: Node):
  visit_counts = [(child.visit_count, action)
                  for action, child in root.children.iteritems()]
  if len(game.history) < config.num_sampling_moves:
    _, action = softmax_sample(visit_counts)
  else:
    _, action = max(visit_counts)
  return action


# Select the child with the highest UCB score.
def select_child(config: AlphaZeroConfig, node: Node):
  _, action, child = max((ucb_score(config, node, child), action, child)
                         for action, child in node.children.iteritems())
  return action, child


# The score for a node is based on its value, plus an exploration bonus based on
# the prior.
def ucb_score(config: AlphaZeroConfig, parent: Node, child: Node):
  pb_c = math.log((parent.visit_count + config.pb_c_base + 1) /
                  config.pb_c_base) + config.pb_c_init
  pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

  prior_score = pb_c * child.prior
  value_score = child.value()
  return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, game: Game, network: Network):
  value, policy_logits = network.inference(game.make_image(-1))

  # Expand the node.
  node.to_play = game.to_play()
  policy = {a: math.exp(policy_logits[a]) for a in game.legal_actions()}
  policy_sum = sum(policy.itervalues())
  for action, p in policy.iteritems():
    node.children[action] = Node(p / policy_sum)
  return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float, to_play):
  for node in search_path:
    node.value_sum += value if node.to_play == to_play else (1 - value)
    node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
  actions = node.children.keys()
  noise = numpy.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
  frac = config.root_exploration_fraction
  for a, n in zip(actions, noise):
    node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac


######### End Self-Play ##########
##################################

##################################
####### Part 2: Training #########


def train_network(config: AlphaZeroConfig, storage: SharedStorage,
                  replay_buffer: ReplayBuffer):
  network = Network()
  optimizer = tf.train.MomentumOptimizer(config.learning_rate_schedule,
                                         config.momentum)
  for i in range(config.training_steps):
    if i % config.checkpoint_interval == 0:
      storage.save_network(i, network)
    batch = replay_buffer.sample_batch()
    update_weights(optimizer, network, batch, config.weight_decay)
  storage.save_network(config.training_steps, network)


def update_weights(optimizer: tf.train.Optimizer, network: Network, batch,
                   weight_decay: float):
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


######### End Training ###########
##################################

################################################################################
############################# End of pseudocode ################################
################################################################################


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
  return 0, 0


def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return Network()
