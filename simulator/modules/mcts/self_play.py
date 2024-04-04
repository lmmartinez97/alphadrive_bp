"""Selfplay functions for the AlphaZero algorithm."""


from __future__ import division

import numpy as np

from rich import print
from typing import List, Tuple

from .helpers import SharedStorage, ReplayBuffer, AlphaZeroConfig, Game, Node
from .network import Network
from .utils import softmax_sample
from ..carla.simulation import Simulation

# Each self-play job is independent of all others; it takes the latest network
# snapshot, produces a game and makes it available to the training job by
# writing it to a shared replay buffer.
def run_selfplay(
    config: AlphaZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer, simulation: Simulation,
):
    """
    Continuously runs self-play to generate game data for training.

    Parameters:
        config (AlphaZeroConfig): Instance of the AlphaZeroConfig class that contains parameters for execution and training.
        storage (SharedStorage): Object responsible for storing and retrieving neural network checkpoints during training.
        replay_buffer (ReplayBuffer): Buffer for storing self-play games to be used in training.
        simulation (Simulation): Instance of the Simulation class that represents the game.

    """
    network = storage.latest_network()
    for i in range(config.games_per_iteration):
        print(f"Starting game {i+1}/{config.games_per_iteration}")
        simulation.init_game()
        game = play_game(config, network, simulation)
        replay_buffer.save_game(game)


# Each game is produced by starting at the initial board position, then
# repeatedly executing a Monte Carlo Tree Search to generate moves until the end
# of the game is reached.
def play_game(config: AlphaZeroConfig, network: Network, simulation: Simulation) -> "Game":
    """
    Plays a single game using Monte Carlo Tree Search (MCTS).

    Args:
        config (AlphaZeroConfig): Instance of the AlphaZeroConfig class containing parameters for execution and training.
        network (Network): Instance of the Network class representing the current neural network model.
        simulation (Simulation): Instance of the Simulation class that represents the game simulation.

    Returns:
        Game: The final state of the game after completing the self-play.
    """
    game = Game()
    while not game.terminal(simulation=simulation) and len(game.action_history) < config.max_moves:
        print(f"Playing game - move {len(game.action_history)+1}/{config.max_moves}")
        action, root = run_mcts(config, game, network, simulation)
        game.apply(action=action, simulation=simulation, recording=True)
        game.store_search_statistics(root)
    return game


# Core Monte Carlo Tree Search algorithm.
# To decide on an action, we run N simulations, always starting at the root of
# the search tree and traversing the tree according to the UCB formula until we
# reach a leaf node.
def run_mcts(config: AlphaZeroConfig, game: Game, network: Network, simulation: Simulation) -> Tuple[int, Node]:
    """
    Runs the Monte Carlo Tree Search (MCTS) algorithm to select the best action.

    Args:
        config (AlphaZeroConfig): Configuration settings for AlphaZero.
        game (Game): The current state of the game.
        network (Network): The neural network used for value and policy predictions.
        simulation (Simulation): Instance of the Simulation class that represents the game simulation.

    Returns:
        Tuple[int, Node]: The selected action and the root node of the search tree.
    """
    # Initialize the root node
    root = Node(0)
    root.assign_state(simulation.get_state(decision_index=len(game.action_history)))
    
    # Evaluate the root node
    evaluate(root, network)
    
    # Add exploration noise to the root node
    add_exploration_noise(config, root)

    # Run the specified number of simulations
    for i in range(config.num_simulations):
        print(f"Running simulation {i+1}/{config.num_simulations} in monte carlo tree search.")
        node = root
        scratch_game = game.clone()
        node.assign_state(simulation.get_state(decision_index=len(game.action_history)))
        search_path = [node]

        # Traverse the tree until we reach either a terminal state, or the maximum number of moves allowed for mock game, or an unexpanded node
        while node.expanded() and not scratch_game.terminal(simulation=simulation) and len(scratch_game.action_history) < config.max_moves:
            action, node = select_child(config, node)
            scratch_game.apply(action=action, simulation=simulation, recording=True)
            node.assign_state(simulation.get_state(decision_index=len(scratch_game.action_history)))
            search_path.append(node)
            #evaluate new node to generate children
            value = evaluate(node, network)

        #return simulation to the original state - before MCTS started
        print(f"Restoring game state to {len(game.action_history)}")
        simulation.state_manager.restore_frame(frame_number=len(game.action_history), vehicle_list=simulation.world.actor_list) #we restore to frame of main game, NOT mock game for mcts
        simulation.decision_counter = len(game.action_history) #decision counter has increased with MCTS mock game, so we need to reset it to the main game decision counter
        
        # Evaluate the leaf node and propagate the value back up the search path
        backpropagate(search_path, value)
        

    # Select the best action from the root node
    return select_action(config, game, root), root


def select_action(config: AlphaZeroConfig, game: Game, root: Node) -> int:
    """
    Selects the action to take based on the current game state and the MCTS search results.

    During the initial phase of the game (as defined by `num_sampling_moves` in the config), 
    actions are selected probabilistically based on the visit counts of the root's children, 
    using a softmax function to give a higher probability to actions with higher visit counts.

    After this initial phase, the action with the highest visit count is always selected.

    Args:
        config (AlphaZeroConfig): Configuration settings for AlphaZero.
        game (Game): The current state of the game.
        root (Node): The root node of the MCTS search tree.

    Returns:
        action: The selected action to take.
    """
    visit_counts = [
        (child.visit_count, action) for action, child in root.children.items()
    ]
    if len(game.action_history) < config.num_sampling_moves:
        _, action = softmax_sample(visit_counts=visit_counts)
    else:
        _, action = max(visit_counts)
    return action


# Select the child with the highest UCB score.


def select_child(config: AlphaZeroConfig, node: Node) -> Tuple[int, Node]:
    """
    Selects the child node with the highest UCB (Upper Confidence Bound) score.

    The UCB score is a measure used in the Monte Carlo Tree Search (MCTS) algorithm to balance 
    exploration and exploitation when selecting the next node to visit. It takes into account 
    both the estimated value of a node (exploitation) and the uncertainty about that estimate 
    (exploration).

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

    The UCB score is a measure used in the MCTS algorithm to balance exploration and exploitation when selecting the next node to visit. 
    It takes into account both the estimated value of a node (exploitation) and the uncertainty about that estimate (exploration).

    The UCB score is calculated as the sum of the value score (the average value of the outcomes of the simulations that have passed through the node) 
    and the prior score (a measure of the prior probability of the action that led to the node, adjusted for the number of times the parent node has been visited).

    Args:
      - config: (AlphaZeroConfig): Configuration settings for AlphaZero.
      - parent: (Node): The parent node in the search tree.
      - child: (Node): The child node for which the UCB score is calculated.

    Returns:
      - float: The UCB score for the child node.
    """
    pb_c = (
        np.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * child.prior
    value_score = child.value()
    return prior_score + value_score


# We use the neural network to obtain a value and policy prediction.
def evaluate(node: Node, network: Network) -> float:
    """
    Evaluate the given node in the Monte Carlo Tree Search using the neural network.

    This function uses the neural network to infer the value and policy logits for the current game state. 
    The value is returned as the result of the function.

    The policy logits are converted into probabilities using the softmax function, and these probabilities are 
    used to initialize the children of the node for each legal action in the game. The probabilities are normalized 
    so that they sum to 1.

    Args:
        node (Node): The node to be evaluated.
        network (Network): The neural network used for value and policy predictions.

    Returns:
        float: The value predicted by the neural network for the given game state.
    """
    value, policy_dist = network.inference(node.state)
    policy_sum = np.sum(policy_dist)
    for action, p in enumerate(policy_dist):
        node.children[action] = Node(p / policy_sum)

    return value


# At the end of a simulation, we propagate the evaluation all the way up the
# tree to the root.
def backpropagate(search_path: List[Node], value: float):
    """
    Backpropagates the evaluation value through the Monte Carlo Tree Search (MCTS) tree.

    Args:
        - search_path (List[Node]): List of nodes representing the search path from the leaf to the root.
        - value (float): The evaluation value to be backpropagated.

    Returns:
        None

    """
    for node in search_path:
        node.value_sum += value 
        node.visit_count += 1


# At the start of each search, we add dirichlet noise to the prior of the root
# to encourage the search to explore new actions.
def add_exploration_noise(config: AlphaZeroConfig, node: Node):
    """
    Adds Dirichlet noise to the prior of the root to encourage exploration.
    
    This function adds exploration noise to the prior probabilities of the actions of the root node. 
    This is done by first generating Dirichlet noise with the given alpha parameter and the number of actions. 
    The noise is then added to the prior probabilities of the actions, with the amount of noise determined by 
    the root exploration fraction. The prior probability of each action is updated to be a weighted sum of 
    its original value and the generated noise.

    Args:
      - config (AlphaZeroConfig): Configuration settings for AlphaZero.
      - node (Node): The root node of the MCTS search tree.

    Return:
      - None
    """
    actions = list(node.children.keys())
    noise = np.random.gamma(config.root_dirichlet_alpha, 1, len(actions))
    frac = config.root_exploration_fraction
    for a, n in zip(actions, noise):
        node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac
