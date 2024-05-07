"""Selfplay functions for the AlphaZero algorithm."""


from __future__ import division

import numpy as np

from rich import print
from typing import List, Tuple

from .game import Game
from .node import Node
from .replaybuffer import ReplayBuffer
from .alphazeroconfig import AlphaZeroConfig
from .sharedstorage import SharedStorage
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
    #while not over max moves allowed, and not terminal state:
    while not game.terminal(simulation=simulation) and len(game.node_history) < config.max_moves:
        print(f"Playing game - move {len(game.node_history)+1}/{config.max_moves}")
        action, root = run_mcts(config, game, network, simulation)
        game.apply(action=action, simulation=simulation, recording=True)
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
    root = Node()
    root.assign_state(simulation.get_state(decision_index=len(game.node_history)))
    
    # Run the specified number of simulations
    for i in range(config.num_simulations):
        print(f"Running simulation {i+1}/{config.num_simulations} in monte carlo tree search.")
        scratch_game = game.clone() #initialize scratch_game by copying the real game with node history until now        
        #determine the first action to take in the scratch game using policy rollout
        action = policy_based_rollout(node=root, network=Network)
        root_child = action
        scratch_game.apply(action=action, simulation=simulation, recording=False, node=root) #apply the action to the scratch game
        
        #simulate the rest of the scratch game using a random policy
        while not scratch_game.terminal(simulation=simulation) and len(scratch_game.node_history) < config.max_moves:
            #generate node to store next state
            node = Node()
            node.assign_state(simulation.get_state(decision_index=len(scratch_game.node_history)))
            #select action using policy rollout
            action, node = policy_based_rollout(node=node, network=network)
            scratch_game.apply(action=action, simulation=simulation, recording=False, node=node)

        #return simulation to the original state - before MCTS started
        print(f"Restoring game state to {len(game.node_history)}")
        simulation.state_manager.restore_frame(frame_number=len(game.node_history), vehicle_list=simulation.world.actor_list) #we restore to frame of main game, NOT mock game for mcts
        simulation.decision_counter = len(game.node_history) #decision counter has increased with MCTS mock game, so we need to reset it to the main game decision counter
        
        # Evaluate the leaf node and propagate the termination value to root node
        value = simulation.get_reward()
        backpropagate(root, value, root_child)
        

    # Select the best action from the root node
    return select_action(config, game, root), root

def random_rollout(node: Node) -> int:
    """
    Selects a random action in a scratch game. Used in fiticious games, after the policy rollout move from root.

    Args:
        node (Node): The node to be evaluated.
    
    Returns:
        next_action: The action to be taken next.
    """
    node.visit_count += 1
    next_action = np.random.choice(list(node.children.keys()))
    node.update_child_visit_count(child_visited=next_action)

    return next_action

def policy_based_rollout(node: Node, network: Network) -> float:
    """
    Evaluate the given node in the Monte Carlo Tree Search using the neural network.
    This function uses the neural network to infer the value and policy probabilities for the current game state. 
    Since it is part of the rollout phase, a epsilon-greedy policy is used to select the next action to take.

    Args:
        node (Node): The node to be evaluated.
        network (Network): The neural network used for value and policy predictions.

    Returns:
        next_action: The action to be taken next.
    """
    value, policy_dist = network.inference(node.state)

    # Select the next action using an epsilon-greedy policy
    epsilon = 0.25
    if np.random.rand() < epsilon:
        next_action = np.random.choice(list(node.children.keys()))
    else: #choose the action with the highest probability
        next_action = np.argmax(policy_dist)

    node.update_child_visit_count(child_visited=next_action)

    return next_action

def select_action(config: AlphaZeroConfig, node: Node) -> Tuple[int, Node]:
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
    return action


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
      - config (AlphaZeroConfig): Configuration settings for AlphaZero.
      - parent (Node): The parent node of the child node for which the UCB score is calculated.
      - child (Node): The child node for which the UCB score is calculated.

    Returns:
      - float: The UCB score for the child node.
    """
    pb_c = (
        np.log((parent.visit_count + config.pb_c_base + 1) / config.pb_c_base)
        + config.pb_c_init
    )
    pb_c *= np.sqrt(parent.visit_count) / (child.visit_count + 1)

    prior_score = pb_c * parent.value()
    value_score = child.value()
    return prior_score + value_score

def backpropagate(root: Node, value: float, root_child: int) -> None:
    """
    Backpropagates the evaluation value of a ficticious game to the root node.

    Args:
        root (Node): The root node of the search tree.
        value (float): The evaluation value to be backpropagated.
        root_child (int): The action that led to the root child node.

    Returns:
        None

    """
    root.value_sum += value
    root.visit_count += 1
    #update child
    root.update_child(child_visited=root_child, value=value)

