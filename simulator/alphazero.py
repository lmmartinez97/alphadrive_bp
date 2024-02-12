"""AlphaZero training loop."""


from __future__ import division

from modules.mcts.helpers import SharedStorage, ReplayBuffer, AlphaZeroConfig
from modules.mcts.network import Network
from modules.mcts.self_play import run_selfplay
from modules.mcts.utils import create_directory, train_network
from modules.carla.simulation import Simulation

def alphazero() -> Network:
    """
    The main function that coordinates the AlphaZero training process.

    This function sets up the configuration for the AlphaZero algorithm, 
    initializes the shared storage, replay buffer, and simulation, 
    and then runs the training process for a specified number of iterations. 
    After training, it saves and returns the latest trained neural network.

    The configuration dictionary includes parameters such as:
    - num_actors: The number of actors for the self-play process.
    - num_sampling_moves: The number of moves up to which actions are decided probabilisticly 
                          -instead of greedily- in the self-play process.
    - max_moves: The maximum number of moves in a game.
    - num_simulations: The number of Monte Carlo Tree Search simulations to run for each move.
    - root_dirichlet_alpha and root_exploration_fraction: Parameters for exploration during MCTS.
    - pb_c_base and pb_c_init: Parameters for the PUCT algorithm.
    - network_arch: The architecture of the neural network.
    - training_steps: The number of training steps to run.
    - checkpoint_interval: The interval at which to save checkpoints.
    - training_iterations: The number of training iterations to run.
    - games_per_iteration: The number of games to play in each iteration.
    - window_size: The size of the window for the replay buffer.
    - batch_size: The size of the batch for training the network.
    - weight_decay and momentum: Parameters for the optimizer.
    - learning_rate_schedule: The learning rate schedule for training.

    Returns:
      'Network': The latest trained neural network.
    """
    # Configuration for the AlphaZero algorithm
    config_dict = {
        "num_actors": 5000,
        "num_sampling_moves": 30,
        "max_moves": 512,
        "num_simulations": 800,
        "root_dirichlet_alpha": 0.3,
        "root_exploration_fraction": 0.25,
        "pb_c_base": 19652,
        "pb_c_init": 1.25,
        "network_arch": [200, 64, 16],
        "training_steps": int(700e3),
        "checkpoint_interval": int(1e3),
        "training_iterations": 60,
        "games_per_iteration": 50,
        "window_size": int(1e6),
        "batch_size": 4096,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "learning_rate_schedule": {0: 2e-1, 100e3: 2e-2, 300e3: 2e-3, 500e3: 2e-4},
    }
    # Initialize the AlphaZero configuration, shared storage, replay buffer, and simulation
    config = AlphaZeroConfig(config_dict)
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)
    simulation = Simulation()

    # Save the initial network to the shared storage
    storage.save_network(0, Network(config.network_arch))
    save_path = create_directory("../checkpoints")

    # Run the training process for the specified number of iterations
    for i in range(config.training_iterations):
        print(f"Iteration {i+1}/{config.training_iterations}")
        run_selfplay(config=config, storage=storage, replay_buffer=replay_buffer, simulation=simulation)
        train_network(config=config, storage=storage, replay_buffer=replay_buffer, training_iter=i, path=save_path)

    # Retrieve, save, and return the latest trained network
    latest_network = storage.latest_network()
    latest_network.save_model("latest_network")
    
    return storage.latest_network()

if __name__ == "__main__":
    alphazero()
