"""AlphaZero training loop."""


from __future__ import google_type_annotations
from __future__ import division

from helpers import SharedStorage, ReplayBuffer, AlphaZeroConfig
from network import Network, train_network
from self_play import run_selfplay
from utils import create_directory

def alphazero() -> "Network":
    """
    The main function that coordinates the AlphaZero training process.

    Returns:
      'Network': The latest trained neural network.
    """
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
    config = AlphaZeroConfig(config_dict)
    storage = SharedStorage()
    replay_buffer = ReplayBuffer(config)

    storage.save_network(0, Network(config.network_arch))
    save_path = create_directory("../checkpoints")
    for i in range(config.training_iterations):
        print(f"Iteration {i+1}/{config.training_iterations}")
        run_selfplay(config, storage, replay_buffer)
        train_network(config=config, storage=storage, replay_buffer=replay_buffer, training_iter=i, path=save_path)

    latest_network = storage.latest_network()
    latest_network.save_model("latest_network")
    
    return storage.latest_network()
