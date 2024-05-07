"""AlphaZero training loop."""
from __future__ import division

import argparse
import logging
import pygame
import traceback

from rich import print

from modules.mcts.alphazeroconfig import AlphaZeroConfig
from modules.mcts.sharedstorage import SharedStorage
from modules.mcts.replaybuffer import ReplayBuffer

from modules.mcts.network import Network
from modules.mcts.self_play import run_selfplay
from modules.mcts.utils import create_directory, train_network
from modules.carla.simulation import Simulation

def alphazero(args) -> Network:
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
        "num_actors": 1,
        "num_sampling_moves": 50,
        "max_moves": 3,
        "num_simulations": 1,
        "root_dirichlet_alpha": 0.3,
        "root_exploration_fraction": 0.25,
        "pb_c_base": 19652,
        "pb_c_init": 1.25,
        "network_arch": [200, 32, 10],
        "training_steps": int(1e2),
        "checkpoint_interval": int(1e3),
        "training_iterations": 4,
        "games_per_iteration": 1,
        "window_size": int(600),
        "batch_size": 64,
        "weight_decay": 1e-4,
        "momentum": 0.9,
        "learning_rate_schedule": {0: 2e-1, 100e3: 2e-2, 300e3: 2e-3, 500e3: 2e-4},
    }
    try:
        # Initialize the AlphaZero configuration, shared storage, replay buffer, and simulation
        config = AlphaZeroConfig(config_dict)
        storage = SharedStorage()
        replay_buffer = ReplayBuffer(config)
        simulation = Simulation(args=args)

        # Save the initial network to the shared storage
        storage.save_network(0, Network(config.network_arch))
        save_path = create_directory("../checkpoints")
        print(f"Saving network to {save_path}")

        # Run the training process for the specified number of iterations
        for i in range(config.training_iterations):
            print(f"Iteration {i+1}/{config.training_iterations}")
            run_selfplay(config=config, storage=storage, replay_buffer=replay_buffer, simulation=simulation)
            # if len(replay_buffer.buffer) < config.batch_size:
            #     print("Not enough data in replay buffer. Skipping training.")
            #     continue
            print(f"Training network {i+1}/{config.training_iterations}")
            train_network(config=config, storage=storage, replay_buffer=replay_buffer, training_iter=i, path=save_path)

        # Retrieve, save, and return the latest trained network
        latest_network = storage.latest_network()
        latest_network.save_model("latest_network")
    
    except KeyboardInterrupt as e:
        print("\n")
        #simulation.plot_results()
        if simulation.world is not None:
            settings = simulation.world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            simulation.world.world.apply_settings(settings)
            simulation.world.destroy()
            simulation.world = None
        pygame.quit()
        return -1

    except Exception as e:
        print(traceback.format_exc())

    finally:
        if simulation.world is not None:
            settings = simulation.world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            simulation.world.world.apply_settings(settings)
            simulation.world.destroy()
        pygame.quit()

        print("Bye, bye")
    
    return storage.latest_network()

def main():
    argparser = argparse.ArgumentParser(description="CARLA Automatic Control Client")
    argparser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        dest="debug",
        help="Print debug information",
    )
    argparser.add_argument(
        "--host",
        metavar="H",
        default="127.0.0.1",
        help="IP of the host server (default: 127.0.0.1)",
    )
    argparser.add_argument(
        "-p",
        "--port",
        metavar="P",
        default=2000,
        type=int,
        help="TCP port to listen to (default: 2000)",
    )
    argparser.add_argument(
        "--res",
        metavar="WIDTHxHEIGHT",
        default="800x540",
        help="Window resolution (default: 800x540)",
    )
    argparser.add_argument(
        "--sync", action="store_true", help="Synchronous mode execution"
    )
    argparser.add_argument(
        "--filter",
        metavar="PATTERN",
        default="vehicle.*",
        help='Actor filter (default: "vehicle.*")',
    )
    argparser.add_argument(
        "-l",
        "--loop",
        action="store_true",
        dest="loop",
        help="Sets a new random destination upon reaching the previous one (default: False)",
    )
    argparser.add_argument(
        "-a",
        "--agent",
        type=str,
        choices=["Behavior", "Basic"],
        help="select which agent to run",
        default="Behavior",
    )
    argparser.add_argument(
        "-b",
        "--behavior",
        type=str,
        choices=["cautious", "normal", "aggressive"],
        help="Choose one of the possible agent behaviors (default: normal) ",
        default="normal",
    )
    argparser.add_argument(
        "-s",
        "--seed",
        help="Set seed for repeating executions (default: None)",
        default=None,
        type=int,
    )
    argparser.add_argument(
        "-ff",
        "--fileflag",
        help="Set flag for logging each frame into client_log",
        default=0,
        type=int,
    )
    argparser.add_argument(
        "-sc",
        "--static_camera",
        help="Set flag for using static camera",
        default=0,
        type=int,
    )

    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split("x")]
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)
    alphazero(args)


if __name__ == "__main__":
    main()