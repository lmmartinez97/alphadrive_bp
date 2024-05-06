from typing import Dict, List, Union

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