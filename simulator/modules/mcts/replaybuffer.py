## Global imports
from typing import List, Tuple
import numpy as np
import json

## Local imports
from .game import Game
from .alphazeroconfig import AlphaZeroConfig

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

    def __init__(self, config: AlphaZeroConfig) -> None:
        """
        Initializes a new ReplayBuffer instance.

        Args:
            config (AlphaZeroConfig): Configuration object containing parameters.
        """
        self.window_size: int = config.window_size
        self.batch_size: int = config.batch_size
        self.buffer: List[Game] = []
        self.batch_iter = 1

    def save_game(self, game: Game ) -> None:
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
        ret = [(g.make_image(node_index=i), g.make_target(node_index = i)) for (g, i) in game_pos]
        #dump batch into file for state-action-value analysis
        #create file name with iteration number
        fname = '/home/lmmartinez/Tesis/codigo_tesis/simulator/logs/' + 'batch' + str(self.batch_iter) + '.txt'
        #batch is a list of tuples, each tuple is a pair of lists: [state, [value, policy]]        
        self.batch_iter += 1
        with open(fname, 'w') as f:
            for item in ret:
                dict_ = {'state': item[0].tolist(), 'value': item[1][0], 'policy': item[1][1]}
                json.dump(dict_, f)
                f.write('\n')
        return ret