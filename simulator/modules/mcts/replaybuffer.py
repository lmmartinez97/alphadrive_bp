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
        self.total_moves = 0

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
        self.total_moves = sum([len(g.node_history) for g in self.buffer])

    def sample_batch(self) -> List[Tuple[List[np.array], Tuple[float, List[float]]]]:
        """
        Samples a batch of self-play game data for training.

        Returns:
            List[Tuple[List[np.array], Tuple[float, List[float]]]]:
                A list of tuples containing game states (images) and their target values (value, policy).
        """       
        # Initialize a dictionary to keep track of sampled moves per game
        sampled_moves = {game: set() for game in self.buffer}
        
        # Sample moves until reaching desired batch size
        batch = []
        while len(batch) < self.batch_size:
            # Sample a game with probability proportional to its length
            game = np.random.choice(self.buffer, p=[len(g.node_history) / self.total_moves for g in self.buffer])
            
            # Check if all moves in the game have been sampled
            if len(sampled_moves[game]) == len(game.action_history):
                # If all moves have been sampled, continue to the next iteration
                continue
            
            # Sample a move index from the game that hasn't been sampled yet
            while True:
                move_index = np.random.randint(len(game.action_history))
                if move_index not in sampled_moves[game]:
                    # Mark the sampled move as sampled
                    sampled_moves[game].add(move_index)
                    break
            
            # Append the sampled state and target values to the batch
            state = game.node_history[move_index].state
            target = (game.node_history[move_index].value(), game.node_history[move_index].policy())
            batch.append((state, target))

        #dump batch into file for state-action-value analysis
        #create file name with iteration number
        fname = '/home/lmmartinez/Tesis/codigo_tesis/simulator/logs/' + 'batch' + str(self.batch_iter) + '.txt'
        #batch is a list of tuples, each tuple is a pair of lists: [state, [value, policy]]        
        self.batch_iter += 1
        with open(fname, 'w') as f:
            for item in batch:
                dict_ = {'state': item[0].tolist(), 'value': item[1][0], 'policy': item[1][1]}
                json.dump(dict_, f)
                f.write('\n')
        
        return batch
