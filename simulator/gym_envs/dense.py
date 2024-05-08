
## Global imports
import numpy as np
import gymnasium as gym
from gymnasium import spaces

## Local imports
from ..modules.carla.simulation import Simulation


class Dense(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    def __init__(self, simulation: Simulation):
        super(Dense, self).__init__()
        
        self.simulation = simulation

        # Define action and observation space
        self.observation_space = spaces.Box(low=-1, high=1, shape=(200,), dtype=np.float32)
        self.action_space = spaces.Discrete(3) # 0 keep lane, 1 shift right, 2 shift left

        #create logging directory if it does not exist
        

    def reset(self):
        #setup logging

        self.simulation.init_game()

    def step(self, action):
        self.simulation.mcts_step(verbose=False, recording=False, action=action)
    
    def render(self):
        pass

    def close(self):
        pass