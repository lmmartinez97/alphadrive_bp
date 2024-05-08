
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

    def reset(self, seed=None, options=None):
        self.simulation.init_game()

    def step(self, action):
        self.simulation.mcts_step(verbose=False, recording=false, action=action)
    
    def render(self):
        pass

    def close(self):
        pass