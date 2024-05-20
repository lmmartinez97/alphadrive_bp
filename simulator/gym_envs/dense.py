
## Global imports
import re
from turtle import speed
import gymnasium as gym
import json
from matplotlib.pyplot import hist
import numpy as np
from gymnasium import spaces

## Local imports
from modules.carla.simulation import Simulation
from .helper_functions import create_logging_directory


class Dense(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """
    def __init__(self, simulation: Simulation, options: dict = {}):
        super(Dense, self).__init__()
        
        self.simulation = simulation
        self.step_count = 0
        self.episode_count = 0

        #Extract options from dictionary
        self.options = options
        self.max_steps = options.get("max_steps", 1000)
        self.max_episodes = options.get("max_episodes", 1000)
        self.verbose = options.get("verbose", False)
        self.ego_speed = options.get("ego_speed", 10) # m/s
        self.ego_lane = options.get("ego_lane", 0) # favour right lane

        #Reward weights
        self.collision_weight = options.get("collision_weight", 1)
        self.mpc_weight = options.get("mpc_weight", 1)
        self.timeout_weight = options.get("timeout_weight", 1)
        self.speed_weight = options.get("speed_weight", 0.1)
        self.lane_weight = options.get("lane_weight", 0.4)

        # Define action and observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(200,), dtype=np.float32)
        self.action_space = spaces.Discrete(2) # 0 default lane, 1 left lane
        
        # Declare step variables
        self.observation = None
        self.reward = 0
        self.done = False
        self.truncated = False
        self.info = {}

        #Declare episode variables
        self.episode_list = [] #Contains a dictionary for each step with the step data

        #create logging directory if it does not exist
        log_directory_path ="./gym_envs/env_logs"
        self.logging_path = create_logging_directory(log_directory_path)
        
    def reset(self, seed=None, options=None):
        """
        Resets the environment and the simulation to the initial state.
        """
        print("Resetting environment")

        self.simulation.init_game()
        self.step_count = 0

        #Clear step variables
        self.observation = np.zeros(200, dtype = np.float32)
        self.reward = 0
        self.done = False
        self.truncated = False
        self.info = {}

        #Clear episode variables
        self.episode_list.clear()

        return self.observation, {}

    def step(self, action):
        """
        Step function for the environment. It applies an action to the simulation and returns the next state, reward, done and info.

        It also logs the necessary information for the training process.

        Inputs:
            action (int): The action to be applied to the simulation. 0 is default lane, 1 is shift right, 2 is shift left.
        Returns:
            observation (np.array): The observation for the current step.
            reward (float): The reward for the current step.
            done (bool): Whether the episode is done.
            info (dict): Additional information for the current step.
        """
        #apply action to simulation
        #map environment action to simulation action
        if action == 0: #if action is 0, stay in the right lane
            self.action = 0
        elif action == 1: #if action is 1, shift to the left lane
            self.action = 2
        self.simulation.mcts_step(verbose=False, recording=True, action=self.action)
        self.step_count += 1

        #gather step data
        self.observation = self._get_observation()
        self.done_dict, self.done = self._get_done()
        self.truncated = False
        self.info = self._get_info()
        self.reward = self._get_reward()

        #logging
        self.log_step()

        if self.done:
            self.log_episode()
            self.reset()
            self.episode_count += 1

        return self.observation, self.reward, self.done, self.truncated, self.info
    
    def log_step(self):
        """
        Logs the step data for the training process.
        """
        temp_dict = {
            "episode": self.episode_count,
            "step": self.step_count,
            "observation": self.observation.tolist(),
            "reward": self.reward,
            "done": self.done,
            "info": self.info
        }
        self.episode_list.append(temp_dict)

    def log_episode(self):
        """
        Logs the episode data for the training process.
        """
        with open(self.logging_path + f"/episode_{self.episode_count}.json", "w") as f:
            for episode in self.episode_list:
                json.dump(episode, f)
                f.write('\n')
        
    
    def _get_observation(self):
        """
        Retrieves the observation for the current step.

        Inputs:
            None
        Returns:
            observation (np.array): The observation for the current step. Comes from an encoded potential field calculation that has been flattened.
        """
        self.observation = self.simulation.get_state(decision_index=self.step_count)
        return self.observation

    def _get_reward(self):
        """
        Calculates the reward for the current step.

        Inputs:
            None
        Returns:
            reward (float): The reward for the current step.
        """
        # Initialize collision terms
        collision_reward = 0
        mpc_reward = 0
        timeout_reward = 0
        speed_reward = 0
        lane_reward = 0

        #Termination rewards
        if self.done_dict["collision"]:
            collision_reward = -1 * self.collision_weight
        elif self.done_dict["mpc"]:
            mpc_reward = 1 * self.mpc_weight
        elif self.done_dict["frame_counter"]:
            timeout_reward = -1 * self.timeout_weight

        #Reward for going as fast as possible
        speed_reward = np.square(self.ego_speed - self.simulation.speed[-1]) * self.speed_weight

        #Reward for staying in the lane
        if self.action != self.ego_lane:
            lane_reward = -1 * self.lane_weight

        #Calculate total weighted reward
        self.reward = collision_reward + mpc_reward + timeout_reward + speed_reward + lane_reward
        
        return self.reward

    def _get_done(self):
        """
        Returns the termination condition for the current episode.
        """
        self.done = self.simulation.is_terminal()
        self.done_dict = self.simulation.termination_dict
        if any(self.done_dict.values()):
            self.done = True
        
        return self.done_dict, self.done
        
    def _get_info(self):
        """
        Obtain the information dictionary for the current step.
        """
        self.info = {}
        self.info["step_count"] = self.step_count
        self.info["episode_count"] = self.episode_count
        # self.info["frame"] = self.simulation.state_manager.return_frame_history(frame_number=self.step_count, history_length=1).to_dict() #vehicles within one hundred meters of ego

        return self.info

    def render(self):
        pass

    def close(self):
        if self.simulation.world is not None:
            settings = self.simulation.world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.simulation.world.world.apply_settings(settings)
            self.simulation.world.destroy()

        print("Bye, bye")