
## Global Imports
import argparse
import logging
from tabnanny import check
import stable_baselines3
import sys
import traceback

from datetime import datetime
from rich import print
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure, JSONOutputFormat
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from typing import Callable

## Local Imports
from gym_envs.dense import Dense
from gym_envs.helper_functions import create_directory
from modules.carla.simulation import Simulation


def linear_schedule(initial_value: float) -> Callable[[float], float]:
  """
  Linear learning rate schedule.

  :param initial_value: Initial learning rate.
  :return: schedule that computes
    current learning rate depending on remaining progress
  """
  def func(progress_remaining: float) -> float:
    """
    Progress will decrease from 1 (beginning) to 0.

    :param progress_remaining:
    :return: current learning rate
    """
    return progress_remaining * initial_value

  return func

def main():
    try:
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

        # Parse arguments
        args = argparser.parse_args()
        args.width, args.height = [int(x) for x in args.res.split("x")]
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)
        logging.info("listening to server %s:%s", args.host, args.port)
       
        #Options dictionary
        options = {
            "max_steps": 2000,
            "max_episodes": 500,
            "verbose": True,
            "ego_speed": 10,  # m/s
            "ego_lane": 0,  # favour right lane
            "collision_weight": -5,
            "mpc_weight": 1,
            "timeout_weight": 1,
            "speed_weight": 0.1,
            "lane_weight": 0.4
        }
        #Initialize simulation
        simulation = Simulation(args=args, frame_limit=10000, episode_limit=1, options=options)

        #Initialize environment
        env = Dense(simulation=simulation, options=options)
    
        #Configure callbacks
        timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        checkpoint_path = f"./gym_envs/checkpoint_models/{timestamp}"
        checkpoint_callback = CheckpointCallback(save_freq=1000, 
                                                save_path=create_directory(checkpoint_path) + "/",
                                                name_prefix='PPO_',
                                                save_replay_buffer=True)
        
        #Training loop
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./gym_envs/tensorboard_logs/")
        model.learn(total_timesteps=100000, callback=checkpoint_callback) #trains for about 1.15 days of simulation time
        model.save("./gym_envs/models/ppo_dense")
    
    except KeyboardInterrupt:
        env.close()
        print ('Bye env')
        sys.exit()

    except Exception as e:
        print(traceback.format_exc())

if __name__ == '__main__':
    main()

