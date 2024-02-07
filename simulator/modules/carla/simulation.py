"""SIMULATION MODULE"""
from __future__ import print_function

# ==============================================================================
# -- Python imports ------------------------------------------------------------
# ==============================================================================

import glob
import numpy as np
import os
import pandas as pd
import sys

from rich import print

# ==============================================================================
# -- pygame import -------------------------------------------------------------
# ==============================================================================

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================

try:
    sys.path.append(
        glob.glob(
            "/home/lmmartinez/CARLA/PythonAPI/carla/dist/carla-*%d.%d-%s.egg"
            % (
                sys.version_info.major,
                sys.version_info.minor,
                "win-amd64" if os.name == "nt" else "linux-x86_64",
            )
        )[0]
    )
except IndexError:
    pass
import carla
from carla import ColorConverter as cc

# ==============================================================================
# -- Local imports --- ---------------------------------------------------------
# ==============================================================================


from .keyboard_control import KeyboardControl
from .printers import print_blue, print_green, print_highlight, print_red
from .world import World
from .hud import HUD
from .potential_field import PotentialField

from ..agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from ..agents.navigation.behavior_agent import (
    BehaviorAgent,
)  # pylint: disable=import-error
from ..agents.navigation.constant_velocity_agent import (
    ConstantVelocityAgent,
)  # pylint: disable=import-error

from .autoencoder import load_model

class Simulation:
  
    def __init__(self, args = None, frame_limit = None, episode_limit = 100):
        """Constructor method for simulation class
            TODO: Add support for multiple vehicles
                  Add support for mcts
        """
        # simulation variables
        self.episode_counter = 0
        self.frame_counter = 0
        if not frame_limit:
            self.frame_limit = np.inf
        else:
            self.frame_limit = frame_limit
        self.episode_limit = episode_limit
        self.simulation_period = 0.01
        self.decision_period = 1

        #load autoencoder
        self.autoencoder = load_model(model_name= "autoencoder_1" ,directory="../scene_representation/training/saved_models")
        #init pygame
        pygame.init()
        pygame.font.init()

        #potential field instance
        rx = 50 # horizontal semiaxis of ellipse to consider as ROI
        ry = 6 # vertical semiaxis of ellipse to consider as ROI
        sx = 0.5
        sy = 0.1
        self.potential_field = PotentialField(radius_x = rx, radius_y = ry, step_x = sx, step_y = sy)

        #include static camera
        if args.static_camera:
            args.width *= 2
        # initialize display
        self.display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        print("Created display")

        # init client and apply settings
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(6.0)
        self.client.load_world("mergin_scene_1")
        # get traffic manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.sim_world = self.client.get_world()

        # initialize hud and world
        self.hud = HUD(args.width, args.height, text=__doc__)
        print("Created hud")
        self.world = World(self.sim_world, self.hud, args)
        print("Created world instance")

        self.controller = KeyboardControl(self.world)

        # initialize agent
        self.agent = None
        if args.agent == "Basic":
            self.agent = BasicAgent(self.world.player, 30)
            self.agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            self.agent = ConstantVelocityAgent(self.world.player, 30)
            ground_loc = self.world.world.ground_projection(self.world.player.get_location(), 5)
            if ground_loc:
                self.world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            self.agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            self.agent = BehaviorAgent(self.world.player, behavior=args.behavior)

        self.clock = pygame.time.Clock()

    def init_game(self):
        """Method for initializing a new episode
        
        TODO: Add support for multiple vehicles
        """
        # Set the agent destination
        route = self.agent.set_destination(
            end_location=self.world.destination, start_location=self.world.spawn_point_ego.location
        )

        print("Spawn point is: ", self.world.spawn_point_ego.location)
        print("Destination is: ", self.world.destination)

        self.world.player.set_transform(self.world.spawn_point_ego)
        self.clock.tick()
        self.world.world.tick()
        self.world.dataframe = pd.DataFrame()

        input("Press Enter to start episode")
        self.prev_timestamp = self.world.world.get_snapshot().timestamp

        for waypoint, _ in route:
            self.world.world.debug.draw_point(
                waypoint.transform.location,
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=100,
            )
    
    def game_step(self):
        self.clock.tick()
        timestamp = self.world.world.get_snapshot().timestamp
        frame_history = self.world.record_frame_state(frame_number=self.frame_counter)
        #self.potential_field.update(frame_history)
        self.world.world.tick()
        self.world.tick(self.clock, self.episode_counter, self.frame_counter)

        if self.display:
            self.world.render(self.display)
            pygame.display.flip()

        if self.controller and self.controller.parse_events():
            return -1, -1, prev_timestamp

        control = self.agent.run_step()
        control.manual_gear_shift = False
        self.world.player.apply_control(control)
        #await send_log_data(log_host, log_port, frame_df)
        prev_timestamp = timestamp

        if self.frame_counter % 100 == 0:
            print("Frame: ", self.frame_counter)
            print("Control action: ", control)

    def run(self):
        """Method for running the simulation"""
        self.init_game()
        while self.episode_counter < self.episode_limit:
            while (self.frame_counter < self.frame_limit) and not self.agent.done():
                self.game_step()
                self.frame_counter += 1
            self.frame_counter = 0
            print("Restored frame state")
            print("Finished episode ", self.episode_counter, " initializing next episode")
            self.episode_counter += 1
            self.clock.tick()
            self.world.world.tick()
            self.init_game()
