"""SIMULATION MODULE"""
from __future__ import print_function

# ==============================================================================
# -- Python imports ------------------------------------------------------------
# ==============================================================================

import glob
import matplotlib.pyplot as plt
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
from .pid import CarController

from ..agents.navigation.global_route_planner import GlobalRoutePlanner
from ..agents.navigation.behavior_types import Cautious, Normal, Aggressive

from .autoencoder import load_model

class Simulation:
  
    def __init__(self, args = None, frame_limit = None, episode_limit = 100):
        """Constructor method for simulation class
            TODO: Add support for multiple vehicles
                  Add support for mcts
        """
        # simulation variables
        print()
        self.episode_counter = 0
        self.frame_counter = 0
        self.args = args
        if not frame_limit:
            self.frame_limit = np.inf
        else:
            self.frame_limit = frame_limit
        self.episode_limit = episode_limit
        self.simulation_period = 0.01
        self.decision_period_seconds = 1
        self.decision_period = int(self.decision_period_seconds / self.simulation_period)
        self.available_actions = {
            0: 0, #do not introduce any offset
            1: 4, #introduce a one lane offset to the right
            2: -4 #introduce a one lane offset to the left
        }

        #record variables for plotting
        self.yaw = []
        self.throttle_brake = []
        self.steer = []
        self.speed = []
        self.position = []
        self.vel_target = []
        self.pos_target = []
        self.yaw_target = []
        self.lat_error = []
        self.lon_error = []
        self.time = []

        #init pygame
        pygame.init()
        pygame.font.init()
        
        self.agents_dict = {"cautious": Cautious, "normal": Normal, "aggressive": Aggressive}
        #include static camera
        if self.args.static_camera:
            self.args.width *= 2
        # initialize display
        self.display = pygame.display.set_mode(
            (self.args.width, self.args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        print_green("Created display")

        # init client and apply settings
        self.client = carla.Client(self.args.host, self.args.port)
        self.client.set_timeout(6.0)
        self.client.load_world("mergin_scene_1")
        # get traffic manager
        self.traffic_manager = self.client.get_trafficmanager()
        self.sim_world = self.client.get_world()

        # initialize hud and world
        self.hud = HUD(self.args.width, self.args.height, text=__doc__)
        print_green("Created hud")
        self.world = World(self.sim_world, self.hud, self.args)
        print_green("Created world instance")
        
        #load autoencoder
        self.autoencoder = load_model(model_name= "autoencoder_1" ,directory="../scene_representation/training/saved_models")
        #Initialize GlobalRoutePlanner
        self.grp = GlobalRoutePlanner(wmap=self.world.map, sampling_resolution=2) # Sampling resolution of 2 meters
        #Initialize Potential Field
        rx = 50 # horizontal semiaxis of ellipse to consider as ROI
        ry = 6 # vertical semiaxis of ellipse to consider as ROI
        sx = 0.5
        sy = 0.1
        self.potential_field = PotentialField(radius_x = rx, radius_y = ry, step_x = sx, step_y = sy)
        
        #Configure PID controllers
        self.longitudinal_params = {
            'Kp': 0.75,
            'Ki': 0.25,
            'Kd': 0.05,
            'dt': self.world.delta_seconds,
            'max_throttle': 0.75,
            'max_brake': 0.5,
            'max_throttle_increment': 0.05,
            'max_brake_increment': 0.03,
            'max_integral_threshold': 0.5,
        }
        self.lateral_params = {
            'Kp': 0.9,
            'Ki': 0.03,
            'Kd': 0,
            'dt': self.world.delta_seconds,
            'max_steering': 1.0,
            'max_steering_increment': 0.03,
        }
        self.pid = CarController(longitudinal_params=self.longitudinal_params, lateral_params=self.lateral_params, reference=None)

        self.controller = KeyboardControl(self.world)
        self.clock = pygame.time.Clock()

    def init_game(self):
        """Method for initializing a new episode"""
        self.world.restart(self.args)
    
        self.agent_type = self.agents_dict[self.args.behavior]
        #hand over control of npcs to traffic manager
        self.traffic_manager.set_synchronous_mode(True)
        for vehicle in self.world.npcs:
            self.traffic_manager.auto_lane_change(vehicle, True)
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, np.random.randint(-20, 20))
            self.traffic_manager.distance_to_leading_vehicle(vehicle, 5)
            self.traffic_manager.collision_detection(vehicle, self.world.player, False)
            self.traffic_manager.ignore_lights_percentage(vehicle, 0)
            vehicle.set_autopilot(True)
            
        # Set the agent destination
        self.route = self.grp.trace_route(self.world.spawn_waypoint, self.world.dest_waypoint)
        self.reference = [[self.agent_type.max_speed, waypoint.transform.location] for waypoint, _ in self.route]
        self.pid.update_reference(self.reference)

        self.world.player.set_transform(self.world.spawn_point_ego)
        self.clock.tick()
        self.world.world.tick()
        self.world.dataframe = pd.DataFrame()

        self.prev_timestamp = self.world.world.get_snapshot().timestamp
        self.first_timestamp = self.prev_timestamp

        for waypoint, _ in self.route:
            self.world.world.debug.draw_point(
                waypoint.transform.location,
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=100,
            )

        # Clear record variables
        self.yaw.clear()
        self.throttle_brake.clear()
        self.steer.clear()
        self.speed.clear()
        self.position.clear()
        self.vel_target.clear()
        self.pos_target.clear()
        self.yaw_target.clear()
        self.lat_error.clear()
        self.lon_error.clear()
        self.time.clear()

    def game_step(self, verbose = False, action = None, recording = False):
        self.clock.tick()
        self.world.world.tick()
        self.world.tick(self.clock, self.episode_counter, self.frame_counter)

        timestamp = self.world.world.get_snapshot().timestamp

        if self.display:
            self.world.render(self.display)
            pygame.display.flip()

        if self.controller and self.controller.parse_events():
            return -1, -1, prev_timestamp
        
        if self.frame_counter % self.decision_period == 0 and action is not None:
            self.pid.lateral_controller.set_offset(self.available_actions[action])
            self.decision_period += 1
        else:
            self.pid.lateral_controller.set_offset(0)

        self.world.record_frame_state(frame_number=self.frame_counter)
        self.potential_field.update(self.world.return_frame_history(frame_number=self.frame_counter, history_length=5))
        pf = self.potential_field.calculate_field()
        state = self.autoencoder.encode(pf).flatten()
        
        velocity = self.world.player.get_velocity()
        speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2)
        transform = self.world.player.get_transform()
        control = self.pid.run_step(current_speed=speed, current_transform=transform)
        control.manual_gear_shift = False
        self.world.player.apply_control(control)
        
        #record control signals for plotting
        self.yaw.append(self.world.player.get_transform().rotation.yaw)
        self.throttle_brake.append(control.throttle - control.brake)
        self.steer.append(control.steer)
        self.speed.append(speed)
        self.position.append(self.world.player.get_location())
        self.vel_target.append(self.pid.longitudinal_controller.target_speed)
        self.pos_target.append(self.pid.lateral_controller.target_location)
        self.yaw_target.append(self.pid.lateral_controller.target_yaw)
        self.lon_error.append(self.pid.longitudinal_controller.error)
        self.lat_error.append(self.pid.lateral_controller.error)
        self.time.append(timestamp.elapsed_seconds - self.first_timestamp.elapsed_seconds)

        #await send_log_data(log_host, log_port, frame_df)
        prev_timestamp = timestamp

        if verbose:
            print("State: ", state)
            print("State len: ", len(state))
            print("Frame: ", self.frame_counter)
            print("Control action: ", control)
            self.potential_field.plot_field()
            self.autoencoder.compare(pf, num_plots=1)
            input("Press Enter to continue")
            
        return state
    
    def plot_results(self):
        """Method for plotting the results of the simulation"""
        width = 2
        fig, ax = plt.subplots(2, 2, figsize=(15, 15))
        ax[0, 0].plot(self.time ,self.yaw_target, label="Target Yaw", linewidth=width)
        ax[0, 0].plot(self.time,self.lat_error, label="Lateral Error", linewidth=width)
        ax[0, 0].set_title("Yaw")
        ax[0, 0].set_xlabel("Time")
        ax[0, 0].set_ylabel("Yaw")
        ax[0, 0].legend()

        ax[0, 1].plot(self.time,self.speed, label="Speed", linewidth=width)
        ax[0, 1].plot(self.time,self.vel_target, label="Target Speed", linewidth=width)
        ax[0, 1].plot(self.time,self.lon_error, label="Longitudinal Error", linewidth=width)
        ax[0, 1].set_title("Speed")
        ax[0, 1].set_xlabel("Time")
        ax[0, 1].set_ylabel("Speed")
        ax[0, 1].legend()

        ax[1, 0].plot(self.time,self.steer, label="Steer", linewidth=width)
        ax[1, 0].plot(self.time, np.gradient(self.steer), label="Steer Derivative", linewidth=2)
        ax[1, 0].set_title("Steer")
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_ylabel("Steer")
        ax[1, 0].legend()

        ax[1, 1].plot(self.time,self.throttle_brake, label="Throttle/Brake", linewidth=width)
        ax[1, 1].plot(self.time, np.gradient(self.throttle_brake), label="ThrottleBrake Derivative", linewidth=2)
        ax[1, 1].set_title("Throttle/Brake")
        ax[1, 1].set_xlabel("Time")
        ax[1, 1].set_ylabel("Throttle/Brake")
        ax[1, 1].legend()

        fig, ax = plt.subplots(figsize=(15, 15))
        ax.plot([pos.x for pos in self.position], [pos.y for pos in self.position], label="Position", linewidth=width)
        ax.plot([pos.x for pos in self.pos_target], [pos.y for pos in self.pos_target], label="Target Position", linewidth=width)
        ax.set_title("Position")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        plt.show()

    def run(self):
        """Method for running the simulation"""
        self.init_game()
        while self.episode_counter < self.episode_limit:
            while (self.frame_counter < self.frame_limit) and not self.pid.done:
                if self.frame_counter % 100 == 0:
                    verbose = False
                self.game_step(verbose=verbose)
                self.frame_counter += 1
            self.plot_results()
            input("Press Enter to continue")
            self.frame_counter = 0
            print("Restored frame state")
            print("Finished episode ", self.episode_counter, " initializing next episode")
            self.episode_counter += 1
            self.clock.tick()
            self.world.world.tick()
            self.init_game()
