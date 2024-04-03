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




from ..agents.navigation.behavior_types import Cautious, Normal, Aggressive
from ..agents.navigation.global_route_planner import GlobalRoutePlanner
from .autoencoder import load_model
from .hud import HUD
from .keyboard_control import KeyboardControl
from .pid import CarController
from .potential_field import PotentialField
from .printers import print_blue, print_green, print_highlight, print_red
from .state_manager import StateManager
from .world import World

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
        self.decision_counter = 0
        self.args = args
        if not frame_limit:
            self.frame_limit = np.inf
        else:
            self.frame_limit = frame_limit
        self.episode_limit = episode_limit
        self.simulation_period = 0.1
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
        
        #init state manager
        self.state_manager = StateManager()

        # initialize hud and world
        self.hud = HUD(self.args.width, self.args.height, text=__doc__)
        print_green("Created hud")
        self.world = World(self.sim_world, self.hud, self.args, self.simulation_period)
        print_green("Created world instance")
        
        #load autoencoder
        self.autoencoder = load_model(model_name= "autoencoder_1" ,directory="../scene_representation/training/saved_models")
        #Initialize GlobalRoutePlanner
        self.grp = GlobalRoutePlanner(wmap=self.world.map, sampling_resolution=10) 
        #Initialize Potential Field
        rx = 50 # horizontal semiaxis of ellipse to consider as ROI
        ry = 6 # vertical semiaxis of ellipse to consider as ROI
        sx = 0.5 # step size in x direction
        sy = 0.1 # step size in y direction
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
            'Kp': 0.15,
            'Ki': 0.05,
            'Kd': 0.1,
            'dt': self.world.delta_seconds,
            'max_steering': 1,
            'max_steering_increment': 1,
            'max_integral_threshold': None,
        }
        #Configure Stanley controller
        self.stanley_params = {
            'Ke': 0.8,
            'Kv': 1,
            'Kr': 0.5,
            'max_steering': 1.22,
            'vehicle_length': 2.7,
            'dt': self.world.delta_seconds,
        }
        
        if 0:
            self.pid = CarController(longitudinal_params=self.longitudinal_params, lateral_params=self.lateral_params, reference=None, lateral_controller_type="PID")
        else:
            self.pid = CarController(longitudinal_params=self.longitudinal_params, lateral_params=self.stanley_params, reference=None, lateral_controller_type="Stanley")
    
        self.controller = KeyboardControl(self.world)
        self.clock = pygame.time.Clock()

    def init_game(self):
        """Method for initializing a new episode"""
        print("Initializing new game")
        self.world.restart(self.args)
        self.state_manager.reset() #reset state manager and frame history
        self.pid.done = False #reset done flag of pid controller
    
        self.agent_type = self.agents_dict[self.args.behavior]
        self.agent_type.max_speed = 50
        #hand over control of npcs to traffic manager
        self.traffic_manager.set_synchronous_mode(True)
        for vehicle in self.world.npcs:
            self.traffic_manager.auto_lane_change(vehicle, False)
            self.traffic_manager.vehicle_percentage_speed_difference(vehicle, np.random.randint(-100, -50))
            self.traffic_manager.distance_to_leading_vehicle(vehicle, 5)
            self.traffic_manager.collision_detection(vehicle, self.world.player, False)
            self.traffic_manager.ignore_lights_percentage(vehicle, 0) #vehicle does not ignore lights
            self.traffic_manager.ignore_vehicles_percentage(vehicle, 0) #vehicle does not ignore other vehicles
            vehicle.set_autopilot(True)
            vehicle.set_target_velocity(carla.Vector3D(x=self.agent_type.max_speed*np.random.uniform(0.85, 1.15)/3.6, y=0, z=0)) 
            #vehicle.set_target_velocity(carla.Vector3D(x=self.agent_type.max_speed/3.6, y=0, z=0))
            #vehicles spawn with random speed between 85% and 115% of the agents target speed
        # Set the agent destination
        self.route = self.grp.trace_route(self.world.spawn_waypoint, self.world.dest_waypoint)
        self.reference = [[self.agent_type.max_speed, waypoint.transform] for waypoint, _ in self.route[1:]]
        self.pid.update_reference(self.reference)

        self.prev_timestamp = self.world.world.get_snapshot().timestamp
        self.first_timestamp = self.prev_timestamp

        for waypoint, _ in self.route:
            self.world.world.debug.draw_point(
                waypoint.transform.location,
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=100,
            )
        self.world.player.set_target_velocity(carla.Vector3D(x=self.agent_type.max_speed/3.6, y=0, z=0))
        self.world.world.tick()
        self.world.world.tick() #tick two times to set target velocity
        # Clear record variables
        self.yaw.clear()
        self.throttle_brake.clear()
        self.steer.clear()
        self.speed.clear()
        self.position.clear()
        self.vel_target.clear()
        self.pos_target.clear()
        self.lat_error.clear()
        self.lon_error.clear()
        self.time.clear()
        
        self.action = 0
        
    def mcts_step(self, verbose = False, recording = False, action = 0):
        self.action = action
        #set pid offset to the action
        target_offset = self.available_actions[self.action]
        self.pid.set_offset(target_offset)

        for _ in range(self.decision_period):
            if recording:
                self.world.render(self.display)
                pygame.display.flip()
            #Get ego vehicle variables and run PID controller
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
            self.lon_error.append(self.pid.longitudinal_controller.error)
            self.lat_error.append(self.pid.lateral_controller.error)
            self.time.append(self.prev_timestamp.elapsed_seconds - self.first_timestamp.elapsed_seconds)
        
            #Tick the clock to advance the simulation
            self.clock.tick()
            self.world.world.tick()
            self.world.tick(self.clock, self.episode_counter, self.frame_counter)

            prev_timestamp = self.world.world.get_snapshot().timestamp
            self.frame_counter += 1

        # if recording:
        #     self.world.render(self.display)
        #     pygame.display.flip()
        
        print(f"Saving simulation state on decision counter {self.decision_counter}")
        self.state_manager.save_frame(frame_number=self.decision_counter, vehicle_list=self.world.actor_list)
        self.decision_counter += 1
        # for state in self.state_manager.frame_list[self.decision_counter]:
        #     state.display()
    
    def is_terminal(self):
        """
        Method for checking if the current state is terminal
        """
        ret_dict = {
            "frame_counter": False,
            "pid": False,
            "collision": False,
        }
        if self.frame_counter >= self.frame_limit:
            ret_dict["frame_counter"] = True
        if self.pid.done:
            ret_dict["pid"] = True
        if len(self.world.collision_sensor.get_collision_history()) > 0:
            ret_dict["collision"] = True

        ret = any(ret_dict.values())
        if ret:
            print_blue("Terminal state: ")
            for key, value in ret_dict.items():
                print(key,": ", value)
        
        return ret
    
    def get_reward(self):
        """
        Method for getting the reward for the current state
        """
        reward = 0
        if self.pid.done:
            reward = 100
        if len(self.world.collision_sensor.get_collision_history()) > 0:
            reward = -100
        return reward
    
    def get_state(self, decision_index):
        """
        Method for getting the state at the current decision index
        """
        if decision_index is None:
            decision_index = self.decision_counter
        self.state_manager.save_frame(frame_number=decision_index, vehicle_list=self.world.actor_list)
        self.potential_field.update(self.state_manager.return_frame_history(frame_number=decision_index, history_length=5))
        pf = self.potential_field.calculate_field()
        state = self.autoencoder.encode(pf).flatten()

        return state
    
    def game_step(self, verbose = False, recording = False):
        self.clock.tick()
        self.world.world.tick()
        self.world.tick(self.clock, self.episode_counter, self.frame_counter)

        timestamp = self.world.world.get_snapshot().timestamp

        if self.display:
            self.world.render(self.display)
            pygame.display.flip()

        if self.controller and self.controller.parse_events():
            return -1, -1, prev_timestamp
                    
        target_offset = self.available_actions[self.action]
        if self.frame_counter % self.decision_period == 0 and self.action is not None:
            self.decision_counter += 1
            print("Decision counter: ", self.decision_counter)
            if self.decision_counter % 15 == 0:
                print("Toggling action")
                self.action = 0 if self.action == 2 else 2
                print("New action: ", self.action)
                target_offset = self.available_actions[self.action]
                self.pid.set_offset(target_offset)


        self.state_manager.save_frame(frame_number=self.frame_counter, vehicle_list=self.world.actor_list)
        self.potential_field.update(self.state_manager.return_frame_history(frame_number=self.frame_counter, history_length=5))
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
        self.lon_error.append(self.pid.longitudinal_controller.error)
        self.lat_error.append(self.pid.lateral_controller.error)
        self.time.append(timestamp.elapsed_seconds - self.first_timestamp.elapsed_seconds)
        
        # #print control signals, errors and values for debugging
        # if 1:
        # #     print()
        # #     print("Control: ", control)
        # #     print("Lateral control:")
        # #     print("Lat error: ", self.pid.lateral_controller.error)
        #     print("Current location: ", self.world.player.get_location())
        #     print("Target location: ", self.pid.lateral_controller.target_location)
        #     print("Frame: ", self.frame_counter)

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
            
        return 1
    
    def plot_results(self):
        """
        Plots the results of the simulation.
        """
        width = 2
        figsize = (15, 15)
        # Speed and Throttle/Brake figure
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        ax[0].plot(self.time,self.speed, label="Speed", linewidth=width)
        ax[0].plot(self.time,self.vel_target, label="Target Speed", linewidth=width)
        ax[0].plot(self.time,self.lon_error, label="Longitudinal Error", linewidth=width)
        ax[0].set_title("Speed")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Speed (m/s)")
        ax[0].legend()

        ax[1].plot(self.time,self.throttle_brake, label="Throttle/Brake", linewidth=width)
        ax[1].plot(self.time, np.gradient(self.throttle_brake), label="ThrottleBrake Derivative", linewidth=width)
        ax[1].set_title("Throttle/Brake")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Throttle/Brake")
        ax[1].legend()

        # Steer and Lateral control figure
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        ax[0].plot(self.time , [0]*len(self.time), label="Lateral error reference", linewidth=width)
        ax[0].plot(self.time,self.lat_error, label="Lateral Error", linewidth=width)
        ax[0].set_title("Lateral control")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Yaw")
        ax[0].legend()

        ax[1].plot(self.time,self.steer, label="Steer", linewidth=width)
        ax[1].plot(self.time, np.gradient(self.steer), label="Steer Derivative", linewidth=width)
        ax[1].set_title("Steer")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Steer")
        ax[1].legend()

        # Position figure
        plt.figure(figsize=figsize)
        plt.plot([pos.x for pos in self.position], [pos.y for pos in self.position], label="Position", linewidth=width)
        plt.scatter([pos.x for pos in self.pos_target], [pos.y for pos in self.pos_target], label="Target Position", linewidth=width)
        plt.title("Position")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

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
            self.decision_counter = 0
            print("Restored frame state")
            print("Finished episode ", self.episode_counter, " initializing next episode")
            self.episode_counter += 1
            self.clock.tick()
            self.world.world.tick()
            self.init_game()
