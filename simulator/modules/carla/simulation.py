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
from .hud import HUD
from .potential_field import PotentialField
from .mpc import MPCController
from ..agents.navigation.global_route_planner import GlobalRoutePlanner
from ..agents.navigation.behavior_types import Cautious, Normal, Aggressive

from .autoencoder import load_model

class Simulation:
  
    def __init__(self, args = None, frame_limit = None, episode_limit = 100, options = None):
        """
        Constructor method for simulation class
        Args:
            args (argparse.Namespace): Arguments for the simulation
            frame_limit (int): The number of frames to run the simulation for
            episode_limit (int): The number of episodes to run the simulation for
            options (dict): Dictionary of options for the simulation
        """
        # simulation variables
        print()
        self.episode_counter = 0
        self.frame_counter = 0
        self.decision_counter = 0
        self.reward = 0
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
        self.agent_type = self.agents_dict[self.args.behavior]
        self.agent_type.max_speed = 35 if options is None else options.get("ego_speed", 10)*3.6 # m/s to km/h
        
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
        
        #load autoencoder -- remove these features for now until mpc works
        self.autoencoder = load_model(model_name= "autoencoder_3" ,directory="../scene_representation/training/saved_models")
         #Initialize Potential Field -- remove these features for now until mpc works
        rx = 50 # horizontal semiaxis of ellipse to consider as ROI
        ry = 6 # vertical semiaxis of ellipse to consider as ROI
        sx = 0.5 # step size in x direction
        sy = 0.1 # step size in y direction
        self.potential_field = PotentialField(radius_x = rx, radius_y = ry, step_x = sx, step_y = sy)
 
        #Initialize GlobalRoutePlanner
        self.grp = GlobalRoutePlanner(wmap=self.world.map, sampling_resolution= self.simulation_period * self.agent_type.max_speed / 3.6) 

        #initialize MPC controller
        parameters = {
            'mass': 1500,
            'L': 2.7, #wheelbase
            'a': 1.2, #distance from CoG to front axle
            'b': 1.5, #distance from CoG to rear axle
            'frontal_area': 2.4,
            'drag_coefficient': 0.24,
            'max_acceleration': 2.5,
            'max_deceleration': 4,
            'air_density': 1.2,
            'gravity': 9.81,
            'dt': self.simulation_period,
            'prediction_horizon': 4,
            'max_steering_rate': 0.02,
            'max_pedal_rate': 0.02,
            'tracking_cost_weight': 3,
            'velocity_cost_weight': 4,
            'yaw_cost_weight': 2,
            'steering_rate_cost_weight': 0, #DISABLED IN MPC CODE
            'pedal_rate_cost_weight': 0, #DISSABLED IN MPC CODE
            'exponential_decay_rate': 0.65,
        }
        self.mpc = MPCController(parameters)
        
        self.controller = KeyboardControl(self.world)
        self.clock = pygame.time.Clock()

    def init_game(self):
        """Method for initializing a new episode"""
        print("Initializing new game")
        self.world.restart(self.args)
        self.state_manager.reset()

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
        self.reference = [[[waypoint.transform.location.x, 
                            waypoint.transform.location.y, 
                            waypoint.transform.location.z], 
                           self.agent_type.max_speed] for waypoint, _ in self.route[1:]]
        self.mpc.update_reference(self.reference)

        self.world.dataframe = pd.DataFrame()
        self.timestamp = self.world.world.get_snapshot().timestamp
        self.first_timestamp = self.timestamp

        for waypoint, _ in self.route:
            self.world.world.debug.draw_point(
                waypoint.transform.location,
                size=0.1,
                color=carla.Color(255, 0, 0),
                life_time=100,
            )
        self.world.player.set_target_velocity(carla.Vector3D(x=self.agent_type.max_speed/3.6, y=0, z=0))
        self.world.world.tick()
        self.world.world.tick()
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
        print("Action: ", self.action)
        #set pid offset to the action
        target_offset = self.available_actions[self.action]
        self.mpc.set_offset(target_offset)

        for _ in range(self.decision_period):
            if recording:
                self.world.render(self.display)
                pygame.display.flip()
            #Get ego vehicle variables and run MPC controller
            velocity = self.world.player.get_velocity()
            speed = np.linalg.norm([velocity.x, velocity.y])
            transform = self.world.player.get_transform()
            mpc_state = {
                'x': transform.location.x,
                'y': transform.location.y,
                'yaw': np.deg2rad(transform.rotation.yaw),
                'velocity': speed,
                'pitch': np.deg2rad(transform.rotation.pitch), #used to calculate weight component in the direction of the vehicle
            }
            control_actions, predictions = self.mpc(state=mpc_state)
            control = carla.VehicleControl(
                steer=control_actions['steering'],
                throttle=control_actions['throttle'],
                brake=control_actions['brake'],
                hand_brake=False,
                manual_gear_shift=False
            )       
            self.world.player.apply_control(control)
            #calculate errors
            error_dict = self.mpc.calculate_errors(mpc_state, verbose=False)
            #record control signals for plotting
            self.yaw.append(self.world.player.get_transform().rotation.yaw)
            self.throttle_brake.append(control.throttle - control.brake)
            self.steer.append(control.steer)
            self.speed.append(speed)
            self.position.append(self.world.player.get_location())
            self.vel_target.append(error_dict['velocity_target'])
            self.pos_target.append(error_dict['pos_target'])
            self.lon_error.append(error_dict['velocity_error'])
            self.lat_error.append(error_dict['lateral_error'])
            self.time.append(self.timestamp.elapsed_seconds - self.first_timestamp.elapsed_seconds)
        
            #Tick the clock to advance the simulation
            self.clock.tick()
            self.world.world.tick()
            self.world.tick(self.clock, self.episode_counter, self.frame_counter)

            self.timestamp = self.world.world.get_snapshot().timestamp
            self.frame_counter += 1

        # if recording:
        #     self.world.render(self.display)
        #     pygame.display.flip()
        
        self.state_manager.save_frame(frame_number=self.decision_counter, vehicle_list=self.world.actor_list)
        self.decision_counter += 1
        # for state in self.state_manager.frame_list[self.decision_counter]:
        #     state.display()
    
    def is_terminal(self):
        """
        Method for checking if the current state is terminal
        """
        self.termination_dict = {
            "frame_counter": False,
            "mpc": False,
            "collision": False,
        }
        if self.frame_counter >= self.frame_limit:
            self.termination_dict["frame_counter"] = True
            self.frame_counter = -1
            self.reward = -100
        if self.mpc.is_done():
            self.termination_dict["mpc"] = True
            self.reward = 100
        if len(self.world.collision_sensor.get_collision_history()) > 0:
            self.termination_dict["collision"] = True
            self.reward = -100
            self.world.collision_sensor.history.clear()

        ret = any(self.termination_dict.values())
        if ret:
            print_blue("Terminal state: ")
            for key, value in self.termination_dict.items():
                print(key,": ", value)
        
        return ret
    
    def get_reward(self):
        """
        Method for getting the reward for the current state
        """
        return self.reward
    
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

        # if self.controller and self.controller.parse_events():
        #     return -1, -1, prev_timestamp
                    
        # decision loop
        if self.frame_counter % self.decision_period == 0 and self.action is not None:
            self.decision_counter += 1
            self.state_manager.save_frame(frame_number=self.decision_counter, vehicle_list=self.world.actor_list)
            self.potential_field.update(self.state_manager.return_frame_history(frame_number=self.decision_counter, history_length=5))
            #Return potential field dropping last column and row to match the autoencoder input shape
            pf = self.potential_field.calculate_field()
            state = self.autoencoder.encode(pf).flatten()
            if self.decision_counter % 15 == 0:
                print("Toggling action")
                self.action = 0 if self.action == 2 else 2
                print("New action: ", self.action)
                self.mpc.set_offset(self.available_actions[self.action])        
        
        # control loop
        velocity = self.world.player.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y])
        transform = self.world.player.get_transform()
        mpc_state = {
            'x': transform.location.x,
            'y': transform.location.y,
            'yaw': np.deg2rad(transform.rotation.yaw),
            'velocity': speed,
            'pitch': np.deg2rad(transform.rotation.pitch), #used to calculate weight component in the direction of the vehicle
        }
        control_actions, predictions = self.mpc(state=mpc_state)
        control = carla.VehicleControl(
            steer=control_actions['steering'],
            throttle=control_actions['throttle'],
            brake=control_actions['brake'],
            hand_brake=False,
            manual_gear_shift=False
        )       
        self.world.player.apply_control(control)
        
        #calculate errors
        error_dict = self.mpc.calculate_errors(mpc_state, verbose=False)
        
        #record control signals for plotting
        self.yaw.append(self.world.player.get_transform().rotation.yaw)
        self.throttle_brake.append(control.throttle - control.brake)
        self.steer.append(control.steer)
        self.speed.append(speed)
        
        self.position.append(self.world.player.get_location())
        self.vel_target.append(error_dict['velocity_target'])
        self.pos_target.append(error_dict['pos_target'])
        self.lon_error.append(error_dict['velocity_error'])
        self.lat_error.append(error_dict['lateral_error'])
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
            # print("State: ", state)
            # print("State len: ", len(state))
            print("Frame: ", self.frame_counter)
            print("Control action: ", control)
            # self.potential_field.plot_field()
            # self.autoencoder.compare(pf, num_plots=1)
            input("Press Enter to continue")
            
        #plot the predicted trajectory using carla modules
        for idx in range(len(predictions)-1):
            self.world.world.debug.draw_arrow(
                begin=carla.Location(x=predictions[idx]['x'], y=predictions[idx]['y'], z=2),
                end=carla.Location(x=predictions[idx+1]['x'], y=predictions[idx+1]['y'], z=2),
                thickness=0.05,
                arrow_size=0.05,
                color=carla.Color(0, 0, 255),
                life_time=0.5,
            )
            
            
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
        ax[1].plot(self.time[:-1], [self.throttle_brake[i+1]-self.throttle_brake[i] for i,_ in enumerate(self.throttle_brake[:-1])], label="ThrottleBrake increment", linewidth=width)
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
        ax[0].set_ylabel("Later Error (m)")
        ax[0].legend()

        ax[1].plot(self.time,self.steer, label="Steer", linewidth=width)
        ax[1].plot(self.time[:-1], [self.steer[i+1]-self.steer[i] for i,_ in enumerate(self.steer[:-1])], label="Steer increment", linewidth=width)
        ax[1].set_title("Steer")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Steer")
        ax[1].legend()

        # Position figure
        plt.figure(figsize=figsize)
        plt.plot([pos.x for pos in self.position], [pos.y for pos in self.position], label="Position", linewidth=width)
        plt.scatter([pos[0] for pos in self.pos_target], [pos[1] for pos in self.pos_target], label="Target Position", linewidth=width)
        plt.title("Position")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        # Acceleration and jerk figure
        fig, ax = plt.subplots(2, 1, figsize=figsize)
        ax[0].plot(self.time, np.gradient(self.speed, self.simulation_period), label="Acceleration", linewidth=width)
        ax[0].set_title("Acceleration")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Acceleration (m/s^2)")
        ax[0].legend()

        ax[1].plot(self.time, np.gradient(np.gradient(self.speed, self.simulation_period), self.simulation_period), label="Jerk", linewidth=width)
        ax[1].set_title("Jerk")
        ax[1].set_xlabel("Time")
        ax[1].set_ylabel("Jerk (m/s^3)")
        ax[1].legend()
        
        plt.show()

    def run(self):
        """Method for running the simulation"""
        self.init_game()
        while self.episode_counter < self.episode_limit:
            while (self.frame_counter < self.frame_limit) and not self.mpc.is_done():
                if self.frame_counter % 100 == 0:
                    verbose = False
                self.game_step(verbose=verbose)
                self.frame_counter += 1
            # self.plot_results()
            # input("Press Enter to continue")
            self.frame_counter = 0
            self.decision_counter = 0
            print("Restored frame state")
            print("Finished episode ", self.episode_counter, " initializing next episode")
            self.episode_counter += 1
            self.clock.tick()
            self.world.world.tick()
            self.init_game()
