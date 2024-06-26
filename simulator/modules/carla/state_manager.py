### Description:
# This module contains the StateManager class, which is responsible for managing the state of vehicles in the simulation.
# The StateManager class is used to record the state of vehicles at each frame, restore the state of vehicles to a specific 
# frame, and return the state history of vehicles to calculate a potential field instance.

# The VehicleState class represents the state of a vehicle in the simulation at a specific frame. It contains attributes such as
# the vehicle's position, orientation, velocity, angular velocity, width, height, and whether it is the hero vehicle.


### Dataclass and type hinting imports
from dataclasses import dataclass, asdict
from typing import List

### Main imports
import glob
import numpy as np
import os
import pandas as pd
import sys

### Partial imports
from rich import print

### Carla imports
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

@dataclass
class VehicleState:
    """
    Represents the state of a vehicle in the simulation at a specific frame.

    Attributes:
        id (int): Unique identifier for the vehicle actor.
        x (float): Current x position of the vehicle.
        y (float): Current y position of the vehicle.
        z (float): Current z position of the vehicle.
        pitch (float): Current pitch of the vehicle in radians.
        yaw (float): Current yaw of the vehicle in radians.
        roll (float): Current roll of the vehicle in radians.
        xVelocity (float): Current x velocity of the vehicle.
        yVelocity (float): Current y velocity of the vehicle.
        zVelocity (float): Current z velocity of the vehicle.
        xAngVelocity (float): Current x angular velocity of the vehicle in radians.
        yAngVelocity (float): Current y angular velocity of the vehicle in radians.
        zAngVelocity (float): Current z angular velocity of the vehicle in radians.
        width (float): Width of the bounding box of the vehicle.
        height (float): Height of the bounding box of the vehicle.
        hero (bool): Indicates if the vehicle is the hero vehicle.
        frame (int): Frame number when the state was recorded.
    """
    id: int
    x: float
    y: float
    z: float
    pitch: float
    yaw: float
    roll: float
    xVelocity: float
    yVelocity: float
    zVelocity: float
    xAngVelocity: float
    yAngVelocity: float
    zAngVelocity: float
    width: float
    height: float
    hero: bool
    frame: int

    def display(self):
        """
        Prints the state of the vehicle.
        """
        print()
        print("==== Vehicle State ====")
        print(f"ID: {self.id}")
        print(f"Position: ({self.x}, {self.y}, {self.z})")
        print(f"Orientation: ({self.pitch}, {self.yaw}, {self.roll})")
        print(f"Velocity: ({self.xVelocity}, {self.yVelocity}, {self.zVelocity})")
        print(f"Angular Velocity: ({self.xAngVelocity}, {self.yAngVelocity}, {self.zAngVelocity})")
        print(f"Width: {self.width}")
        print(f"Height: {self.height}")
        print(f"Hero: {self.hero}")
        print(f"Frame: {self.frame}")

    def dict(self):
        """
        Returns the state as a dictionary.

        Returns:
            dict: Dictionary representation of the state.
        """
        return asdict(self)


class StateManager:
    """
    Manages the state of vehicles in the simulation.

    Attributes:
        vehicle_states (list): List to store VehicleState objects.
    """
    def __init__(self):
        self.frame_list = [] #index of list is frame number, each element is a list of vehicle states

    def save_vehicle_state(self, vehicle, frame_number):
        """
        Records the state of a vehicle at the specified frame number.

        Args:
            vehicle (carla.Vehicle): The vehicle whose state is to be saved.
            frame_number (int): The frame number when the state is recorded.
        """
        if vehicle.attributes["role_name"] == "hero":
            hero = 1
        else:
            hero = 0
        
        position = vehicle.get_location()
        rotation = vehicle.get_transform().rotation
        velocity = vehicle.get_velocity()
        ang_velocity = vehicle.get_angular_velocity()
        bounding_box = vehicle.bounding_box
        state_dict = {
            "id": vehicle.id,
            "x": position.x,
            "y": position.y,
            "z": position.z,
            "pitch": np.deg2rad(rotation.pitch),
            "yaw": np.deg2rad(rotation.yaw),
            "roll": np.deg2rad(rotation.roll),
            "xVelocity": velocity.x,
            "yVelocity": velocity.y,
            "zVelocity": velocity.z,
            "xAngVelocity": np.deg2rad(ang_velocity.x),
            "yAngVelocity": np.deg2rad(ang_velocity.y),
            "zAngVelocity": np.deg2rad(ang_velocity.z),
            "width": 2 * bounding_box.extent.x,
            "height": 2 * bounding_box.extent.y,
            "hero": hero,
            "frame": frame_number,
        }
        state = VehicleState(**state_dict)
        return state

    def save_frame(self, frame_number: int, vehicle_list: List[carla.Actor]):
        """
        Saves the current frame, with all vehicles that are present in the simulation.

        Args:
            frame_number (int): The frame number to save.
            vehicle_list (list): List of all vehicles in the simulation.
        """
        frame = []
        for vehicle in vehicle_list:
            state = self.save_vehicle_state(vehicle, frame_number)
            frame.append(state)
        self.frame_list.append(frame)
        
    def restore_frame(self, frame_number: int, vehicle_list: List[carla.Actor]):
        """
        Restores the state of all vehicles to the specified frame.

        Args:
            frame_number (int): The frame number to restore.
            vehicle_list (list): List of all vehicles in the simulation.
        """
        for vehicle in vehicle_list:
            self.restore_vehicle_state(vehicle, frame_number)
        #Remove all frames after the specified frame number
        self.frame_list = self.frame_list[:frame_number+1]
        
    def restore_vehicle_state(self, vehicle: carla.Vehicle, target_frame_number: int):
        """
        Restores the state of a vehicle to the specified frame.

        Args:
            vehicle (carla.Vehicle): The vehicle to restore the state.
            target_frame_number (int): The frame number to restore the state.
        """
        #Find the vehicle state that matches the vehicle_id and the target_frame_number
        frame_state = self.frame_list[target_frame_number]
        matching_vehicle_state = None
        for state in frame_state:
            if state.id == vehicle.id:
                matching_vehicle_state = state
                break
        if matching_vehicle_state is None:
            raise ValueError("No matching vehicle state found.")
        #Restore the vehicle state
        vehicle.set_transform(
            carla.Transform(
                carla.Location(matching_vehicle_state.x, matching_vehicle_state.y, matching_vehicle_state.z),
                carla.Rotation(
                    np.rad2deg(matching_vehicle_state.pitch),
                    np.rad2deg(matching_vehicle_state.yaw),
                    np.rad2deg(matching_vehicle_state.roll)
                )
            )
        )
        vehicle.set_target_velocity(
            carla.Vector3D(
                matching_vehicle_state.xVelocity,
                matching_vehicle_state.yVelocity,
                matching_vehicle_state.zVelocity
            )
        )
        vehicle.set_target_angular_velocity(
            carla.Vector3D(
                np.rad2deg(matching_vehicle_state.xAngVelocity),
                np.rad2deg(matching_vehicle_state.yAngVelocity),
                np.rad2deg(matching_vehicle_state.zAngVelocity)
            )
        )
    
    def return_frame_history(self, frame_number: int, history_length: int):
        """
        Returns an appropiate dataframe to calculate a potential field instance.

        Args:
            frame_number (int): The frame number to return the state.
            history_length (int): The length of the history to return.

        Returns:
            pd.DataFrame: A dataframe with the state history of all vehicles
        """
        #Get the frame number to start the history
        start_frame = frame_number - history_length
        if start_frame < 0:
            start_frame = 0
        #Get the frame list to construct the dataframe
        frame_list = self.frame_list[start_frame:frame_number+1]
        #Create a dataframe with the state history of all vehicles
        cols = frame_list[0][0].dict().keys()
        lst = []
        for frame in frame_list:
            for vehicle_state in frame:
                lst.append(vehicle_state.dict().values())
        df = pd.DataFrame(lst, columns=cols)
        return df

    def reset(self):
        """
        Resets the state manager.
        """
        self.frame_list = []
