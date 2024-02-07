
from __future__ import print_function

# ==============================================================================
# -- Python imports ------------------------------------------------------------
# ==============================================================================

import glob
import numpy as np
import os
import pandas as pd
import re
import sys

from rich import print
from numpy import random

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

from .camera import CameraManager, StaticCamera
from .hud import HUD, get_actor_display_name
from .keyboard_control import KeyboardControl
from .printers import print_blue, print_green, print_highlight, print_red
from .sensors import CollisionSensor, GnssSensor, LaneInvasionSensor
from .utils import get_straight_angle

from ..agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from ..agents.navigation.behavior_agent import (
    BehaviorAgent,
)  # pylint: disable=import-error
from ..agents.navigation.constant_velocity_agent import (
    ConstantVelocityAgent,
)  # pylint: disable=import-error

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")

    def name(x):
        return " ".join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]

class World(object):
    """Class representing the surrounding environment"""

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        self.delta_simulated = 0.1

        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print("RuntimeError: {}".format(error))
            print("The server could not send the OpenDRIVE (.xodr) file:")
            print(
                "Make sure it exists, has the same name of your town, and is correct."
            )
            sys.exit(1)

        self.hud = hud
        self.actor_list = []
        self.sensor_list = []
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self.static_camera = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0
        self.blueprint_toyota_prius = None

        self.dataframe_record = {}

        self.distance = 300
        self.spawn_point_ego = carla.Transform(
            carla.Location(x=-850, y=-65, z=0.5), carla.Rotation(yaw=0, pitch=0, roll=0)
        ) #Spwan point for ego vehicle
        
        self.destination = carla.Location(
            x=self.spawn_point_ego.location.x + self.distance,
            y=self.spawn_point_ego.location.y - 2, #straight in map is not completely straight
            z=self.spawn_point_ego.location.z,
        ) #Destination for ego vehicle

        self.static_camera_flag = 0

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_simulated
        settings.no_rendering_mode = True
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = int(self.delta_simulated * 50)
        self.world.apply_settings(settings)
        self.restart(args)

        print("World created")

    def spawn_ego_vehicle(self):
        """Spawn ego vehicle"""

        # Get a random blueprint.
        self.blueprint_toyota_prius = random.choice(
            self.world.get_blueprint_library().filter("vehicle.toyota.prius")
        )
        self.blueprint_toyota_prius.set_attribute("role_name", "hero")
        self.blueprint_toyota_prius.set_attribute("color", "(11,166,86)")

        if not self.map.get_spawn_points():
            print("There are no spawn points available in your map/town.")
            print("Please add some Vehicle Spawn Point to your UE4 scene.")
            sys.exit(1)

        # Spawn the player.
        self.player = self.world.try_spawn_actor(
            self.blueprint_toyota_prius, self.spawn_point_ego
        )
        self.modify_vehicle_physics(self.player)
        print("Spawned ego vehicle")
        self.world.tick()

    def setup_sensors(self):
        """Set up sensors"""

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)

        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.set_sensor(0, notify=False)

        self.actor_list.append(self.player)
        self.sensor_list.extend(
            [
                self.collision_sensor.sensor,
                self.lane_invasion_sensor.sensor,
                self.gnss_sensor.sensor,
                self.camera_manager.sensor,
            ]
        )

    def setup_static_camera(self, width, height):
        """Set up static camera"""

        self.static_camera = StaticCamera(
            carla.Transform(
                carla.Location(x=-475, y=-67, z=100),
                carla.Rotation(yaw=0, pitch=-75, roll=0),
            ),
            self.world,
            width,
            height,
        )
        self.static_camera.set_sensor()
        return 1

    def restart(self, args):
        """Restart the world"""
        
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 0
        )

        self.spawn_ego_vehicle()
        self.setup_sensors()

        if args.static_camera:
            self.static_camera_flag = self.setup_static_camera(args.width, args.height)

        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def record_frame_state(self, frame_number):
        """
        Saves the state of all vehicles in the world at the specified frame number.

        Args:
            frame_number (int): The frame number when the state is recorded.
        """
        local_df = pd.DataFrame()
        for vehicle in self.world.get_actors().filter("vehicle.*"):
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
                "pitch": rotation.pitch,
                "yaw": rotation.yaw,
                "roll": rotation.roll,
                "xVelocity": velocity.x,
                "yVelocity": velocity.y,
                "zVelocity": velocity.z,
                "xAngVelocity": ang_velocity.x,
                "yAngVelocity": ang_velocity.y,
                "zAngVelocity": ang_velocity.z,
                "width": 2 * bounding_box.extent.x,
                "height": 2 * bounding_box.extent.y,
                "hero": hero
            }
        local_df = pd.concat([local_df, pd.DataFrame([state_dict])], ignore_index=True)
        self.dataframe_record[frame_number] = local_df

    def return_frame_history(self, frame_number: int, history_length: int):
        """
        Returns an appropiate dataframe to calculate a potential field instance.

        Args:
            frame_number (int): The frame number to return the state.
            history_length (int): The length of the history to return.

        Returns:
            pd.DataFrame: A dataframe with the state history of all vehicles
        """
        history = pd.DataFrame()
        for i in range(history_length):
            history = pd.concat(
                [history, self.dataframe_record[frame_number - i]], ignore_index=True
            )
        return history

    def restore_frame_state(self, frame_number):
        """
        Restores the state of all vehicles in the world to the specified frame.

        Args:
            frame_number (int): The frame number to restore the state.
        """
        groups = self.dataframe.groupby("frame")
        frame_df = groups.get_group(frame_number)
        for index, row in frame_df.iterrows():
            actor = self.world.get_actor(int(row["id"]))
            actor.set_transform(
                carla.Transform(
                    carla.Location(x=row["x"], y=row["y"], z=row["z"]),
                    carla.Rotation(
                        pitch=row["pitch"], yaw=row["yaw"], roll=row["roll"]
                    ),
                )
            )
            actor.set_target_velocity(
                carla.Vector3D(
                    x=row["xVelocity"], y=row["yVelocity"], z=row["zVelocity"]
                )
            )
            actor.set_target_angular_velocity(
                carla.Vector3D(
                    x=row["xAngVelocity"],
                    y=row["yAngVelocity"],
                    z=row["zAngVelocity"],
                )
            )

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        # If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock, episode_number, frame_number):
        """Method for every tick"""
        self.hud.tick(self, clock, episode_number, frame_number)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)
        # self.static_camera.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""

        [sensor.destroy() for sensor in self.sensor_list]
        [actor.destroy() for actor in self.actor_list]
        self.actor_list.clear()
        self.sensor_list.clear()

        # sensors = list(self.world.get_actors().filter("sensor.*"))
        # [print(sensor.type_id, sensor.id) for sensor in sensors if sensor is not None]
        # print("_" * 20)
        # [sensor.destroy() for sensor in sensors]
        # actors = list(self.world.get_actors().filter("vehicle.*"))
        # [print(actor.type_id, actor.id) for actor in actors]
        # [actor.destroy() for actor in actors if actor is not None]

        # self.players.clear()
        print("Finished destroying actors")