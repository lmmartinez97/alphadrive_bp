#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""CARLA MCTS IMPLEMENTATION"""

from __future__ import print_function

import argparse
import glob
import logging
import os
import numpy as np
import re
import sys
import traceback

from rich import print
from numpy import random


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

from modules.camera import CameraManager, StaticCamera
from modules.hud import HUD, get_actor_display_name
from modules.keyboard_control import KeyboardControl
from modules.logger import DataLogger
from modules.printers import print_blue, print_green, print_highlight, print_red
from modules.sensors import CollisionSensor, GnssSensor, LaneInvasionSensor
from modules.shared_mem import SharedMemory
from modules.state_manager import VehicleState, StateManager

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.constant_velocity_agent import ConstantVelocityAgent  # pylint: disable=import-error

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)")

    def name(x):
        return " ".join(m.group(0) for m in rgx.finditer(x))

    presets = [x for x in dir(carla.WeatherParameters) if re.match("[A-Z].+", x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


# ==============================================================================
# -- Destination reached exception ---------------------------------------------
# ==============================================================================


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    """Class representing the surrounding environment"""

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        self.delta_simulated = 0.05

        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print("RuntimeError: {}".format(error))
            print("The server could not send the OpenDRIVE (.xodr) file:")
            print("Make sure it exists, has the same name of your town, and is correct.")
            sys.exit(1)

        self.hud = hud
        self.actor_list = []
        self.sensor_list = []
        self.player_ego = None
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


        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.delta_simulated
        settings.no_rendering_mode = False
        settings.substepping = True
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = int(self.delta_simulated * 100)
        self.world.apply_settings(settings)
        self.restart(args)

        self.state_manager = StateManager()  # Creating StateManager instance
        print("World created")

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = (
            self.camera_manager.transform_index
            if self.camera_manager is not None
            else 0
        )

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
        spawn_points = self.map.get_spawn_points()
        spawn_point_ego = random.choice(spawn_points)
        self.player_ego = self.world.try_spawn_actor(
            self.blueprint_toyota_prius, spawn_point_ego
        )
        self.modify_vehicle_physics(self.player_ego)
        print("Spawned ego vehicle")
        self.world.tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player_ego, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player_ego, self.hud)
        self.gnss_sensor = GnssSensor(self.player_ego)

        self.camera_manager = CameraManager(self.player_ego, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)

        self.actor_list.append(self.player_ego)
        self.sensor_list.extend([self.collision_sensor.sensor, self.lane_invasion_sensor.sensor, self.gnss_sensor.sensor, self.camera_manager.sensor]) 

        # self.static_camera = StaticCamera(
        #     carla.Transform(
        #         carla.Location(x=-475, y=-67, z=100),
        #         carla.Rotation(yaw=0, pitch=-75, roll=0),
        #     ),
        #     self.world,
        #     args.width,
        #     args.height,
        # )
        # self.static_camera.set_sensor()

        actor_type = get_actor_display_name(self.player_ego)
        self.hud.notification(actor_type)

    def save_frame_state(self, frame_number):
        """
        Saves the state of all vehicles in the world at the specified frame number.

        Args:
            frame_number (int): The frame number when the state is recorded.
        """
        for vehicle in self.world.get_actors().filter("vehicle.*"):
            self.state_manager.save_vehicle_state(vehicle, frame_number)

    def restore_frame_state(self, frame_number):
        """
        Restores the state of all vehicles in the world to the specified frame.

        Args:
            frame_number (int): The frame number to restore the state.
        """
        for vehicle in self.world.get_actors().filter("vehicle.*"):
            self.state_manager.restore_vehicle_state(vehicle, frame_number)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification("Weather: %s" % preset[1])
        self.player_ego.get_world().set_weather(preset[0])

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
        #self.static_camera.render(display)

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

def game_loop(args, episode_counter):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """
    print("I'm in game loop")

    ret = 0

    pygame.init()
    pygame.font.init()
    world = None
    frame_counter = 0

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(6.0)

        #get traffic manager
        traffic_manager = client.get_trafficmanager()
        sim_world = client.get_world()
        # apply settings
        settings = sim_world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        sim_world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        #initialize display
        display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        print("Created display")

        #initialize hud and world
        hud = HUD(args.width, args.height, text = __doc__)
        print("Created hud")
        world = World(sim_world, hud, args)
        print("Created world instance")
        
        # Draw waypoints in the map
        if 0:
            waypoints = world.map.generate_waypoints(0.05)
            for w in waypoints:
                world.world.debug.draw_string(
                    w.transform.location,
                    "O",
                    draw_shadow=False,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=120.0,
                    persistent_lines=True,
                )

        controller = KeyboardControl(world)

        #initialize agent
        if args.agent == "Basic":
            agent = BasicAgent(world.player_ego, 30)
            agent.follow_speed_limits(True)
        elif args.agent == "Constant":
            agent = ConstantVelocityAgent(world.player_ego, 30)
            ground_loc = world.world.ground_projection(world.player_ego.get_location(), 5)
            if ground_loc:
                world.player.set_location(ground_loc.location + carla.Location(z=0.01))
            agent.follow_speed_limits(True)
        elif args.agent == "Behavior":
            agent = BehaviorAgent(world.player_ego, behavior=args.behavior)

        # Set the agent destination
        spawn_points = world.map.get_spawn_points()
        destination = random.choice(spawn_points).location
        agent.set_destination(destination)
        print("Spawn point is: ", world.player_ego.get_location())
        print("Destination is: ", destination)

        prev_timestamp = world.world.get_snapshot().timestamp
        simulated_time = 0

        clock = pygame.time.Clock()

        while True:
            clock.tick()
            timestamp = world.world.get_snapshot().timestamp

            world.world.tick()
            # controller.parse_events()
            world.tick(clock, episode_counter, frame_counter)
            world.render(display)
            pygame.display.flip()

            if agent.done():
                print("The target has been reached, stopping the simulation")
                break

            simulated_time = (
                simulated_time
                + timestamp.elapsed_seconds
                - prev_timestamp.elapsed_seconds
            )

            control = agent.run_step()
            control.manual_gear_shift = False
            world.player_ego.apply_control(control)

            prev_timestamp = timestamp
            frame_counter+= 1

    except KeyboardInterrupt as e:
        print("\n")
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            world.destroy()
            world = None
        pygame.quit()
        return -1

    except Exception as e:
        print(traceback.format_exc())

    finally:
        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            world.destroy()

        print("Bye, bye")
        pygame.quit()
        return -1


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

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

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)

    print(__doc__)
    episode_counter = 0
    try:
        while True:
            ret = game_loop(args, episode_counter)
            if ret:
                sys.exit()
            episode_counter += 1

    except KeyboardInterrupt:
        print("\nCancelled by user. Bye!")


if __name__ == "__main__":
    main()
