#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""CARLA MCTS IMPLEMENTATION"""

from __future__ import print_function


import argparse
import asyncio
import glob
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import sys
import traceback

from rich import print
from numpy import random

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
from modules.simulation import Simulation
# # ==============================================================================
# # -- Global parameters ----------------------------------------------------------
# # ==============================================================================

# log_host = "127.0.0.1"  # Replace with your server's IP address
# log_port = 8888  # Replace with your server's port

# ==============================================================================
# async def send_log_data(host, port, log_data):
#     try:
#         reader, writer = await asyncio.open_connection(host, port)
#         data = json.dumps(log_data.to_dict(orient="records")).encode()
#         writer.write(data)
#         await writer.drain()
#         writer.close()
#         await writer.wait_closed()

#     except ConnectionRefusedError:
#         pass

#     except Exception as e:
#         template = "An exception of type {0} occurred. Arguments:\n{1!r}"
#         message = template.format(type(e).__name__, e.args)
#         print(message)

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
    argparser.add_argument(
        "-sc",
        "--static_camera",
        help="Set flag for using static camera",
        default=0,
        type=int,
    )

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split("x")]
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=log_level)

    logging.info("listening to server %s:%s", args.host, args.port)

    try:
        simulation = Simulation(args=args, frame_limit=1000, episode_limit=100)
        simulation.run()

    except KeyboardInterrupt as e:
        print("\n")
        if simulation.world is not None:
            settings = simulation.world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            simulation.world.world.apply_settings(settings)
            simulation.world.destroy()
            simulation.world = None
        pygame.quit()
        return -1

    except Exception as e:
        print(traceback.format_exc())

    finally:
        if simulation.world is not None:
            settings = simulation.world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            simulation.world.world.apply_settings(settings)
            simulation.world.destroy()

        print("Bye, bye")
        pygame.quit()
        return -1


if __name__ == "__main__":
    main()
