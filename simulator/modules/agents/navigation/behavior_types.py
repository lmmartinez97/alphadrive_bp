# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains the different parameters sets for each behavior. """

from dataclasses import dataclass

@dataclass
class Cautious:
    """Class for Cautious agent."""
    max_speed: int = 40
    speed_lim_dist: int = 6
    speed_decrease: int = 12
    safety_time: int = 3
    min_proximity_threshold: int = 12
    braking_distance: int = 6
    tailgate_counter: int = 0

@dataclass
class Normal:
    """Class for Normal agent."""
    max_speed: int = 50
    speed_lim_dist: int = 3
    speed_decrease: int = 10
    safety_time: int = 3
    min_proximity_threshold: int = 10
    braking_distance: int = 5
    tailgate_counter: int = 0

@dataclass
class Aggressive:
    """Class for Aggressive agent."""
    max_speed: int = 70
    speed_lim_dist: int = 1
    speed_decrease: int = 8
    safety_time: int = 3
    min_proximity_threshold: int = 8
    braking_distance: int = 4
    tailgate_counter: int = -1
