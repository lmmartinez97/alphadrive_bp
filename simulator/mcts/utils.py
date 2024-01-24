"""Utility functions for the AlphaZero algorithm."""


from __future__ import google_type_annotations
from __future__ import division

import os
import datetime
from network import Network


# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
    return 0, 0


def launch_job(f, *args):
    f(*args)


def make_uniform_network():
    return Network()


def create_directory(base_path):
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    dir_name = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the full path to the new directory
    full_path = os.path.join(base_path, dir_name)

    # Create the new directory
    os.makedirs(full_path, exist_ok=True)

    return full_path
