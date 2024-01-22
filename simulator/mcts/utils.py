"""Utility functions for the AlphaZero algorithm."""


from __future__ import google_type_annotations
from __future__ import division

from network import Network

# Stubs to make the typechecker happy, should not be included in pseudocode
# for the paper.
def softmax_sample(d):
  return 0, 0


def launch_job(f, *args):
  f(*args)


def make_uniform_network():
  return Network()