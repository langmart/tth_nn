from __future__ import absolute_import, division, print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import sys
import os
import time
import pickle

class Plotter:
    def __init__(self, savedir):
        """Initializes the Plotter.

        Arguments:
        ----------------
        savedir (string):
            Path to the directory the plots will be saved to.
        """

        self.savedir = savedir

    def plot_5_top_weights():

