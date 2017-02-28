# A plotting tool to compare different features such as different activation
# functions.

from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import os
import time
import pickle

class ComparePlotter:
    """Plotter for multiple plots in one Canvas."""
    
    def __init__(self):
        """Initializes the Plotter.
        """
    def _get_arrays(self):
        """Loads the arrays into a dict."""
        
        data_dict = dict()
        self.produce_paths = []
        for path in self.paths:
            produce_path = path + '/' + self.subpath
            try:
                with open(produce_path, 'rb') as f:
                    data_dict[produce_path] = pickle.load(f)
            except FileNotFoundError:
                sys.exit('ERROR: The requested file {} does not exist.'.format(produce_path))
            self.produce_paths.append(produce_path)
        return data_dict

    def plot(self, paths, subpath, title, labels, epoch_range, out_path,
            ylabel='Validation accuracy'):
        """Plots the data in the range determined by epoch_range.

        Arguments:
        ----------------
        paths (list of str):
            List containing the paths to all features to be included.
        subpath (str):
            Contains the subdirectory from where to fetch the features. 
        title (str):
            Contains the plot title.
        labels (list of str):
            Contains the plot label strings.
        """
        self.paths = paths
        self.subpath = subpath
        self.title = title
        self.labels = labels
        self.out_path = out_path
        self.epoch_range = epoch_range
        self.ylabel = ylabel
        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path)

        data = self._get_arrays()
        for i in range(len(data)):
            path = self.produce_paths[i]
            plt.plot(data[path], label=labels[i])
        plt.xlabel('Epoch')
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.legend(loc='best')
        plt.savefig(self.out_path + '/out.pdf')
        plt.clf()
            # print(i)
            # print(data[i])
        self._write_out()


    def _write_out(self):
        with open(self.out_path + '/info.txt', 'w') as f:
            f.write('paths: \n')
            for path in self.paths:
                f.write(path + '\n')
            f.write('subpath: {}\n'.format(self.subpath))
            f.write('title: {}\n'.format(self.title))
