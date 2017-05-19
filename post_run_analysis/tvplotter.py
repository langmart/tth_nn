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

titlefontsize = 20
labelfontsize = 18
tickfontsize = 17
class TVPlotter:
    """Plotter training and validation accuracy in one canvas."""
    
    def __init__(self):
        """Initializes the Plotter.
        """
    def _get_arrays(self):
        """Loads the arrays into a dict."""
        
        data_dict = dict()
        self.produce_paths = []
        for subpath in self.subpaths:
            produce_path = self.path + '/' + subpath
            try:
                with open(produce_path, 'rb') as f:
                    data_dict[produce_path] = pickle.load(f)
            except FileNotFoundError:
                sys.exit('ERROR: The requested file {} does not exist.'.format(produce_path))
            self.produce_paths.append(produce_path)
        return data_dict

    def plot(self, path, subpaths, title, labels, epoch_range, out_path,
            ylabel='Validation accuracy'):
        """Plots the data in the range determined by epoch_range.

        Arguments:
        ----------------
        paths (list of str):
            String containing the path to all features to be included.
        subpaths (str):
            List containing the subpaths.
        title (str):
            Contains the plot title.
        labels (list of str):
            Contains the plot label strings.
        """
        self.path = path
        self.subpaths = subpaths
        self.title = r'{}'.format(title)
        self.labels = labels
        self.out_path = out_path
        self.epoch_range = epoch_range
        self.ylabel = r'{}'.format(ylabel)
        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path)

        data = self._get_arrays()
        colormap = plt.cm.jet
        plot_colors = [colormap(i) for i in np.linspace(0, 0.9, len(self.subpaths))]
        for i in range(len(data)):
            path = self.produce_paths[i]
            plt.plot(data[path], label=r'{}'.format(labels[i]), 
                    color=plot_colors[i])
        ax = plt.gca()
        plt.setp(ax.get_yticklabels(), fontsize=tickfontsize)
        plt.setp(ax.get_xticklabels(), fontsize=tickfontsize)
        plt.xlabel(r'Epoch', fontsize=labelfontsize)
        plt.ylabel(self.ylabel, fontsize=labelfontsize)
        plt.title(self.title, fontsize=titlefontsize)
        plt.legend(loc='best')
        plt.savefig(self.out_path + '/out.pdf')
        plt.clf()
        self._write_out()


    def _write_out(self):
        with open(self.out_path + '/info.txt', 'w') as f:
            f.write('paths: \n')
            f.write('path: {}\n'.format(self.path))
            for subpath in self.subpaths:
                f.write(subpath + '\n')
            f.write('title: {}\n'.format(self.title))
