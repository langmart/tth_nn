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

class ttHComparePlotter:
    """Plotter for multiple plots in one Canvas."""

    def __init__(self):
        """Initializes the Plotter.
        """
    def _get_arrays(self):
        """Loads the arrays into a dict."""

        purity_dict = dict()
        significance_dict = dict()
        product_dict = dict()
        self.produce_paths = []
        for path in self.paths:
            produce_path = path + '/' + self.subpath
            try:
                with open(produce_path + '/val_purity.txt', 'rb') as f:
                    purity = pickle.load(f)
            except FileNotFoundError:
                sys.exit('ERROR: The requested file {} does not exist.'.format(produce_path + '/purity.txt'))
            try:
                with open(produce_path + '/val_significance.txt', 'rb') as f:
                    significance = pickle.load(f)
            except FileNotFoundError:
                sys.exit('ERROR: The requested file {} does not exist.'.format(produce_path + '/significance.txt'))
            

            self.produce_paths.append(produce_path)
            prod = [purity[i] * significance[i] for i in range(len(purity))]
            product_dict[produce_path] = prod
            purity_dict[produce_path] = purity
            significance_dict[produce_path] = significance

        return purity_dict, significance_dict, product_dict

    def plot(self, paths, subpath, title, labels, epoch_range, out_path,
            ylabel=r'Purity $\cdot$ significance'):
        self.paths = paths
        self.subpath = subpath
        self.title = r'{}'.format(title)
        self.labels = labels
        self.out_path = out_path
        self.epoch_range = epoch_range
        self.ylabel = r'{}'.format(ylabel)
        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path)

        purity, significance, product = self._get_arrays()
        

        colormap = plt.cm.jet
        plot_colors = [colormap(i) for i in np.linspace(0, 0.9, len(self.paths))]
        for i in range(len(product)):
            path = self.produce_paths[i]
            plt.plot(product[path], label=r'{}'.format(labels[i]), 
                    color=plot_colors[i])
        plt.xlabel(r'Epoch')
        plt.ylabel(self.ylabel)
        plt.title(self.title)
        plt.legend(loc='best')
        plt.savefig(self.out_path + '/out.pdf')
        plt.clf()
        self._write_out()


    def _write_out(self):
        with open(self.out_path + '/info.txt', 'w') as f:
            f.write('paths: \n')
            for path in self.paths:
                f.write(path + '\n')
            f.write('subpath: {}\n'.format(self.subpath))
            f.write('title: {}\n'.format(self.title))


