import numpy as np
import datetime
from post_run_analysis.compare_plotter import ComparePlotter

paths = ['data/executed/analyses/adam_tanh',
        'data/executed/analyses/adam_relu',
        'data/executed/analyses/adam_elu',
        'data/executed/analyses/adam_softplus',
        'data/executed/analyses/adam_sigmoid']
subpath = 'cross_checks/val_accuracy.txt'
title = 'Validation accuracy for different activation functions'
labels = ['tanh', 'relu', 'elu', 'softplus', 'sigmoid']
ylabel= 'Validation accuracy'
out_path = 'data/studies/activation'

plotter = ComparePlotter()
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths = ['data/executed/analyses/adam_tanh_dropout_1',
        'data/executed/analyses/adam_tanh_dropout_2',
        'data/executed/analyses/adam_tanh_dropout_3',
        'data/executed/analyses/adam_tanh_dropout_4',
        'data/executed/analyses/adam_tanh_dropout_5',
        'data/executed/analyses/adam_tanh_dropout_6']
subpath = 'cross_checks/val_accuracy.txt'
title = 'Validation accuracy for different dropout values'
labels = ['d=0.9', 'd=0.8', 'd=0.7', 'd=0.6', 'd=0.5', 'd=0.4']

out_path = 'data/studies/dropout'
plotter.plot(paths, subpath, title, labels, [30,40], out_path)
