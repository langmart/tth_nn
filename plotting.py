import numpy as np
import datetime
from post_run_analysis.compare_plotter import ComparePlotter

a_path = 'data/executed/analyses/'
paths_1 = ['adam_tanh', 'adam_relu', 'adam_elu', 'adam_softplus',
        'adam_sigmoid']
paths = [a_path + i for i in paths_1]

subpath = 'cross_checks/val_accuracy.txt'
title = 'Validation accuracy for different activation functions'
labels = ['tanh', 'relu', 'elu', 'softplus', 'sigmoid']
ylabel= 'Validation accuracy'
out_path = 'data/studies/activation'

plotter = ComparePlotter()
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['adam_tanh_dropout_13',
        'adam_tanh_dropout_12',
        'adam_tanh_dropout_11',
        'adam_tanh_dropout_10',
        'adam_tanh_dropout_9',
        'adam_tanh_dropout_8', 
        'adam_tanh_dropout_7',
        'adam_tanh_dropout_1',
        'adam_tanh_dropout_2',
        'adam_tanh_dropout_3',
        'adam_tanh_dropout_4',
        'adam_tanh_dropout_5',
        'adam_tanh_dropout_6']
paths = [a_path + i for i in paths_1]
subpath = 'cross_checks/val_accuracy.txt'
title = 'Validation accuracy for different dropout values'
labels = ['d=1.0', 'd=0.995', 'd=0.99', 'd=0.985', 'd=0.98', 'd=0.95', 'd=0.92',
        'd=0.9', 'd=0.8', 'd=0.7', 'd=0.6', 'd=0.5', 'd=0.4']

out_path = 'data/studies/dropout'
plotter.plot(paths, subpath, title, labels, [30,40], out_path)

