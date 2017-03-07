import numpy as np
import datetime
from post_run_analysis.compare_plotter import ComparePlotter

plotter = ComparePlotter()

a_path = 'data/executed/analyses/'
paths_1 = ['adadelta', 'adagrad', 'adam', 'graddescent', 'momentum']
paths = [a_path + i for i in paths_1]
subpath = 'cross_checks/val_accuracy.txt'
title = 'Validation accuracy for different optimizers'
labels = ['Adadelta', 'Adagrad', 'Adam', 'Gradient Descent', 'Momentum']
ylabel = 'Validation accuracy'
out_path = 'data/studies/optimizers'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

# a_path = 'data/executed/analyses/'
# paths_1 = ['adam_tanh', 'adam_relu', 'adam_elu', 'adam_softplus',
#         'adam_sigmoid']
# paths = [a_path + i for i in paths_1]
# 
# subpath = 'cross_checks/val_accuracy.txt'
# title = 'Validation accuracy for different activation functions'
# labels = ['tanh', 'relu', 'elu', 'softplus', 'sigmoid']
# ylabel= 'Validation accuracy'
# out_path = 'data/studies/activation'
# 
# plotter = ComparePlotter()
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

# paths_1 = ['adam_tanh_dropout_13',
#         'adam_tanh_dropout_12',
#         'adam_tanh_dropout_11',
#         'adam_tanh_dropout_10',
#         'adam_tanh_dropout_9',
#         'adam_tanh_dropout_8', 
#         'adam_tanh_dropout_7',
#         'adam_tanh_dropout_1',
#         'adam_tanh_dropout_2',
#         'adam_tanh_dropout_3',
#         'adam_tanh_dropout_4',
#         'adam_tanh_dropout_5',
#         'adam_tanh_dropout_6']
# paths = [a_path + i for i in paths_1]
# subpath = 'cross_checks/val_accuracy.txt'
# title = 'Validation accuracy for different dropout values'
# labels = ['d=1.0', 'd=0.995', 'd=0.99', 'd=0.985', 'd=0.98', 'd=0.95', 'd=0.92',
#         'd=0.9', 'd=0.8', 'd=0.7', 'd=0.6', 'd=0.5', 'd=0.4']
# 
# out_path = 'data/studies/dropout'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path)

# paths_1 = ['adam_1x20', 'adam_2x20', 'adam_3x20', 'adam_4x20', 'adam_5x20']
# paths = [a_path + i for i in paths_1]
# subpath = 'cross_checks/val_accuracy.txt'
# title = 'Validation accuracy for different network architectures'
# labels = ['1x20', '2x20', '3x20', '4x20', '5x20']
# out_path = 'data/studies/archi_20'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path)
# paths_1 = ['adam_1x50', 'adam_2x50', 'adam_3x50', 'adam_4x50', 'adam_5x50']
# paths = [a_path + i for i in paths_1]
# subpath = 'cross_checks/val_accuracy.txt'
# title = 'Validation accuracy for different network architectures'
# labels = ['1x50', '2x50', '3x50', '4x50', '5x50']
# out_path = 'data/studies/archi_50'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path)
# paths_1 = ['adam_1x100', 'adam_2x100', 'adam_3x100', 'adam_4x100', 'adam_5x100']
# paths = [a_path + i for i in paths_1]
# subpath = 'cross_checks/val_accuracy.txt'
# title = 'Validation accuracy for different network architectures'
# labels = ['1x100', '2x100', '3x100', '4x100', '5x100']
# out_path = 'data/studies/archi_100'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path)
# paths_1 = ['adam_1x200', 'adam_2x200', 'adam_3x200', 'adam_4x200', 'adam_5x200']
# paths = [a_path + i for i in paths_1]
# subpath = 'cross_checks/val_accuracy.txt'
# title = 'Validation accuracy for different network architectures'
# labels = ['1x200', '2x200', '3x200', '4x200', '5x200']
# out_path = 'data/studies/archi_200'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path)
# paths_1 = ['adam_5x20', 'adam_5x50', 'adam_5x100', 'adam_5x200']
# paths = [a_path + i for i in paths_1]
# subpath = 'cross_checks/val_accuracy.txt'
# title = 'Validation accuracy for different network architectures'
# labels = ['5x20', '5x50', '5x100', '5x200']
# out_path = 'data/studies/archi_5x'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path)


# paths_1 = ['adam_architecture_21', 'adam_architecture_22',
#         'adam_architecture_23', 'adam_architecture_24', 'adam_architecture_25']
# paths = [a_path + i for i in paths_1]
# subpath = 'cross_checks/val_accuracy.txt'
# title = 'Validation accuracy for different network architectures'
# labels = ['400+200+200+50', '200+150+100+50', '100+200+200+300',
#         '300+170+90+30', '300+100+300+100']
# out_path = 'data/studies/archi_inhom'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path)
