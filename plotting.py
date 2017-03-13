import numpy as np
import datetime
from post_run_analysis.compare_plotter import ComparePlotter
from post_run_analysis.train_val_plotter import TrainValPlotter

plotter = ComparePlotter()

# Different optimizers
a_path = 'data/executed/analyses/'
paths_1 = ['adadelta', 'adagrad', 'adam', 'graddescent', 'momentum']
paths = [a_path + i for i in paths_1]
subpath = 'cross_checks/val_accuracy.txt'
title = 'Validation accuracy for different optimizers'
labels = ['Adadelta', 'Adagrad', 'Adam', 'Gradient Descent', 'Momentum']
ylabel = 'Validation accuracy'
out_path = 'data/studies/optimizers'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

a_path = 'data/executed/analyses/'

# l2 regularization and dropout
paths_1 = ['adadelta', 'adadelta_l2', 'adadelta_dropout', 'adadelta_l2_dropout']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different settings'
labels = ['Adadelta', 'Adadelta + l2', 'Adadelta + dropout', 'Adadelta + l2 + dropout']
out_path = 'data/studies/no_overfit/adadelta'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['adagrad', 'adagrad_l2', 'adagrad_dropout', 'adagrad_l2_dropout']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different settings'
labels = ['Adagrad', 'Adagrad + l2', 'Adagrad + dropout', 'Adagrad + l2 + dropout']
out_path = 'data/studies/no_overfit/adagrad'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['adam', 'adam_l2', 'adam_dropout', 'adam_l2_dropout']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different settings'
labels = ['Adam', 'Adam + l2', 'Adam + dropout', 'Adam + l2 + dropout']
out_path = 'data/studies/no_overfit/adam'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['graddescent', 'graddescent_l2', 'graddescent_dropout', 'graddescent_l2_dropout']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different settings'
labels = ['Gradient Descent', 'Gradient Descent + l2', 'Gradient Descent + dropout', 'Gradient Descent + l2 + dropout']
out_path = 'data/studies/no_overfit/graddescent'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['momentum', 'momentum_l2', 'momentum_dropout', 'momentum_l2_dropout']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different settings'
labels = ['Momentum', 'Momentum + l2', 'Momentum + dropout', 'Momentum + l2 + dropout']
out_path = 'data/studies/no_overfit/momentum'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['adadelta', 'adadelta_l2', 'adadelta_dropout', 'adadelta_l2_dropout',
        'adagrad', 'adagrad_l2', 'adagrad_dropout', 'adagrad_l2_dropout',
        'adam', 'adam_l2', 'adam_dropout', 'adam_l2_dropout', 'graddescent',
        'graddescent_l2', 'graddescent_dropout', 'graddescent_l2_dropout',
        'momentum', 'momentum_l2', 'momentum_dropout', 'momentum_l2_dropout']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different settings'
labels = ['adadelta', 'adadelta_l2', 'adadelta_dropout', 'adadelta_l2_dropout',
        'adagrad', 'adagrad_l2', 'adagrad_dropout', 'adagrad_l2_dropout',
        'adam', 'adam_l2', 'adam_dropout', 'adam_l2_dropout', 'graddescent',
        'graddescent_l2', 'graddescent_dropout', 'graddescent_l2_dropout',
        'momentum', 'momentum_l2', 'momentum_dropout', 'momentum_l2_dropout']
out_path = 'data/studies/no_overfit/all'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['adadelta_l2_dropout', 'adagrad_l2_dropout', 'adam_l2_dropout',
        'graddescent_l2_dropout', 'momentum_l2_dropout']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different optimizers'
labels = ['Adadelta + l2 + dropout' , 'Adagrad + l2 + dropout', 'Adam + l2 + dropout', 
        'Gradient Descent + l2 + dropout', 'Momentum + l2 + dropout']
out_path = 'data/studies/no_overfit/l2_dropout'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

labels = paths_1 = ['adam_l2_dropout', 'adam_l2_dropout_2', 'adam_l2_dropout_3', 
        'adam_l2_dropout_4', 'adam_l2_dropout_5', 'adam_l2_dropout_6',
        'adam_l2_dropout_7']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy'
out_path = 'data/studies/no_overfit/adam_studies'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)


# Optimizer + activation function
a_path = 'data/executed/analyses/opt_act/'
paths_1 = ['adadelta_elu', 'adadelta_relu', 'adadelta_sigmoid',
        'adadelta_softplus', 'adadelta_tanh']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different activation functions'
labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
out_path = 'data/studies/opt_act/adadelta'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
paths_1 = ['adagrad_elu', 'adagrad_relu', 'adagrad_sigmoid',
        'adagrad_softplus', 'adagrad_tanh']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different activation functions'
labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
out_path = 'data/studies/opt_act/adagrad'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
paths_1 = ['adam_elu', 'adam_relu', 'adam_sigmoid',
        'adam_softplus', 'adam_tanh']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different activation functions'
labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
out_path = 'data/studies/opt_act/adam'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
paths_1 = ['graddescent_elu', 'graddescent_relu', 'graddescent_sigmoid',
        'graddescent_softplus', 'graddescent_tanh']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different activation functions'
labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
out_path = 'data/studies/opt_act/graddescent'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
paths_1 = ['momentum_elu', 'momentum_relu', 'momentum_sigmoid',
        'momentum_softplus', 'momentum_tanh']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy for different activation functions'
labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
out_path = 'data/studies/opt_act/momentum'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['adadelta_relu', 'adagrad_elu', 'adam_elu',
        'graddescent_elu', 'momentum_elu']
# paths_1 = ['adadelta_elu', 'adadelta_relu', 'adagrad_elu', 'adam_elu',
#         'graddescent_elu', 'graddescent_relu', 'momentum_elu']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy'
labels = ['Adadelta + ReLU', 'Adagrad + ELU', 'Adam + ELU',
        'Gradient Descent + ELU', 'Momentum + ELU']
out_path = 'data/studies/opt_act/best'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

a_path = 'data/executed/analyses/lrate/'
paths_1 = ['lrate_2', 'lrate_3', 'lrate_4', 'lrate_5', 'lrate_6',
        'lrate_7', 'lrate_8', 'lrate_9', 'lrate_10', 'lrate_11']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy'
labels = ['$\eta = 0.05$', '$\eta = 0.02$', '$\eta = 0.01$', 
        '$\eta = 0.005$', '$\eta = 0.002$', '$\eta = 0.001$', 
        '$\eta = 5\cdot 10^{-4}$', '$\eta = 2\cdot 10^{-4}$', 
        '$\eta = 10^{-4}$', '$\eta = 5\cdot 10^{-5}$']
out_path = 'data/studies/lrate'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

paths_1 = ['lrate_3', 'lrate_4', 'lrate_5', 'lrate_6',
        'lrate_7', 'lrate_8', 'lrate_9', 'lrate_10', 'lrate_11']
paths = [a_path + i for i in paths_1]
title = 'Validation accuracy'
labels = ['$\eta = 0.02$', '$\eta = 0.01$', 
        '$\eta = 0.005$', '$\eta = 0.002$', '$\eta = 0.001$', 
        '$\eta = 5\cdot 10^{-4}$', '$\eta = 2\cdot 10^{-4}$', 
        '$\eta = 10^{-4}$', '$\eta = 5\cdot 10^{-5}$']
out_path = 'data/studies/lrate_corrected'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

plotter = TrainValPlotter()
a_path = 'data/executed/analyses/graddescent/'
subpaths = ['cross_checks/train_accuracy.txt', 'cross_checks/val_accuracy.txt']
title = 'Accuracy development'
labels = ['training set', 'validation set']
out_path = 'data/studies/overfit/'
plotter.plot(a_path, subpaths, title, labels, [30,40], out_path, ylabel)
