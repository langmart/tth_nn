import numpy as np
import datetime
from post_run_analysis.compare_plotter import ComparePlotter
from post_run_analysis.tvplotter import TVPlotter
from post_run_analysis.ttH_compare_plotter import ttHComparePlotter

plotter = ComparePlotter()
tv_plotter = TVPlotter()
subpath = 'cross_checks/val_accuracy.txt'
ylabel = 'Validation accuracy'

# # Different optimizers
# a_path = 'data/executed/analyses/'
# paths_1 = ['adadelta', 'adagrad', 'adam', 'graddescent', 'momentum']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different optimizers'
# labels = ['Adadelta', 'Adagrad', 'Adam', 'Gradient Descent', 'Momentum']
# out_path = 'data/studies/optimizers'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# a_path = 'data/executed/analyses/'
# 
# # l2 regularization and dropout
# paths_1 = ['adadelta', 'adadelta_l2', 'adadelta_dropout', 'adadelta_l2_dropout']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different settings'
# labels = ['Adadelta', 'Adadelta + l2', 'Adadelta + dropout', 'Adadelta + l2 + dropout']
# out_path = 'data/studies/no_overfit/adadelta'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['adagrad', 'adagrad_l2', 'adagrad_dropout', 'adagrad_l2_dropout']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different settings'
# labels = ['Adagrad', 'Adagrad + l2', 'Adagrad + dropout', 'Adagrad + l2 + dropout']
# out_path = 'data/studies/no_overfit/adagrad'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['adam', 'adam_l2', 'adam_dropout', 'adam_l2_dropout']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different settings'
# labels = ['Adam', 'Adam + l2', 'Adam + dropout', 'Adam + l2 + dropout']
# out_path = 'data/studies/no_overfit/adam'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['graddescent', 'graddescent_l2', 'graddescent_dropout', 'graddescent_l2_dropout']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different settings'
# labels = ['Gradient Descent', 'Gradient Descent + l2', 'Gradient Descent + dropout', 'Gradient Descent + l2 + dropout']
# out_path = 'data/studies/no_overfit/graddescent'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['momentum', 'momentum_l2', 'momentum_dropout', 'momentum_l2_dropout']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different settings'
# labels = ['Momentum', 'Momentum + l2', 'Momentum + dropout', 'Momentum + l2 + dropout']
# out_path = 'data/studies/no_overfit/momentum'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['adadelta', 'adadelta_l2', 'adadelta_dropout', 'adadelta_l2_dropout',
#         'adagrad', 'adagrad_l2', 'adagrad_dropout', 'adagrad_l2_dropout',
#         'adam', 'adam_l2', 'adam_dropout', 'adam_l2_dropout', 'graddescent',
#         'graddescent_l2', 'graddescent_dropout', 'graddescent_l2_dropout',
#         'momentum', 'momentum_l2', 'momentum_dropout', 'momentum_l2_dropout']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different settings'
# labels = ['adadelta', 'adadelta_l2', 'adadelta_dropout', 'adadelta_l2_dropout',
#         'adagrad', 'adagrad_l2', 'adagrad_dropout', 'adagrad_l2_dropout',
#         'adam', 'adam_l2', 'adam_dropout', 'adam_l2_dropout', 'graddescent',
#         'graddescent_l2', 'graddescent_dropout', 'graddescent_l2_dropout',
#         'momentum', 'momentum_l2', 'momentum_dropout', 'momentum_l2_dropout']
# out_path = 'data/studies/no_overfit/all'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['adadelta_l2_dropout', 'adagrad_l2_dropout', 'adam_l2_dropout',
#         'graddescent_l2_dropout', 'momentum_l2_dropout']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different optimizers'
# labels = ['Adadelta + l2 + dropout' , 'Adagrad + l2 + dropout', 'Adam + l2 + dropout', 
#         'Gradient Descent + l2 + dropout', 'Momentum + l2 + dropout']
# out_path = 'data/studies/no_overfit/l2_dropout'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# labels = paths_1 = ['adam_l2_dropout', 'adam_l2_dropout_2', 'adam_l2_dropout_3', 
#         'adam_l2_dropout_4', 'adam_l2_dropout_5', 'adam_l2_dropout_6',
#         'adam_l2_dropout_7']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy'
# out_path = 'data/studies/no_overfit/adam_studies'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# 
# # Optimizer + activation function
# a_path = 'data/executed/analyses/opt_act/'
# paths_1 = ['adadelta_elu', 'adadelta_relu', 'adadelta_sigmoid',
#         'adadelta_softplus', 'adadelta_tanh']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different activation functions'
# labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
# out_path = 'data/studies/opt_act/adadelta'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# paths_1 = ['adagrad_elu', 'adagrad_relu', 'adagrad_sigmoid',
#         'adagrad_softplus', 'adagrad_tanh']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different activation functions'
# labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
# out_path = 'data/studies/opt_act/adagrad'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# paths_1 = ['adam_elu', 'adam_relu', 'adam_sigmoid',
#         'adam_softplus', 'adam_tanh']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different activation functions'
# labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
# out_path = 'data/studies/opt_act/adam'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# paths_1 = ['graddescent_elu', 'graddescent_relu', 'graddescent_sigmoid',
#         'graddescent_softplus', 'graddescent_tanh']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different activation functions'
# labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
# out_path = 'data/studies/opt_act/graddescent'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# paths_1 = ['momentum_elu', 'momentum_relu', 'momentum_sigmoid',
#         'momentum_softplus', 'momentum_tanh']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy for different activation functions'
# labels = ['ELU', 'ReLU', 'Sigmoid', 'Softplus', 'tanh']
# out_path = 'data/studies/opt_act/momentum'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['adadelta_relu', 'adagrad_elu', 'adam_elu',
#         'graddescent_elu', 'momentum_elu']
# # paths_1 = ['adadelta_elu', 'adadelta_relu', 'adagrad_elu', 'adam_elu',
# #         'graddescent_elu', 'graddescent_relu', 'momentum_elu']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy'
# labels = ['Adadelta + ReLU', 'Adagrad + ELU', 'Adam + ELU',
#         'Gradient Descent + ELU', 'Momentum + ELU']
# out_path = 'data/studies/opt_act/best'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# a_path = 'data/executed/analyses/lrate/'
# paths_1 = ['lrate_2', 'lrate_3', 'lrate_4', 'lrate_5', 'lrate_6',
#         'lrate_7', 'lrate_8', 'lrate_9', 'lrate_10', 'lrate_11']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy'
# labels = ['$\eta = 0.05$', '$\eta = 0.02$', '$\eta = 0.01$', 
#         '$\eta = 0.005$', '$\eta = 0.002$', '$\eta = 0.001$', 
#         '$\eta = 5\cdot 10^{-4}$', '$\eta = 2\cdot 10^{-4}$', 
#         '$\eta = 10^{-4}$', '$\eta = 5\cdot 10^{-5}$']
# out_path = 'data/studies/lrate'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['lrate_3', 'lrate_4', 'lrate_5', 'lrate_6',
#         'lrate_7', 'lrate_8', 'lrate_9', 'lrate_10', 'lrate_11']
# paths = [a_path + i for i in paths_1]
# title = 'Validation accuracy'
# labels = ['$\eta = 0.02$', '$\eta = 0.01$', 
#         '$\eta = 0.005$', '$\eta = 0.002$', '$\eta = 0.001$', 
#         '$\eta = 5\cdot 10^{-4}$', '$\eta = 2\cdot 10^{-4}$', 
#         '$\eta = 10^{-4}$', '$\eta = 5\cdot 10^{-5}$']
# out_path = 'data/studies/lrate_corrected'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# a_path = 'data/executed/analyses/graddescent/'
# subpaths = ['cross_checks/train_accuracy.txt', 'cross_checks/val_accuracy.txt']
# title = 'Accuracy development'
# labels = ['training set', 'validation set']
# out_path = 'data/studies/overfit/'
# tv_plotter.plot(a_path, subpaths, title, labels, [30,40], out_path, ylabel)
# 
# a_path = 'data/executed/analyses/momentum_l2_dropout'
# subpaths = ['cross_checks/train_accuracy.txt', 'cross_checks/val_accuracy.txt']
# title = 'Accuracy development'
# labels = ['training set', 'validation set']
# out_path = 'data/studies/no_overfit/train_val/'
# tv_plotter.plot(a_path, subpaths, title, labels, [30,40], out_path, ylabel)
# 
# a_path = 'data/executed/analyses/momentum_l2'
# subpaths = ['cross_checks/train_accuracy.txt', 'cross_checks/val_accuracy.txt']
# title = 'Accuracy development'
# labels = ['training set', 'validation set']
# out_path = 'data/studies/no_overfit/train_val_l2/'
# tv_plotter.plot(a_path, subpaths, title, labels, [30,40], out_path, ylabel)
# 
# a_path = 'data/executed/analyses/momentum_dropout'
# subpaths = ['cross_checks/train_accuracy.txt', 'cross_checks/val_accuracy.txt']
# title = 'Accuracy development'
# labels = ['training set', 'validation set']
# out_path = 'data/studies/no_overfit/train_val_dropout/'
# tv_plotter.plot(a_path, subpaths, title, labels, [30,40], out_path, ylabel)

# a_path = 'data/executed/training/architecture/'
# paths_1 = ['1x20', '1x50', '1x100', '1x200', '2x20', '2x50', '2x100', '2x200']
# paths = [a_path + 'train_' + i for i in paths_1]
# title = 'Validation accuracy'
# labels = ['1x20', '1x50', '1x100', '1x200', '2x20', '2x50', '2x100', '2x200']
# out_path = 'data/studies/architecture/1_and_2_layers/'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# a_path = 'data/executed/training/architecture/'
# paths_1 = ['3x20', '3x50', '3x100', '3x200', '4x20', '4x50', '4x100', '4x200']
# paths = [a_path + 'train_' + i for i in paths_1]
# title = 'Validation accuracy'
# labels = ['3x20', '3x50', '3x100', '3x200', '4x20', '4x50', '4x100', '4x200']
# out_path = 'data/studies/architecture/3_and_4_layers/'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# a_path = 'data/executed/training/architecture/'
# paths_1 = ['5x20', '5x50', '5x100', '5x200', '6x20', '6x50', '6x100', '6x200']
# paths = [a_path + 'train_' + i for i in paths_1]
# title = 'Validation accuracy'
# labels = ['5x20', '5x50', '5x100', '5x200', '6x20', '6x50', '6x100', '6x200']
# out_path = 'data/studies/architecture/5_and_6_layers/'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)
# 
# paths_1 = ['300+100+200+100', '300+150+100+50', '30+70+50+20',
#         '500+400+300+200+100', '200+50+50+200', '50+200+200+50']
# paths = [a_path + 'train_' + i for i in paths_1]
# title = 'Validation accuracy'
# labels = ['300+100+200+100', '300+150+100+50', '30+70+50+20',
#         '500+400+300+200+100', '200+50+50+200', '50+200+200+50']
# out_path = 'data/studies/architecture/inhomogeneous/'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

a_path = 'data/executed/training/architecture/'
paths_1 = ['1x200', '2x200', '3x200', '4x200', '5x200', '6x200']
labels = paths_1
title = 'Validation accuracy'
paths = [a_path + 'train_' + i for i in paths_1]
out_path = 'data/studies/architecture/x200/'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel)

a_path = 'data/executed/analyses/batch_size/'
paths_1 = ['bs_1', 'bs_2', 'bs_3', 'bs_4', 'bs_5', 'bs_6', 'bs_7', 'bs_8']
labels = ['100', '150', '200', '250', '300', '400', '500', '600']
title = 'Validation accuracy'
legend_title = 'Batch size'
paths = [a_path + i for i in paths_1]
out_path = 'data/studies/batch_size/1to8/'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel,
        legend_title)
paths_1 = ['bs_9', 'bs_10', 'bs_11', 'bs_12', 'bs_13', 'bs_14']
labels = ['800', '1000', '1200', '1500', '2000', '2500']
title = 'Validation accuracy'
legend_title = 'Batch size'
paths = [a_path + i for i in paths_1]
out_path = 'data/studies/batch_size/9to14/'
plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel,
        legend_title)
# paths_1 = ['bs_9', 'bs_10', 'bs_11', 'bs_12', 'bs_13', 'bs_14', 'bs_15', 'bs_16']
# labels = ['800', '1000', '1200', '1500', '2000', '2500', '3000', '4000']
# title = 'Validation accuracy'
# legend_title = 'Batch size'
# paths = [a_path + i for i in paths_1]
# out_path = 'data/studies/batch_size/9to16/'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel,
#         legend_title)
# paths_1 = ['bs_17', 'bs_18', 'bs_19', 'bs_20', 'bs_21', 'bs_22', 'bs_23']
# labels = ['5000', '7000', '10000', '15000', '20000', '25000', '30000']
# title = 'Validation accuracy'
# legend_title = 'Batch size'
# paths = [a_path + i for i in paths_1]
# out_path = 'data/studies/batch_size/17to23/'
# plotter.plot(paths, subpath, title, labels, [30,40], out_path, ylabel,
#         legend_title)



ttH_plotter = ttHComparePlotter()
ttH_path = 'data/executed/analyses_ttH/'
ttH_subpath = 'cross_checks/'
paths_1 = ['ttH_adadelta', 'ttH_adagrad', 'ttH_adam', 'ttH_graddescent',
        'ttH_momentum']
paths = [ttH_path + i for i in paths_1]
title = 'ttH purity $\cdot$ significance'
labels = ['Adadelta', 'Adagrad', 'Adam', 'Gradient descent', 'Momentum']
out_path = 'data/studies_ttH/optimizers/'
ttH_plotter.plot(paths, ttH_subpath, title, labels, [30,40], out_path)

paths_1 = ['ttH_1', 'ttH_2', 'ttH_3', 'ttH_4', 'ttH_5', 'ttH_6', 'ttH_7',
        'ttH_8', 'ttH_9', 'ttH_10', 'ttH_11']
paths = [ttH_path + i for i in paths_1]
title = 'ttH purity $\cdot$ significance'
labels = paths_1
out_path = 'data/studies_ttH/penalty'
ttH_plotter.plot(paths, ttH_subpath, title, labels, [30,40], out_path)

a_path = ttH_path + 'penalty/'
paths_1 = ['penalty_1', 'penalty_2', 'penalty_3', 'penalty_4', 'penalty_5',
        'penalty_6', 'penalty_7', 'penalty_8', 'penalty_9', 'penalty_10',
        'penalty_11', 'penalty_12']
paths = [a_path + i for i in paths_1]
title = 'ttH purity $\cdot$ significance'
labels = ['$\kappa=1.0$', '$\kappa=0.8$', '$\kappa=0.6$', '$\kappa=0.5$',
        '$\kappa=0.4$', '$\kappa=0.3$', '$\kappa=0.2$', '$\kappa=0.1$',
        '$\kappa=0.08$', '$\kappa=0.06$', '$\kappa=0.05$', '$\kappa=0.04$']
out_path = 'data/studies_ttH/penalty_2'
ttH_plotter.plot(paths, ttH_subpath, title, labels, [30,40], out_path)

