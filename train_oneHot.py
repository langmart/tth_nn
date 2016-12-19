# written by Martin Lang.
import numpy as np
from onehot_mlp import OneHotMLP
from data_frame_oneHot import DataFrame


path = 'data/converted/'
outpath = 'data/executed/'
print('Loading data...')
train = np.load(path + 'even1_new.npy')
val = np.load(path + 'odd1_new.npy')
print('done.')

outsize = 6
N_EPOCHS = 100
learning_rate = 1e-5
hidden_layers = [250, 250, 250, 250]
exec_name = 'nn_models/4x250_oneHot'
train = DataFrame(train, out_size=outsize)
val = DataFrame(val, out_size=outsize)

cl = OneHotMLP(train.nfeatures,
                hidden_layers, outsize, outpath + exec_name)
cl.train(train, val, epochs=N_EPOCHS, batch_size=256, learning_rate =
        learning_rate, keep_prob=0.9, beta=1e-16,
        out_size=outsize)

