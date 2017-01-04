# written by Martin Lang.
import numpy as np
from onehot_mlp import OneHotMLP
from data_frame_oneHot import DataFrame


path = '/storage/7/lang/nn_data/converted/'
outpath = 'data/executed/'
print('Loading data...')
train = np.load(path + 'even1_bdt_evt_jets_weights.npy')
val = np.load(path + 'odd1_bdt_evt_jets_weights.npy')
print('done.')

# print(train[0])

beta = 0.0
outsize = 6
N_EPOCHS = 1000
learning_rate = 1e-6
hidden_layers = [500, 500, 500]
exec_name = 'nn_models/3x500_bdt_evt_jets'
train = DataFrame(train, out_size=outsize)
val = DataFrame(val, out_size=outsize)

cl = OneHotMLP(train.nfeatures, 
        hidden_layers, outsize, outpath + exec_name)
cl.train(train, val, epochs=N_EPOCHS, batch_size=20000, learning_rate=
        learning_rate, keep_prob=0.7, beta=beta, out_size=outsize)

