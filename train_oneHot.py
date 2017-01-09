# written by Martin Lang.
import numpy as np
from onehot_mlp import OneHotMLP
from data_frame_oneHot import DataFrame


path = '/storage/7/lang/nn_data/converted/'
outpath = 'data/executed/'
print('Loading data...')
train = np.load(path + 'even3_bdt_evt_jets_weights.npy')
val = np.load(path + 'odd3_bdt_evt_jets_weights.npy')
print('done.')

# print(train[0])

beta = 1e-8
outsize = 6
N_EPOCHS = 2000
learning_rate = 1e-7
hidden_layers = [300,300]
exec_name = 'nn_models/2x300_testing'
train = DataFrame(train, out_size=outsize)
val = DataFrame(val, out_size=outsize)

cl = OneHotMLP(train.nfeatures, 
        hidden_layers, outsize, outpath + exec_name)
cl.train(train, val, epochs=N_EPOCHS, batch_size=20000, learning_rate=
        learning_rate, keep_prob=0.5, beta=beta, out_size=outsize)

