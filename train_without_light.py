# written by Martin Lang.
import numpy as np
from onehot_mlp import OneHotMLP
from data_frame_oneHot import DataFrame


path = '/storage/7/lang/nn_data/converted/'
outpath = 'data/executed/'
print('Loading data...')
train = np.load(path + 'even1_without_light.npy')
val = np.load(path + 'odd1_without_light.npy')
print('done.')

# print(train[0])

beta = 0.0
outsize = 5
N_EPOCHS = 1000
learning_rate = 1e-5
hidden_layers = [200, 100]
exec_name = 'nn_models/200+100_oneHot_without_light'
train = DataFrame(train, out_size=outsize)
val = DataFrame(val, out_size=outsize)

cl = OneHotMLP(train.nfeatures, 
        hidden_layers, outsize, outpath + exec_name)
cl.train(train, val, epochs=N_EPOCHS, batch_size=256, learning_rate=
        learning_rate, keep_prob=0.9, beta=beta, out_size=outsize)

