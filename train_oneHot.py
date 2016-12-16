# written by Martin Lang.
import numpy as np
from tth_tensorflow.neuralnet.onehot_mlp import OneHotMLP
from data_frame_oneHot import DataFrame
# from onehot_mlp import OneHotMLP
# from data_frame import DataFrame


path = 'data/converted/'

print('Loading data...')
train = np.load(path + 'even1_test_array.npy')
val = np.load(path + 'odd1_test_array.npy')
print('done.')
print(train[0][0:6])
print(train[100000][0:6])
print(train[200000][0:6])
print(train[300000][0:6])

print('Shape of train: {}'.format(train.shape))
print('Shape of val: {}'.format(val.shape))


outsize = 6
print(110*'-')
print(110*'-')
print(110*'-')
print('Getting dataframes.')
# train = DataFrame(x=train[:, 1:-2], y=train[:, :1], out_size=outsize, ratio_weight=True)
# val = DataFrame(x=val[:, 1:-2], y=val[:, :1], out_size=outsize)
train = DataFrame(train, out_size=outsize)
val = DataFrame(val, out_size=outsize)
print('Done.')
print(110*'-')
print(110*'-')
print(110*'-')

# train = DataFrame(even)
# test = DataFrame(odd)
# print('shape of training data: {}'.format(train.shape))
# print('shape of validation data: {}'.format(test.shape))

N_EPOCHS = 20

print(train.nfeatures)
print(110*'-')
print(110*'-')
print(110*'-')
print('Starting training.')
print(110*'-')
print(110*'-')
print(110*'-')
cl = OneHotMLP(train.nfeatures,
                [250,250], outsize, path + 'nn_models/2x250_oneHot')
cl.train(train, val, epochs=N_EPOCHS, batch_size=256, keep_prob=0.9, beta=1e-4,
        out_size=outsize)
print(train.nfeatures)
print(110*'-')
print(110*'-')
print(110*'-')
print('Training completed.')
print(110*'-')
print(110*'-')
print(110*'-')

