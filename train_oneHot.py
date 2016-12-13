# written by Martin Lang.
import numpy as np
from tth_tensorflow.neuralnet.onehot_mlp import OneHotMLP
from tth_tensorflow.neuralnet.data_frame import DataFrame



path = 'data/converted/'

train = np.load(path + 'even1.npy')
val = np.load(path + 'odd1.npy')
print(train.shape, val.shape)
# test = np.load(path + 'test.npy')

train = DataFrame(x=train[:, 1:-2], y=train[:, :1], ratio_weight=True)
val = DataFrame(x=val[:, 1:-2], y=val[:, :1])

outsize = 6
N_EPOCHS = 500
print(train.nfeatures)
cl = OneHotMLP(train.nfeatures,
                [250,250], outsize, path + 'nn_models/2x250_oneHot')
cl.train(train, val, epochs=N_EPOCHS, keep_prob=0.9, batch_size=256,beta=1e-4,
        out_size=outsize)
