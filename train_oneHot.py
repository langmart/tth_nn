# written by Martin Lang.
import numpy as np
from tth_tensorflow.neuralnet.onehot_mlp import OneHotMLP
from tth_tensorflow.neuralnet.data_frame import DataFrame



path = 'data/bdt/63_wo_weights/'

train = np.load(path + 'train.npy')
val = np.load(path + 'val.npy')
print(train.shape, val.shape)
# test = np.load(path + 'test.npy')

train = DataFrame(x=train[:, 1:-2], y=train[:, :1], ratio_weight=True)
val = DataFrame(x=val[:, 1:-2], y=val[:, :1])

outsize = 1
N_EPOCHS = 50
print(train.nfeatures)
cl = OneHotMLP(train.nfeatures,
                [100,150,100,50], outsize, path + 'nn_models/4x100_wratio')
cl.train(train, val, epochs=N_EPOCHS, keep_prob=0.9, batch_size=256,beta=1e-4,
        out_size=outsize)
