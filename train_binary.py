# written by Max Welsch.
import numpy as np
from tth_tensorflow.neuralnet.binary_mlp import BinaryMLP
from tth_tensorflow.neuralnet.data_frame import DataFrame



path = 'all_branches/63/'

train = np.load(path + 'even1_binary.npy')
val = np.load(path + 'odd1_binary.npy')
print(train.shape, val.shape)
# test = np.load(path + 'test.npy')

train = DataFrame(x=train[:, 1:-2], y=train[:, :1], ratio_weight=True)
val = DataFrame(x=val[:, 1:-2], y=val[:, :1])

N_EPOCHS = 20
print(train.nfeatures)
cl = BinaryMLP(train.nfeatures,
                [250,250], path + 'nn_models/2x250_binary')
cl.train(train, val, epochs=N_EPOCHS, keep_prob=0.9, batch_size=256, beta=1e-4)
