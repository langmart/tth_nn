# written by Martin Lang.
import numpy as np
import datetime
from onehot_mlp_crosscheck_balanced import OneHotMLP
from balanced_data_frame_oneHot import DataFrame


trainpath = '/storage/7/lang/nn_data/converted/all_categories/even1_equal_events_bdt.npy'
valpath = '/storage/7/lang/nn_data/converted/all_categories/odd1_equal_events_bdt.npy'
datestring = datetime.datetime.now().strftime("%Y_%m_%d")
outpath = 'data/executed/' + datestring + '/'
print('Loading data...')
train = np.load(trainpath)
val = np.load(valpath)
print('done.')


# print(train[0])

beta = 1e-10
outsize = 6
N_EPOCHS = 300
learning_rate = 2e-2
hidden_layers = [300,300,300,300]
exec_name = '4x300'
model_location = outpath + exec_name
train_sizes = np.array([143639, 43665, 29278, 60445, 127357, 453687])
val_sizes = np.array([143511, 43074, 29451, 60860, 127646, 454255])

train = DataFrame(train, out_size=outsize, sizes = train_sizes)
val = DataFrame(val, out_size=outsize, sizes = val_sizes)

cl = OneHotMLP(train.nfeatures, 
        hidden_layers, outsize, model_location)
cl.train(train, val, epochs=N_EPOCHS, batch_size=500, learning_rate=
        learning_rate, keep_prob=0.2, beta=beta, out_size=outsize)
with open('{}/data_info.txt'.format(model_location), 'w') as out:
    out.write('Training data: {}\n'.format(trainpath))
    out.write('Validation data: {}\n'.format(valpath))

