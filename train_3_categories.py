# written by Martin Lang.
import numpy as np
import datetime
from onehot_mlp_3_categories import OneHotMLP
from data_frame_oneHot_3_categories import DataFrame


trainpath = '/storage/7/lang/nn_data/converted/ttH_ttbb_ttlight/even1_bdt.npy'
valpath = '/storage/7/lang/nn_data/converted/ttH_ttbb_ttlight/odd1_bdt.npy'
datestring = datetime.datetime.now().strftime("%Y_%m_%d")
outpath = 'data/executed/' + datestring + '/'
 
print('Loading data...')
train = np.load(trainpath)
val = np.load(valpath)
print('done.')

# print(train[0])

beta = 1e-9
outsize = 3
N_EPOCHS = 500
learning_rate = 1e-6
hidden_layers = [400,400,400]
exec_name = '3x400_3_categories_bdt_equal_categories'
model_location = outpath + exec_name
train = DataFrame(train, out_size=outsize)
val = DataFrame(val, out_size=outsize)

cl = OneHotMLP(train.nfeatures, 
        hidden_layers, outsize, model_location)
cl.train(train, val, epochs=N_EPOCHS, batch_size=200, learning_rate=
        learning_rate, keep_prob=0.9, beta=beta, out_size=outsize)

with open('{}/data_info.txt'.format(model_location), 'w') as out:
    out.write('Training data: {}\n'.format(trainpath))
    out.write('Validation data: {}\n'.format(valpath))
