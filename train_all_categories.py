# written by Martin Lang.
import numpy as np
import datetime
from onehot_mlp import OneHotMLP
from data_frame import DataFrame


trainpath = '/storage/7/lang/nn_data/converted/all_categories/even1_bdt.npy'
valpath = '/storage/7/lang/nn_data/converted/all_categories/odd1_bdt.npy'
datestring = datetime.datetime.now().strftime("%Y_%m_%d")
outpath = 'data/executed/' + datestring + '/'
 
print('Loading data...')
train = np.load(trainpath)
val = np.load(valpath)
print('done.')

# print(train[0])

beta = 1e-9
outsize = 6
N_EPOCHS = 30
learning_rate = 0.0005
hidden_layers = [150, 150, 150]
exec_name = '3x150_equalcat_bdt'
model_location = outpath + exec_name
labels = ['ttH', 'tt+bb', 'tt+2b', 'tt+b', 'tt+cc', 'tt+light']

train = DataFrame(train, out_size=outsize)
val = DataFrame(val, out_size=outsize)

cl = OneHotMLP(train.nfeatures, 
        hidden_layers, outsize, model_location, labels)
cl.train(train, val, epochs=N_EPOCHS, batch_size=2000, learning_rate=
        learning_rate, keep_prob=0.9, beta=beta, out_size=outsize)

with open('{}/data_info.txt'.format(model_location), 'w') as out:
    out.write('Training data: {}\n'.format(trainpath))
    out.write('Validation data: {}\n'.format(valpath))
