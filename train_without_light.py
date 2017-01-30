# written by Martin Lang.
import numpy as np
import datetime
from MLP.onehot_mlp import OneHotMLP
from DataFrame.data_frame import DataFrame


trainpath ='/storage/7/lang/nn_data/converted/all_categories/even1_bdt.npy'
valpath = '/storage/7/lang/nn_data/converted/all_categories/odd1_bdt.npy'
datestring = datetime.datetime.now().strftime("%Y_%m_%d")
outpath = 'data/executed/' + datestring + '/'
print('Loading data...')
train = np.load(trainpath)
val = np.load(valpath)
print('done.')

# print(train[0])

optname = 'Adam'
# implemented Optimizers: 
# 'Adam':               Adam Optimizer
# 'GradDescent':        Gradient Descent Optimizer
# 'Adagrad':            Adagrad Optimizer
# 'Adadelta':           Adadelta Optimizer
# 'Momentum':           Momentum Optimizer
optimizer_options = [0.9, 0.999, 1e-8]
beta = 1e-10
# Optimizer options may have different data types for different optimizers.
# 'Adam':               [beta1=0.9 (float), beta2=0.999 (float), epsilon=1e-8 (float)]
# 'GradDescent':        []
# 'Adagrad':            [initial_accumulator_value=0.1 (float)]
# 'Adadelta':           [rho=0.95 (float), epsilon=1e-8 (float)]
# 'Momentum':           [momentum=0.9 (float), use_nesterov=False (bool)]
# For information about these parameters please refer to the TensorFlow
# documentation.
outsize = 6
N_EPOCHS = 20
learning_rate = 1e-2
hidden_layers = [200, 200]
exec_name = '2x200_equalcat_bdt'
model_location = outpath + exec_name
labels = ['ttH', 'tt+bb', 'tt+2b', 'tt+b', 'tt+cc', 'tt+light']

# Be careful when editing the part below.
train = DataFrame(train, out_size=outsize)
val = DataFrame(val, out_size=outsize)

cl = OneHotMLP(train.nfeatures, 
        hidden_layers, outsize, model_location, labels)
cl.train(train, val, optimizer=optname, epochs=N_EPOCHS, batch_size=5000, learning_rate=
        learning_rate, keep_prob=0.95, beta=beta, out_size=outsize,
        optimizer_options=optimizer_options)
with open('{}/data_info.txt'.format(model_location), 'w') as out:
    out.write('Training data: {}\n'.format(trainpath))
    out.write('Validation data: {}\n'.format(valpath))
