# written by Martin Lang.
import numpy as np
import datetime
from MLP.onehot_mlp_balanced import OneHotMLP
from DataFrame.balanced_data_frame import DataFrame


trainpath = '/storage/7/lang/nn_data/converted/all_categories/even3_bdt.npy'
valpath = '/storage/7/lang/nn_data/converted/all_categories/odd3_bdt.npy'
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
optimizer_options = []
beta = 1e-8
# Optimizer options may have different data types for different optimizers.
# 'Adam':               [beta1=0.9 (float), beta2=0.999 (float), epsilon=1e-8 (float)]
# 'GradDescent':        []
# 'Adagrad':            [initial_accumulator_value=0.1 (float)]
# 'Adadelta':           [rho=0.95 (float), epsilon=1e-8 (float)]
# 'Momentum':           [momentum=0.9 (float), use_nesterov=False (bool)]
# For information about these parameters please refer to the TensorFlow
# documentation.
outsize = 6
N_EPOCHS = 40
learning_rate = 1e-2
hidden_layers = [200, 200, 200, 200, 200]
exec_name = '5x200_equal_events_bdt'
model_location = outpath + exec_name
train_sizes = np.array([143639, 43665, 29278, 60445, 127357, 453687])
val_sizes = np.array([143511, 43074, 29451, 60860, 127646, 454255])
labels = ['ttH', 'tt+bb', 'tt+2b', 'tt+b', 'tt+cc', 'tt+light']
# Choose normalization from 'minmax' or 'gaussian'.
normalization = 'gaussian'

# Be careful when editing the part below.
train = DataFrame(train, out_size=outsize, sizes = train_sizes, 
        normalization=normalization)
val = DataFrame(val, out_size=outsize, sizes = val_sizes,
        normalization=normalization)

cl = OneHotMLP(train.nfeatures, 
        hidden_layers, outsize, model_location, labels_text=labels)
cl.train(train, val, optimizer=optname, epochs=N_EPOCHS, batch_size=2000, learning_rate=
        learning_rate, keep_prob=0.95, beta=beta, out_size=outsize,
        optimizer_options=optimizer_options, early_stop=early_stop)
with open('{}/data_info.txt'.format(model_location), 'w') as out:
    out.write('Training data: {}\n'.format(trainpath))
    out.write('Validation data: {}\n'.format(valpath))
