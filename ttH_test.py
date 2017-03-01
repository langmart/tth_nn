# written by Martin Lang.
import numpy as np
import datetime
from MLP.ttH_classifier_testing import OneHotMLP
from DataFrame.data_frame import DataFrame


trainpath='/storage/7/lang/nn_data/converted/even_branches_corrected_30_20_10_01_light_weights1.npy'
valpath='/storage/7/lang/nn_data/converted/odd_branches_corrected_30_20_10_01_light_weights1.npy'
weight_path = '/storage/7/lang/nn_data/converted/weights.txt'
branchlist='branchlists/branches_corrected_converted.txt'
exec_name = 'ttH_test'
with open(weight_path, 'r') as f:
    weights = [line.strip() for line in f]
    sig_weight = np.float32(weights[0])
    bg_weight = np.float32(weights[1])
outpath = 'data/executed/ttH/'
print('Loading data...')
train = np.load(trainpath)
val = np.load(valpath)
print('done.')
labels = ['ttH', 'tt+bb', 'tt+2b', 'tt+b', 'tt+cc', 'tt+light']
model_location = outpath + exec_name

# For information on some of the following options see below.
optname = 'Adam'
optimizer_options = []
beta = 1e-5
outsize = 6
N_EPOCHS = 500
learning_rate = 2e-3
batch_size = 15000 
hidden_layers = [200, 200, 200]
keep_prob = 0.8
normalization = 'gaussian'
decay_learning_rate = 'yes'
lrate_decay_options = [0.97, 200]
batch_decay = 'no'
batch_decay_options = []
ttH_penalty = 1e-5
enable_early='yes'
early_stop = 20

# Be careful when editing the part below.
train = DataFrame(train, out_size=outsize, normalization=normalization)
val = DataFrame(val, out_size=outsize, normalization=normalization)

cl = OneHotMLP(train.nfeatures, hidden_layers, outsize, model_location, 
        labels_text=labels, branchlist=branchlist, sig_weight=sig_weight,
        bg_weight=bg_weight)
cl.train(train, val, optimizer=optname, epochs=N_EPOCHS, batch_size=batch_size, 
        learning_rate= learning_rate, keep_prob=keep_prob, beta=beta, 
        out_size=outsize, optimizer_options=optimizer_options, 
        decay_learning_rate=decay_learning_rate,
        dlrate_options=lrate_decay_options, batch_decay=batch_decay,
        batch_decay_options=batch_decay_options, enable_early=enable_early,
        early_stop=early_stop, ttH_penalty=ttH_penalty)
with open('{}/data_info.txt'.format(model_location), 'w') as out:
    out.write('Training data: {}\n'.format(trainpath))
    out.write('Validation data: {}\n'.format(valpath))

# implemented Optimizers: 
# 'Adam':               Adam Optimizer
# 'GradDescent':        Gradient Descent Optimizer
# 'Adagrad':            Adagrad Optimizer
# 'Adadelta':           Adadelta Optimizer
# 'Momentum':           Momentum Optimizer
# Optimizer options may have different data types for different optimizers.
# 'Adam':               [beta1=0.9 (float), beta2=0.999 (float), epsilon=1e-8 (float)]
# 'GradDescent':        []
# 'Adagrad':            [initial_accumulator_value=0.1 (float)]
# 'Adadelta':           [rho=0.95 (float), epsilon=1e-8 (float)]
# 'Momentum':           [momentum=0.9 (float), use_nesterov=False (bool)]
# For information about these parameters please refer to the TensorFlow
# documentation.
# Choose normalization from 'minmax' or 'gaussian'.
