import numpy as np
import sys

trainpath='/storage/7/lang/nn_data/converted/even_branches_corrected_30_20_10_01_light_weights0.npy'
valpath='/storage/7/lang/nn_data/converted/odd_branches_corrected_30_20_10_01_light_weights0.npy'

trainpath_weak='/storage/7/lang/nn_data/converted/even_branches_corrected_30_20_10_01_light_weights0_preselection_weak.npy'
valpath_weak='/storage/7/lang/nn_data/converted/odd_branches_corrected_30_20_10_01_light_weights0_preselection_weak.npy'
trainpath_strong='/storage/7/lang/nn_data/converted/even_branches_corrected_30_20_10_01_light_weights0_preselection_strong.npy'
valpath_strong='/storage/7/lang/nn_data/converted/odd_branches_corrected_30_20_10_01_light_weights0_preselection_strong.npy'

train = np.load(trainpath)
val = np.load(valpath)
train_weak = np.load(trainpath_weak)
val_weak = np.load(valpath_weak)
train_strong = np.load(trainpath_strong)
val_strong = np.load(valpath_strong)

y_train = train[:, :6]
y_val = val[:, :6]
y_train_weak = train_weak[:, :6]
y_val_weak = val_weak[:, :6]
y_train_strong = train_strong[:, :6]
y_val_strong = val_strong[:, :6]
labels = ['ttH / signal', 'tt+bb', 'tt+2b', 'tt+b', 'tt+cc', 'tt+light flavor',
        'total background', 'total events']
num_train = [np.count_nonzero(np.argmax(y_train, axis=1) == i) for i in
        range(6)]
num_val = [np.count_nonzero(np.argmax(y_val, axis=1) == i) for i in
        range(6)]
num_train_weak = [np.count_nonzero(np.argmax(y_train_weak, axis=1) == i) for i in
        range(6)]
num_val_weak = [np.count_nonzero(np.argmax(y_val_weak, axis=1) == i) for i in
        range(6)]
num_train_strong = [np.count_nonzero(np.argmax(y_train_strong, axis=1) == i) for i in
        range(6)]
num_val_strong = [np.count_nonzero(np.argmax(y_val_strong, axis=1) == i) for i in
        range(6)]
num_train.append(np.sum(num_train) - num_train[0])
num_val.append(np.sum(num_val) - num_val[0])
num_train_weak.append(np.sum(num_train_weak) - num_train_weak[0])
num_val_weak.append(np.sum(num_val_weak) - num_val_weak[0])
num_train_strong.append(np.sum(num_train_strong) - num_train_strong[0])
num_val_strong.append(np.sum(num_val_strong) - num_val_strong[0])
num_train.append(np.sum(num_train) - num_train[6])
num_val.append(np.sum(num_val) - num_val[6])
num_train_weak.append(np.sum(num_train_weak) - num_train_weak[6])
num_val_weak.append(np.sum(num_val_weak) - num_val_weak[6])
num_train_strong.append(np.sum(num_train_strong) - num_train_strong[6])
num_val_strong.append(np.sum(num_val_strong) - num_val_strong[6])
with open('numbers.txt', 'w') as f:
    f.write('category & training & validation & training & validation & training & validation\\\\ \n')
    for i in range(8): 
        f.write('{} & {} & {} & {} & {} & {} & {} \\\\ \n'.format(labels[i], 
            num_train[i], num_val[i], num_train_weak[i], num_val_weak[i], 
            num_train_strong[i], num_val_strong[i]))
# print(num_train)
# print(num_val)
# print(num_train_weak)
# print(num_val_weak)
# print(num_train_strong)
# print(num_val_strong)
