import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import os

main_path = '../data/executed/'
path = main_path + 'analyses_ttH/momentum_settings/man_7_2' + '/cross_checks/'
fileinpred = 'val_pred_113.txt'
fileintrue = 'val_true_113.txt'
epoch = int(fileinpred.rsplit('_')[2].rsplit('.')[0])
print(epoch)
outpath = '../data/studies_ttH/modified/comp_w_n/hists/'
if not os.path.isdir(outpath):
    os.makedirs(outpath)
weight_path = '/storage/7/lang/nn_data/converted/weights.txt'
labels_text = ['ttH', 'tt+bb', 'tt+2b', 'tt+b', 'tt+cc', 'tt+light']

hist_colors = ['navy', 'firebrick', 'darkgreen', 'purple', 'darkorange',
                'lightseagreen']
titlefontsize = 17
labelfontsize = 17
tickfontsize = 17
numberfontsize = 13
colorbarfontsize = 13
lw = 1.7

with open(path + fileinpred, 'rb') as f:
    arrpred = pickle.load(f)
with open(path + fileintrue, 'rb') as f:
    arrtrue = pickle.load(f)
print(arrpred[0])
print(arrpred.shape)
val_pred = arrpred
val_true = arrtrue
bins = np.linspace(0,1,101)
for i in range(val_pred.shape[1]):
    for j in range(val_pred.shape[1]):
        arr = np.where(np.argmax(val_true, axis=1)==j)
        histo_list = np.transpose(val_pred[arr, i])
        if histo_list.size:
            plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                    normed=True, histtype='step',label=labels_text[j],
                    linewidth=lw)
    plt.xlabel('{} node output'.format(labels_text[i]), fontsize=labelfontsize)
    plt.ylabel('Arbitrary units', fontsize=labelfontsize)
    plt.title('{} node output on validation set'.format(labels_text[i]),
            fontsize=titlefontsize)
    plt.legend(loc='best')
    plt.savefig(outpath + str(epoch) + '_' + str(i+1)+ '.pdf')
    plt.clf()
for i in range(val_pred.shape[1]):
    for j in range(val_pred.shape[1]):
        arr1 = (np.argmax(val_true, axis=1)==j)
        arr2 = (np.argmax(val_pred, axis=1)==i)
        arr = np.multiply(arr1, arr2)
        histo_list = val_pred[arr,i]
        if histo_list.size:
            plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                    normed=True, histtype='step',label=labels_text[j],
                    linewidth=lw)
    plt.xlabel('{} node output'.format(labels_text[i]), fontsize=labelfontsize)
    plt.ylabel('Arbitrary units', fontsize=labelfontsize)
    plt.title('output for predicted {} on the validation set'.format( 
        labels_text[i]), fontsize=titlefontsize)
    if (i == 5):
        plt.legend(loc='upper left')
    else:
        plt.legend(loc='best')
    plt.savefig(outpath + str(epoch) + '_' +
            str(i+1)+'_predicted.pdf')
    plt.clf()
for i in range(val_pred.shape[1]):
    for j in range(val_pred.shape[1]):
        arr = np.where(np.argmax(val_true, axis=1)==j)
        histo_list = np.transpose(val_pred[arr,i])
        if histo_list.size:
            plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                    normed=True, histtype='step',label=labels_text[j],
                    log=True, linewidth=lw)
    plt.xlabel('{} node output'.format(labels_text[i]), fontsize=labelfontsize)
    plt.ylabel('Arbitrary units', fontsize=labelfontsize)
    plt.title('{} node output on validation set'.format( labels_text[i]),
            fontsize=titlefontsize)
    if (i == 5):
        plt.legend(loc='lower center')
    else:
        plt.legend(loc='best')
    plt.savefig(outpath + str(epoch) + '_' + str(i+1)+
            '_log.pdf')
    plt.clf()
for i in range(val_pred.shape[1]):
    for j in range(val_pred.shape[1]):
        arr1 = (np.argmax(val_true, axis=1)==j)
        arr2 = (np.argmax(val_pred, axis=1)==i)
        arr = np.multiply(arr1, arr2)
        histo_list = val_pred[arr,i]
        if histo_list.size:
            plt.hist(histo_list, bins, alpha=1.0, color=hist_colors[j], 
                    normed=True, histtype='step',label=labels_text[j],
                    log=True, linewidth=lw)
    plt.xlabel('{} node output'.format(labels_text[i]), fontsize=labelfontsize)
    plt.ylabel('Arbitrary units', fontsize=labelfontsize)
    plt.title('output for predicted {} on the validation set'.format(
        labels_text[i]), fontsize=titlefontsize)
    plt.legend(loc='best')
    plt.savefig(outpath + str(epoch) + '_' +
            str(i+1)+'_predicted_log.pdf')
    plt.clf()
