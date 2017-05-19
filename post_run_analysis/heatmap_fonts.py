import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import os

def find_limits(arr):
    minimum = np.min(arr) / (np.pi**2.0 * np.exp(1.0)**2.0)
    maximum = np.max(arr) * np.pi**2.0 * np.exp(1.0)**2.0
    return minimum, maximum


main_path = '../data/executed/'
path = main_path + 'analyses_ttH/betattH/man_7' + '/cross_checks/'
fileinpred = 'val_pred_136.txt'
fileintrue = 'val_true_136.txt'
epoch = int(fileinpred.rsplit('_')[2].rsplit('.')[0])
print(epoch)
# outpath = '../data/studies_ttH/modified/betattH/'
outpath = '../data/studies_ttH/modified/betattH/'
if not os.path.isdir(outpath):
    os.makedirs(outpath)
weight_path = '/storage/7/lang/nn_data/converted/weights.txt'
labels_text = ['ttH', 'tt+bb', 'tt+2b', 'tt+b', 'tt+cc', 'tt+light']
with open(weight_path, 'r') as f:
    weights = [line.strip() for line in f]
    sig_weight = np.float32(weights[0])
    bg_weight = np.float32(weights[1])

titlefontsize = 17
labelfontsize = 14
tickfontsize = 13
numberfontsize = 13
colorbarfontsize = 13

with open(path + fileinpred, 'rb') as f:
    arrpred = pickle.load(f)
with open(path + fileintrue, 'rb') as f:
    arrtrue = pickle.load(f)
print(arrpred[0])
print(arrpred.shape)
arr_cross = np.zeros((arrpred.shape[1], arrpred.shape[1]), dtype=np.float32)
index_true = np.argmax(arrtrue, axis=1)
index_pred = np.argmax(arrpred, axis=1)
for i in range(index_true.shape[0]):
    arr_cross[index_true[i]][index_pred[i]] += 1
print(arr_cross)
for i in range(arr_cross.shape[0]):
    for j in range(arr_cross.shape[1]):
        if (i==0):
            arr_cross[i][j] *= sig_weight
        else:
            arr_cross[i][j] *= bg_weight
print(arr_cross)
x = np.linspace(0, arr_cross.shape[0], arr_cross.shape[0] + 1)
y = np.linspace(0, arr_cross.shape[0], arr_cross.shape[0] + 1)
xn, yn = np.meshgrid(x,y)
cmap = matplotlib.cm.RdYlBu_r
cmap.set_bad(color='white')
minimum, maximum = find_limits(arr_cross)
plt.pcolormesh(xn, yn, arr_cross, cmap=cmap, norm=colors.LogNorm(
    vmin=max(minimum, 1e-6), vmax=maximum))
cbar = plt.colorbar()
plt.xlim(0, arr_cross.shape[0])
plt.ylim(0, arr_cross.shape[1])

plt.xlabel("Predicted", fontsize=labelfontsize)
plt.ylabel("True", fontsize=labelfontsize)
for yit in range(arr_cross.shape[0]):
    for xit in range(arr_cross.shape[1]):
        plt.text(xit + 0.5, yit + 0.5, '%.2f' % arr_cross[yit, xit], 
                horizontalalignment='center', verticalalignment='center',
                fontsize=numberfontsize)
ax = plt.gca()
ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
ax.set_xticklabels(labels_text, fontsize=tickfontsize)
ax.set_yticklabels(labels_text, fontsize=tickfontsize)
cbar.ax.tick_params(labelsize=colorbarfontsize)
plt.title('Heat map: Validation after early stopping in epoch {}'.format(epoch),
        fontsize=titlefontsize)
plt.savefig(outpath + '/{}_validation_colorlog_absolute_weights.pdf'.format(epoch))
plt.tight_layout()
plt.clf()
