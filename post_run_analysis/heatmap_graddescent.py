import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import os

outpath = '../data/studies/modified/grad_descent_heat/'
if not os.path.isdir(outpath):
    os.makedirs(outpath)

def find_limits(arr):
    minimum = np.min(arr) / (np.pi**2.0 * np.exp(1.0)**2.0)
    maximum = np.max(arr) * np.pi**2.0 * np.exp(1.0)**2.0
    return minimum, maximum

titlefontsize = 17
labelfontsize = 14
tickfontsize = 13
numberfontsize = 13
colorbarfontsize = 13
labels_text = ['ttH', 'tt+bb', 'tt+2b', 'tt+b', 'tt+cc', 'tt+light']
arr = [[88400, 2654, 802, 1731, 4906, 45018],
        [16650, 3996, 532, 1660, 3316, 16920], 
        [8029, 931, 1529, 1529, 2235, 15198],
        [12336, 1637, 798, 4182, 3808, 38099],
        [16713, 1607, 831, 1933, 10309, 96253],
        [21904, 1371, 748, 2521, 10352, 417359]]
arr_cross = np.asarray(arr)
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
        plt.text(xit + 0.5, yit + 0.5, '%i' % arr_cross[yit, xit], 
                horizontalalignment='center', verticalalignment='center',
                fontsize=numberfontsize)
ax = plt.gca()
ax.set_xticks(np.arange((x.shape[0] - 1)) + 0.5, minor=False)
ax.set_yticks(np.arange((y.shape[0] - 1)) + 0.5, minor=False)
ax.set_xticklabels(labels_text, fontsize=tickfontsize)
ax.set_yticklabels(labels_text, fontsize=tickfontsize)
cbar.ax.tick_params(labelsize=colorbarfontsize)
plt.title('Heat map: Validation after early stopping in epoch 401',
        fontsize=titlefontsize)
plt.savefig(outpath + '/401_validation_colorlog_absolute.pdf')
plt.tight_layout()
plt.clf()
