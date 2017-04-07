import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle

plot_colors = ['navy', 'darkorange', 'darkgreen']

path = '../data/executed/analyses_ttH/betattH/beta_ttH_4_5/cross_checks/'
with open(path + 'val_purity.txt', 'rb') as f:
    pur_list = np.asarray(pickle.load(f))
with open(path + 'val_significance.txt', 'rb') as f:
    sig_list = np.asarray(pickle.load(f))
#print(pur_list)
#print(sig_list)
products = np.multiply(pur_list, sig_list)
plt.plot(pur_list, label=r'purity', color='navy')
plt.plot(sig_list, label=r'significance', color='darkorange')
plt.plot(products, label=r'product', color='darkgreen')
plt.legend(loc='best')
plt.title(r'Purity, significance, and product')
plt.xlabel(r'Epoch')
plt.ylabel(r'Result')
plt.savefig('../data/studies_ttH/betattH/multi_plot.pdf')
plt.clf()
