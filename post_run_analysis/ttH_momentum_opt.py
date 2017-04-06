import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def stats(arr):
    mean = np.mean(arr)
    stddev = np.std(arr)
    return mean, stddev

path1 = '../data/executed/analyses_ttH/momentum_settings/'
paths2 = ['man_9', 'man_8', 'man_7', 'man_6', 'man_5', 'man_4', 'man_3', 'man_2', 'momentum', 'man_1']
apps = ['', '_2', '_3', '_4', '_5']
paths_n = [path1 + i for i in paths2]
#labels = [r'$\mu = 0.5$', r'$\mu = 0.55$',
#        r'$\mu = 0.6$', r'$\mu = 0.65$', r'$\mu = 0.7$', r'$\mu = 0.75$', 
#        r'$\mu = 0.8$', r'$\mu = 0.85$', r'$\mu = 0.9$', r'$\mu = 0.95$']
labels = ['0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9',
        '0.95']
linewidth=3
# paths = [i+j for i in paths_n for j in apps]


purity = dict()
significance = dict()
# ttimes is actually not a time, but the number of epochs trained
ttimes = dict()

for path in paths2:
    purity[path] = []
    significance[path] = []
    ttimes[path] = []
    for j in range(len(apps)):
        with open(path1 + path + apps[j] + '/info.txt', 'r') as f:
            for line in f:
                if "validation purity" in line:
                    number = float(line.rsplit(' ')[3].strip('\n'))
                    purity[path].append(number)
                if "validation significance" in line:
                    number = float(line.rsplit(' ')[3].strip('\n'))
                    significance[path].append(number)
                if "Number of epochs" in line:
                    number = int(line.rsplit(' ')[4].strip('\n'))
                    ttimes[path].append(number)
products = dict()
means = dict()
stddevs = dict()
ttimes_means = dict()
ttimes_stddevs = dict()
for path in paths2:
    products[path] = []
    for j in range(len(purity[path])):
        products[path].append(purity[path][j] * significance[path][j])
    # print(products[path])
    products[path] = np.asarray(products[path])
    ttimes[path] = np.asarray(ttimes[path])
    means[path], stddevs[path] = stats(products[path])
    ttimes_means[path], ttimes_stddevs[path] = stats(ttimes[path])

means_list = []
stddevs_list = []
ttimes_means_list = []
ttimes_stddevs_list = []
for i in range(len(paths2)):
    means_list.append(means[paths2[i]])
    stddevs_list.append(stddevs[paths2[i]])
    ttimes_means_list.append(ttimes_means[paths2[i]])
    ttimes_stddevs_list.append(ttimes_stddevs[paths2[i]])
print(means_list)
print(stddevs_list)
print(ttimes_means_list)
print(ttimes_stddevs_list)

# print(means)
# print(stddevs)
xvalues = np.arange(0, len(paths2), 1)
# print(xvalues)
plt.xticks(xvalues, labels)
# plt.plot(xvalues, means_list, 'ro')
(_, caps, _) = plt.errorbar(xvalues, means_list, yerr=stddevs_list, linestyle="None",
        elinewidth=linewidth, color='navy')
for cap in caps:
    cap.set_markeredgewidth(linewidth)
ax = plt.gca()
ax.xaxis.grid(False)
ax.yaxis.grid(True)
plt.xlim(-0.5, len(paths2)-0.5)
plt.title(r'Product of purity and significance for different optimizers')
plt.ylabel(r'Product')
plt.xlabel(r'Friction parameter $\mu$')
plt.savefig('../data/studies_ttH/momentum_options/all_with_stddev.pdf')
plt.clf()
xvalues = np.arange(0, len(paths2), 1)
# print(xvalues)
plt.xticks(xvalues, labels)
# plt.plot(xvalues, means_list, 'ro')
(_, caps, _) = plt.errorbar(xvalues, ttimes_means_list, yerr=ttimes_stddevs_list,
        linestyle="None", elinewidth=linewidth, color='navy')
for cap in caps:
    cap.set_markeredgewidth(linewidth)
ax = plt.gca()
ax.xaxis.grid(False)
ax.yaxis.grid(True)
plt.xlim(-0.5, len(paths2)-0.5)
plt.title(r'Number of epochs trained')
plt.ylabel(r'Epochs trained')
plt.xlabel(r'Friction parameter $\mu$')
plt.savefig('../data/studies_ttH/momentum_options/num_of_epochs.pdf')
plt.clf()
