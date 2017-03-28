import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

path1 = '../data/executed/analyses/batch_size/'
sizes_list = []
times_list = []
acc_list = []
acc_epoch = []
for i in range(1,15):
    path = path1 + 'bs_{}/'.format(i)
    with open(path + 'info.txt', 'r') as f:
        for line in f:
            if "Training Time" in line:
                times_list.append(float(line.rsplit(' ')[2]))
            if "Batch Size" in line:
                sizes_list.append(int(line.rsplit(' ')[2].strip('\n')))
            if "Best validation accuracy" in line:
                acc_list.append(float(line.rsplit(' ')[3]))
            if "Best validation epoch" in line:
                acc_epoch.append(int(line.rsplit(' ')[3].strip('\n')))
print(sizes_list)
print(times_list)
print(acc_list)
print(acc_epoch)

outpath = '../data/studies/batch_size/data.txt'
with open(outpath, 'w') as f:
    f.write('Batch size & best accuracy & epochs & training time & time per epoch \\\\ \n')
    for i in range(len(sizes_list)):
        f.write('{} & {:.4f} & {} & {:.0f} & {:.2f} \\\\ \n'.format(sizes_list[i], acc_list[i],
            acc_epoch[i], times_list[i], times_list[i] / acc_epoch[i]))
