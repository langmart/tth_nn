from __future__ import absolute_import, division, print_function
import numpy as np
import os
import ROOT
import root_numpy as rnp

class Root2Numpy:
    def __init__(self, rfiles, tree):
        self.rfiles = rfiles
        self.tree = tree

    def convert(self, savedir, filename, branchlist=None):
        # create save directory if needed
        if not os.path.isdir(savedir):
            os.makedirs(savedir)

        # read branchlist
        if branchlist is not None:
            with open(branchlist, 'r') as f:
                branchlist = [line.rstrip('\n') for line in f]

        # data = rnp.root2array(self.rdirs, treename=self.tree,
        #             branches=branchlist)
        N_FILES = len(self.rfiles)
        data = []
        print('Convert {}'.format(filename))
        print(25*'-')
        for n_conv, rfile in enumerate(self.rfiles):
            # open ROOT file an convert the tree to a structured array
            f = ROOT.TFile.Open(rfile)
            tree = f.Get(self.tree)
            struct_arr = rnp.tree2array(tree, branchlist)
            data.append(struct_arr)
            print('{}/{} files converted.'.format(n_conv+1, N_FILES))
        print(25*'-')

        data = np.concatenate(data, axis=0)

        arrayname = savedir +'/' + filename + '.npy'
        np.save(arrayname, data)
        print('Array saved in "{}".npy'.format( arrayname))
