from __future__ import absolute_import, division, print_function

import numpy as np
import os
import ROOT
import root_numpy as rnp

class Root2Array:
    """Convert ROOT nTuples into Numpy ndarrays."""

    def __init__(self, rdirs, label):
        """Initializes the converter.

        Parameters
        ----------
        rdirs : list of str
            Paths to the signal ROOT files .
        bdirs : list of str
            Paths to the background ROOT files .
        """
        self.rdirs = rdirs
        self.label = label

    def _get_data_array(self, data):
        """Takes a structured array type and converts it to a standard numpy
        ndarray.

        Returns
        -------
        ndarray : np.ndarray
            Structured array as ndarray.
        new_branches : list of str
            List containing the corresponding names to the features.
        """

        data, branches = self._to_ndarr(data)
        data, branches = self._add_labels(data, branches)

        return data, branches

    def _to_ndarr(self, struct_arr):
        ndarr = []
        new_branches = []

        # get vector like branches
        fdir = os.path.dirname(__file__)
        with open('{}/config/jets.txt'.format(fdir), 'r') as f:
            jets = [line.rstrip('\n') for line in f]

        for branch in self.branchlist:
            if branch in jets:
                arr = [jet[:4] for jet in struct_arr[branch]]
                ndarr.append(np.vstack(arr))
                new_branches += [branch+'_{}'.format(i) for i in range(1,5)]
            else:
                ndarr.append(struct_arr[branch].reshape(-1,1))
                new_branches.append(branch)

        ndarr = np.hstack(ndarr)

        return ndarr, new_branches

    def _add_labels(self, data, branches):
        labels = np.full((data.shape[0], 1), self.label)
        data = np.hstack((labels, data))
        branches = ['Label'] + branches

        return data, np.array(branches)

    def _get_structured_array(self, rfiles, name):

        N_FILES = len(rfiles)
        arr = []

        print('Convert {}'.format(name))
        print(25*'-')
        for n_conv, rfile in enumerate(rfiles):
            # open ROOT file an convert the tree to a structured array
            f = ROOT.TFile.Open(rfile)
            tree = f.Get(self.tree_name)
            struct_arr = rnp.tree2array(tree, self.branchlist)
            arr.append(struct_arr)
            print('{}/{} files converted.'.format(n_conv+1, N_FILES))

        print(25*'-')

        arr = np.concatenate(arr, axis=0)

        return arr


    def convert(self, tree_name, branchlist, savepath, name):
        """Uses root_numpy's tree2array to convert the nTuples into structured
        np arrays. Also gets rid of structured array and saves the data as a
        np.ndarrays. Axis 0 represents the different branches/variables and axis
        1 represents single events.

        Parameters
        ----------
        tree_name : str
            Tree name in ROOT file.
        branchlist : str
            Path to the txt file containing branches.
        savepath : str
            Path to the directory the converted array will be saved in.
        name : str
            The array's name after saving in the defined directory.
        """
        self.tree_name = tree_name
        self.branchlist = branchlist
        self.savepath = savepath

        if not os.path.isdir(self.savepath):
            os.makedirs(self.savepath)

        with open(self.branchlist, 'r') as f:
            self.branchlist = [line.rstrip('\n') for line in f]

        data = self._get_structured_array(self.rdirs, name)

        data, columns = self._get_data_array(data)

        arrayname = self.savepath + '/' + name
        np.save(arrayname + '.npy', data)
        np.save(arrayname + '_columns.npy', columns)

        print('Array saved in "{}".npy'.format( arrayname))
