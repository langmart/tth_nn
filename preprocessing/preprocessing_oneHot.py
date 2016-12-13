from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#from root_numpy import root2array

# def conv_root2array(save_path, name, files, treename=None, branches=None):
#     """Convert ROOT trees into numpy structured array.
# 
#     Arguments:
#     ----------------
#     save_path (str):
#         Path to the directory the array will be saved in.
#     files (str or list(str)):
#         The name of the files that will be converted into ONE structured array.
#     treename (str, optional (default=None)):
#         Name of the tree to convert.
#     branches (str of list(str), optional(default = None)):
#         List of branch names to include as columns of the array. If None all
#         branches will be included.
#     """
# 
#     arr = root2array(files, treename, branches)
# 
#     if not os.path.isdir(save_path):
#         os.makedirs(save_path)
# 
#     np_file = save_path + '/' + name
#     np.save(npfile, arr)

class GetBranches:
    """Keeps only events of a certain category from numpy structured arrays.

    Attributes:
    ----------------
    categories (list(str)):
        Categories which will be extracted.
    branchlist (str):
        Path to the text file with the branches to be used.
    out_size (int):
        dimension of the output vector, i.e. labels.
    branches (list(str)):
        A list filled with the branches from branchlist.
    save_path (str):
        Path to the directory the processed array will be saved to.
    arr_name (str):
        Name of the new array.
    """

    def __init__(self, savedir, branchlist, categories=['30','20','10','01','light'], out_size = 6):
        """Initializes the class with the given attributes.

        Attributes:
        ----------------
        savedir (str):
            Path to the directory to which the converted data will be saved.
        category (str):
            Category to be extracted from data.
        branchlist (str):
            List containing all branches to be extracted from data.
        out_size (int, optional (default = 6)):
            Dimension of the output vector (labels must be of according
            dimension; by default 1 signal + 5 different backgrounds.

        """

        self.categories = categories
        self.out_size = out_size
        # self.save_path = branchlist.split('/')[-1].split('.')[0] + '/' +category
        self.savedir = savedir
        self.save_path = savedir + '/converted/'

        # read text file in list
        with open(branchlist, 'r') as f:
            self.branches = [line.strip() for line in f]

    def process(self, signal_path, background_path, arr_name):
        self.arr_name = arr_name

        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
            print("Created directory {}.".format(self.save_path))

        print('Loading: {} '.format(signal_path), end='')
        structured_sig = np.load(signal_path, encoding='latin1')
        print('done.')

        print('Loading: {} '.format(background_path), end='')
        structured_bg = np.load(background_path, encoding='latin1')
        print('done.')

        print('Getting branches, ', end='')
        sig_data, sig_branches = self._get_branches(structured_sig,
                self.branches)
        bg_data, bg_branches = self._get_branches(structured_bg, self.branches)
        # print('Background branches: {}'.format(bg_branches))
        print('done.')

        if (sig_branches == bg_branches):
            self.branches = sig_branches

        # TODO: implement control plots
        # print('Doing control plots, ', end='')
        # self._control_plot(sig_data, bg_data, self.branches)
        # print('Done.')

        sig = {'data': sig_data}
        bg = {'data': bg_data}
        # print(bg)

        print('Calculating weights, ', end='')
        sig['weights'] = self._get_weights(structured_sig)
        bg['weights'] = self._get_weights(structured_bg)
        print('done.')

        n_sig_events = sig['data'].shape[0]
        n_bg_events = bg['data'].shape[0]

        print('Getting labels, ', end='')
        sig['labels'] = self._get_labels(sig, structured_sig, n_sig_events,'sig')
        bg['labels'] = self._get_labels(bg, structured_bg, n_bg_events, 'bg')
        print('done.')

        print('Getting categories: {} , '.format(self.categories), end='')
        # signal should not be categorized
        # sig = self._get_category(sig, structured_sig)
        bg = self._get_category(bg, structured_bg)
        print('done.')

        
        #  print('Getting labels, ', end='')
        #  sig['labels'] = self._get_labels(n_sig_events, 'sig')
        #  bg['labels'] = self._get_labels(n_bg_events, 'bg')
        #  print('done.')
        
        self._save_array(sig,bg)


    def _get_branches(self, structured_array, branches):
        """Get branches out of structured array and place them into a normal
        numpy ndarray. If the branch is vector-like, only keep the first four
        entries (jet variables);

        Arguments:
        ----------------
        structured_array (numpy structured array):
            Structured array converted from ROOT file.
        branches (list(str)):
            List of branches to extract from the structured array.

        Returns:
        ----------------
        ndarray (numpy ndarray):
            An array filled with the data.
        new_branches (list(str)):
            List of variables which ndarray is filled with. Each entry
            represents the corresponding column of the array.
        """

        # define vector-like variables
        
        jets = ['CSV', 'Jet_CSV', 'Jet_CosThetaStar_Lepton', 'Jet_CosTheta_cm',
                'Jet_Deta_Jet1', 'Jet_Deta_Jet2','Jet_Deta_Jet3',
                'Jet_Deta_Jet4','Jet_E','Jet_Eta','Jet_Flav','Jet_GenJet_Eta',
                'Jet_GenJet_Pt', 'Jet_M','Jet_PartonFlav', 'Jet_Phi',
                'Jet_PileUpID', 'Jet_Pt']

        ndarray = []
        new_branches = []

        for branch in branches:
            # if branch in jets:
            #     # only keep the first four entries of the jet vector
            #     array = [jet[:4] for jet in structured_array[branch]]
            #     ndarray.append(np.vstack(array))
            #     new_branches += [branch+'_{}'.format(i) for i in range(1,5)]
            # else:
            array = structured_array[branch].reshape(-1,1)
            ndarray.append(array)
            new_branches += [branch]
        return np.hstack(ndarray), new_branches


    def _get_labels(self, data_dict, structured_array, n_events, label):
        """Create labels.

        Arguments:
        ----------------
        n_events (int):
            Number of labels to be created.
        label (str):
            String indicating whether the events belong to the signal or to the
            background.

        Returns:
        ----------------
        labels (numpy ndarray):
            A numpy ndarray of shape (n_events, out_size) filled with the label.
        """

        labels = []
        if (label=='sig'):
            for i in range(n_events):
                labels.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            print('Created {} signal labels.'.format(len(labels)))
            print('Number of signal events without a label: {}'.format(n_events
                - len(labels)))
        else:
            for event in range(structured_array.shape[0]):
                # TTPlusBB = arr[event]['GenEvt_I_TTPlusBB']
                # TTPlusCC = arr[event]['GenEvt_I_TTPlusCC']
                if (structured_array[event]['GenEvt_I_TTPlusBB'] == 3 and
                        structured_array[event]['GenEvt_I_TTPlusCC'] == 0):
                    labels.append([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
                if (structured_array[event]['GenEvt_I_TTPlusBB'] == 2 and
                        structured_array[event]['GenEvt_I_TTPlusCC'] == 0):
                    labels.append([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
                if (structured_array[event]['GenEvt_I_TTPlusBB'] == 1 and
                        structured_array[event]['GenEvt_I_TTPlusCC'] == 0):
                    labels.append([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
                if (structured_array[event]['GenEvt_I_TTPlusBB'] == 0 and
                        structured_array[event]['GenEvt_I_TTPlusCC'] == 1):
                    labels.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
                if (structured_array[event]['GenEvt_I_TTPlusBB'] == 0 and
                        structured_array[event]['GenEvt_I_TTPlusCC'] == 0):
                    labels.append([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                else:
                    labels.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            n30 = 0
            n20 = 0
            n10 = 0
            n01 = 0
            nlight = 0
            for event in range(n_events):
                if (labels[event][1] == 1.0):
                    n30 += 1
                if (labels[event][2] == 1.0):
                    n20 += 1
                if (labels[event][3] == 1.0):
                    n10 += 1
                if (labels[event][4] == 1.0):
                    n01 += 1
                if (labels[event][5] == 1.0):
                    nlight += 1

            print('Created {} tt+bb labels.'.format(n30))
            print('Created {} tt+2b labels.'.format(n20))
            print('Created {} tt+b labels.'.format(n10))
            print('Created {} tt+cc labels.'.format(n01))
            print('Created {} light flavor labels.'.format(nlight))
            print('Number of events without label: {}'.format(n_events - 
                (n30+n20+n10+n01+nlight)))

        return labels
    
    
    def _get_category(self, data_dict, structured_array):
        """Checks if the data belongs to the given category. Only keep matching
        events.

        Arguments:
        ----------------
        data_dict (dict):
            Dictionary filled with event variables and corresponding weights.
        structured_array (numpy structured array):
            Structured array converted from ROOT file.
        """

        keep_events = []
        for event in range(structured_array.shape[0]):
            TTPlusBB = structured_array[event]['GenEvt_I_TTPlusBB']
            TTPlusCC = structured_array[event]['GenEvt_I_TTPlusCC']

            # print(range(len(self.categories)))
            for i in range(len(self.categories)):
                category = self.categories[i]
                if self._check_category(TTPlusBB, TTPlusCC, category):
                    keep_events.append(event)
                else:
                    continue

        # keep_dict = {'data': data_dict['data'][keep_events], 'weights': data_dict['weights'][keep_events]}
        keep_dict = {'data': [data_dict['data'][i] for i in keep_events],
                'weights': [data_dict['weights'][i] for i in keep_events],
                'labels': [data_dict['labels'][i] for i in keep_events]}

        return keep_dict


    def _check_category(self, TTPlusBB, TTPlusCC, name):
        """Returns category bool.

        Arguments:
        ----------------
        TTPlusBB (int): GenEvt_I_TTPlusBB
        TTPlusCC (int): GenEvt_I_TTPlusCC
        """

        category = {'30': (TTPlusBB == 3 and TTPlusCC == 0),
                    '20': (TTPlusBB == 2 and TTPlusCC == 0),
                    '10': (TTPlusBB == 1 and TTPlusCC == 0),
                    '01': (TTPlusBB == 0 and TTPlusCC == 1),
                    'light': (TTPlusBB == 0 and TTPlusCC == 0)}
        
        return category[name]

    
    def _get_weights(self, structured_array):
        """Calculate the weight for each event.

        For each event we will calculate: 
        Weight_XS * Weight_CSV * Weight_pu69p2
        Then the weights are normalized so that the sum over all weights is
        equal to 1.

        Arguments:
        ----------------
        structured_array (numpy structured array):
            Structured array converted from ROOT file.

        Returns:
        -----------------
        weights (numpy ndarray):
            An array of shape (-1,1) filled with the weight of each event.
        """

        weight_names = ['Weight_XS', 'Weight_CSV', 'Weight_pu69p2']
        weights, _ = self._get_branches(structured_array, weight_names)
        weights = np.prod(weights, axis=1).reshape(-1,1)
        weights /+ np.sum(weights)

        return weights


    # def _check_category_single(self


    def _save_array(self, sig, bg):
        """Stacks data and saves the array to the given path.

        Arguments:
        ----------------
        sig (dict):
            Dictionary containing signal events.
        bg (dict):
            Dictionary containing background events.
        """

        array_dir = self.save_path + '/' + self.arr_name + '.npy'
        print('Saving array to {}, '.format(array_dir), end='')
        sig_arr = np.hstack((sig['labels'], sig['data'], sig['weights']))
        bg_arr = np.hstack((bg['labels'], bg['data'], bg['weights']))

        ndarray = np.vstack((sig_arr, bg_arr))
        np.save(array_dir, ndarray)

        with open(self.save_path + '/branches.txt','w') as f:
            for branch in self.branches:
                f.write(branch + '\n')
        print('Done.')

    def _control_plot(self, sig, bg, branches):
        """Plot histograms of all variables.

        Arguments:
        ----------------
        sig (numpy ndarray):
            Array containing the signal data.
        bg (numpy ndarray):
            Array containing the background data.
        branches (list):
            List of variable names.
        """

        plot_dir = self.save_path + '/control_plots' + self.arr_name
        if not os.path.isdir(plot_dir):
            os.makedirs(plot_dir)

        for variable in range(sig.shape[1]):
            # get bin edges
            sig_min, bg_min = np.amin(sig[:, variable]), np.amin(bg[:,
                variable])
            sig_max, bg_max = np.amax(sig[:, variable]), np.amax(bg[:,
                variable])

            if (sig_min < bg_min):
                glob_min = sig_min
            else:
                glob_min = bg_min
            if (sig_max > bg_max):
                glob_max = sig_max
            else:
                glob_max = bg_max

            bin_edges = np.linspace(glob_min, glob_max, 30)

            n, bins, _ = plt.hist(sig[:, variable], bins = bin_edges, histtype =
                    'step', normed = True, label='Signal', color = 'black')
            n, bins, _ = plt.hist(bg[:, variable], bins = bin_edges, histtype =
                    'step', normed = True, label = 'Background', color = 'red',
                    ls='--')
            plt.ylabel('normed to unit area')
            plt.xlabel(branches[variable])
            plt.legend(loc='best', frameon=False)
            plt.savefig(plot_dir + '/' + branches[variable] + '.pdf')
            plt.savefig(plot_dir + '/' + branches[variable] + '.png')
            plt.savefig(plot_dir + '/' + branches[variable] + '.eps')
            plt.clf()
