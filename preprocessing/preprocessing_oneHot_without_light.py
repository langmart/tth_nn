from __future__ import absolute_import, division, print_function
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
#from root_numpy import root2array


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

    def __init__(self, savedir, branchlist, categories=['30','20','10','01'], out_size = 6):
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
        self.save_path = savedir + '/converted'

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

        # if (sig_branches == bg_branches):
        #     self.branches = sig_branches

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

        print('Found categories: {}.'.format(self.categories))

        print('Getting categories and labels, ')
        sign = self._get_categories_and_labels(sig, structured_sig,
                n_sig_events, 'sig')
        bgn = self._get_categories_and_labels(bg, structured_bg, n_bg_events,
        'bg')
        print('done.')

        self._save_array(sign,bgn)


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
            if branch in jets:
                # only keep the first four entries of the jet vector
                array = [jet[:4] for jet in structured_array[branch]]
                ndarray.append(np.vstack(array))
                new_branches += [branch+'_{}'.format(i) for i in range(1,5)]
            else:
                array = structured_array[branch].reshape(-1,1)
                ndarray.append(array)
                new_branches += [branch]
            # print('Appended {}.'.format(new_branches[-1]))
        return np.hstack(ndarray), new_branches

    
    def _get_categories_and_labels(self, data_dict, structured_array, n_events,
            label):
        """Collects events belonging to the categories and adds labels to them. 

        Arguments:
        ----------------
        data_dict (dict):
            Dictionary filled with event variables and corresponding weights.
        structured_array (numpy structured array):
            Structured array converted from ROOT file.
        n_events (int):
            Number of events.
        label (string):
            label indicating signal or background.

        Returns:
        ----------------
        keep_dict (dict):
            dictionary with the data to be kept.
        """

        keep_events = []
        keep_dict = dict()
        for i in ['data', 'weights', 'labels']:
            keep_dict[i] = []
        if (label == 'sig'):
            for event in range(structured_array.shape[0]):
                keep_events.append(event)
                keep_dict['data'].append(data_dict['data'][event])
                keep_dict['weights'].append(data_dict['weights'][event])
                siglab = signal_label()
                keep_dict['labels'].append(siglab)
            print('    Signal data: {}'.format(len(keep_dict['labels'])))
            print('    Signal events without label: {}'.format(len(keep_dict['labels']) - n_events))
        else:
            bg_disc = 0
            count_dict = {'30': 0, '20': 0, '10': 0, '01': 0}
            for event in range(structured_array.shape[0]):
                TTPlusBB = structured_array[event]['GenEvt_I_TTPlusBB']
                TTPlusCC = structured_array[event]['GenEvt_I_TTPlusCC']

                found_one_category = False

                for category in self.categories:
                    if self._check_category(TTPlusBB, TTPlusCC, category):
                        keep_events.append(event)
                        keep_dict['data'].append(data_dict['data'][event])
                        keep_dict['weights'].append(data_dict['weights'][event])
                        bglab = bg_label(category)
                        keep_dict['labels'].append(bglab)
                        count_dict[category] += 1
                        found_one_category = True
                if not found_one_category:
                    bg_disc += 1
            print('    Created {} tt + bb labels.'.format(count_dict['30']))
            print('    Created {} tt + 2b labels.'.format(count_dict['20']))
            print('    Created {} tt + b labels.'.format(count_dict['10']))
            print('    Created {} tt + cc labels.'.format(count_dict['01']))
            count_sum = count_dict['30'] + count_dict['20'] + count_dict['10'] + count_dict['01']
            print('    Total number of created background labels: {}'.format(count_sum))
            print('    Background data: {}'.format(len(keep_dict['labels'])))
            print('    Number of discarded background events: {}'.format(bg_disc))
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
                    '01': (TTPlusBB == 0 and TTPlusCC == 1)}
        
        return category[name]

    
    
    def _get_weights(self, structured_array):
        """Calculate the weight for each event.

        For each event we will calculate:
        Weight_XS
        out-dated: Weight_XS * Weight_CSV * Weight_pu69p2
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

        weight_names = ['Weight_XS']
        weights, _ = self._get_branches(structured_array, weight_names)
        weights = np.prod(weights, axis=1).reshape(-1,1)
        weights /+ np.sum(weights)

        return weights


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
        
        # print('Shape of sig[labels]: {}'.format(sig['labels'][1].shape))
        # print('Shape of sig[data]: {}'.format(sig['data'][1].shape))
        # print('Shape of sig[weights]: {}'.format(sig['weights'][1].shape))
        # print('Shape of bg[labels]: {}'.format(bg['labels'][1].shape))
        # print('Shape of bg[data]: {}'.format(bg['data'][1].shape))
        # print('Shape of bg[weights]: {}'.format(bg['weights'][1].shape))

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


def signal_label():
    return np.array([1.0, 0.0, 0.0, 0.0, 0.0])


def bg_label(category):
    if (category == '30'):
        return np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    elif (category == '20'):
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    elif (category == '10'):
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    elif (category == '01'):
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    else:
        sys.exit('Something went terribly wrong. There appears to be at least one event which has not been sorted out properly.')


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
