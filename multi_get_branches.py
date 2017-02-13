from preprocessing.preprocessing_oneHot_multiconvert import GetBranches

loaddir = '/storage/7/lang/ntuples/numpy/'

ttH_Even = loaddir + 'ttH_Even.npy'
ttH_Odd = loaddir + 'ttH_Odd.npy'
ttbarSL_Even = loaddir + 'ttbarSL_Even.npy'
ttbarSL_Odd = loaddir + 'ttbarSL_Odd.npy'

# branchlists = ['branchlists/bdt.txt', 'branchlists/Jet_N_Evt.txt',
#         'branchlists/branches_new.txt', 'branchlists/bdt_evt_jets.txt',
#         'branchlists/bdt_and_weights.txt', 'branchlists/branches_corrected.txt']
branchlists = ['branchlists/branches_new.txt', 'branchlists/branches_corrected.txt', 'branchlists/bdt.txt']
categories_list = [['30', '20', '10', '01', 'light'], ['30', 'light'], ['30', '20', '10', '01']]
# define categories for background
# '30': tt + bb
# '20': tt + 2b
# '10': tt + b
# '01': tt + cc
# 
savedir='/storage/7/lang/nn_data/converted'
# get_branches = GetBranches('/storage/7/lang/nn_data', branchlist, categorylist, out_size)
get_branches = GetBranches()
get_branches.process(ttH_Even, ttbarSL_Even, arr_name='even_', savedir=savedir, 
        branchlists=branchlists, categories_list=categories_list)
get_branches.process(ttH_Odd, ttbarSL_Odd, arr_name='odd_', savedir=savedir, 
        branchlists=branchlists, categories_list=categories_list)
