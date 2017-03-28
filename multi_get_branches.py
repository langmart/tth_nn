from preprocessing.preprocessing_oneHot_multiconvert import GetBranches

loaddir = '/storage/7/lang/ntuples/numpy/'

ttH_Even = loaddir + 'ttH_Even.npy'
ttH_Odd = loaddir + 'ttH_Odd.npy'
ttbarSL_Even = loaddir + 'ttbarSL_Even.npy'
ttbarSL_Odd = loaddir + 'ttbarSL_Odd.npy'

# branchlists = ['branchlists/bdt.txt', 'branchlists/Jet_N_Evt.txt',
#         'branchlists/branches_new.txt', 'branchlists/bdt_evt_jets.txt',
#         'branchlists/bdt_and_weights.txt', 'branchlists/branches_corrected.txt']
branchlists = ['branchlists/branches_corrected.txt', 'branchlists/bdt.txt',
        'branchlists/branches_reduced.txt',
        'branchlists/branches_corrected_wo_reco.txt']
# branchlists = ['branchlists/branches_new.txt',
#         'branchlists/branches_corrected.txt', 'branchlists/bdt.txt',
#         'branchlists/branches_reduced.txt']
categories_list = [['30', '20', '10', '01', 'light'], ['30','light'], ['30',
    '20', '10', '01']]
# categories_list = [['30', '20', '10', '01', 'light'], ['30', 'light'], ['30', '20', '10', '01']]
# define categories for background
# '30': tt + bb
# '20': tt + 2b
# '10': tt + b
# '01': tt + cc
# 
preselection = 'no'
savedir='/storage/7/lang/nn_data/converted'
# Cross sections in femtobarn.
sigma_ttH_bb_SL = 85.4
sigma_ttbar_SL = 244100
get_branches = GetBranches(sigma_ttH_bb_SL, sigma_ttbar_SL)
get_branches.process(ttH_Even, ttbarSL_Even, arr_name='even_', savedir=savedir, 
        branchlists=branchlists, categories_list=categories_list,
        preselection=preselection)
get_branches.process(ttH_Odd, ttbarSL_Odd, arr_name='odd_', savedir=savedir, 
        branchlists=branchlists, categories_list=categories_list,
        preselection=preselection)
