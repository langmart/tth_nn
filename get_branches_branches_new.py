from preprocessing.preprocessing_oneHot import GetBranches

loaddir = '/storage/7/lang/ntuples/numpy/'

ttH_Even = loaddir + 'ttH_Even.npy'
ttH_Odd = loaddir + 'ttH_Odd.npy'
ttbarSL_Even = loaddir + 'ttbarSL_Even.npy'
ttbarSL_Odd = loaddir + 'ttbarSL_Odd.npy'

# branchlist = 'branchlists/converted_2016-12-19.txt'
branchlist = 'branchlists/branches_new.txt'
# define categories for background
# '30': tt + bb
# '20': tt + 2b
# '10': tt + b
# '01': tt + cc
categorylist = ['30','20','10','01', 'light'] 
# 
out_size = 1 + len(categorylist)
print('Calculated dimension of output vector: {}'.format(out_size))
get_branches = GetBranches('/storage/7/lang/nn_data', branchlist, categorylist, out_size)
get_branches.process(ttH_Even, ttbarSL_Even, 'even1_branches_new_weights')
get_branches.process(ttH_Odd, ttbarSL_Odd, 'odd1_branches_new_weights')
