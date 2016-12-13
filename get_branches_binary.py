from preprocessing.preprocessing import GetBranches

ttH_Even = '/storage/7/lang/ntuples/numpy/ttH_Even.npy'
ttH_Odd = '/storage/7/lang/ntuples/numpy/ttH_Odd.npy'
ttbarSL_Even = '/storage/7/lang/ntuples/numpy/ttbarSL_Even.npy'
ttbarSL_Odd = '/storage/7/lang/ntuples/numpy/ttbarSL_Odd.npy'

branchlist = 'branchlists/all_branches.txt'

get_branches = GetBranches('63', branchlist)
get_branches.process(ttH_Even, ttbarSL_Even, 'even1_binary') 
get_branches.process(ttH_Odd, ttbarSL_Odd, 'odd1_binary')
