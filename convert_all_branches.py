import os
from tth_tensorflow.utils.root2numpy import Root2Numpy

ttH_Even_path = '/storage/a/welsch/ntuples/ttHbb/Even/'
ttH_Odd_path = '/storage/a/welsch/ntuples/ttHbb/Odd/'
ttbarSL_Even_path = '/storage/a/welsch/ntuples/ttbarSL/Even/'
ttbarSL_Odd_path = '/storage/a/welsch/ntuples/ttbarSL/Odd/'

ttH_Even = [ttH_Even_path + x for x in
               os.listdir(ttH_Even_path)]
ttH_Odd = [ ttH_Odd_path + x for x in
               os.listdir(ttH_Odd_path)]
ttbarSL_Even = background_dir1 = [ttbarSL_Even_path
    + x for x in os.listdir(ttbarSL_Even_path)]
ttbarSL_Odd = background_dir1 = [ttbarSL_Odd_path
    + x for x in os.listdir(ttbarSL_Odd_path)]


# only edit this
save_directory = '/storage/a/lang/arrays'
branchlist = 'branchlists/all_branches.txt'
treename = 'MVATree'

r2np = Root2Numpy(ttH_Even, treename)
r2np.convert(save_directory, 'ttH_Even')
r2np = Root2Numpy(ttbarSL_Even, treename)
r2np.convert(save_directory, 'ttbarSL_Even')
r2np = Root2Numpy(ttH_Odd, treename)
r2np.convert(save_directory, 'ttH_Odd')
r2np = Root2Numpy(ttbarSL_Odd, treename)
r2np.convert(save_directory, 'ttbarSL_Odd')
