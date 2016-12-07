import os
from tth_tensorflow.utils.root2array import Root2Array

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
save_directory = '/storage/7/lang/arrays/'
branchlist = 'branchlists/all_branches.txt'
tree_name = 'MVATree'

r2a = Root2Array(ttH_Even, 1.0)
r2a.convert(tree_name, branchlist, save_directory, 'ttH_Even')
print("Converted ttH_Even to {}.".format(save_directory))
r2a = Root2Array(ttbarSL_Even, 0.0)
r2a.convert(tree_name, branchlist, save_directory, 'ttbarSL_Even')
print("Converted ttH_Odd to {}.".format(save_directory))
r2a = Root2Array(ttH_Odd, 1.0)
r2a.convert(tree_name, branchlist, save_directory, 'ttH_Odd')
print("Converted ttbarSL_Even to {}.".format(save_directory))
r2a = Root2Array(ttbarSL_Odd, 0.0)
r2a.convert(tree_name, branchlist, save_directory, 'ttbarSL_Odd')
print("Converted ttbarSL_Odd to {}.".format(save_directory))
